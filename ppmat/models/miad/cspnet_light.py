# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import paddle.nn as nn


class SinusoidsEmbeddingLight(nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        frequencies = 2 * math.pi * paddle.arange(self.n_frequencies)
        self.register_buffer('frequencies', frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        shape = x.shape[:-1]
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        return emb.reshape(shape + (self.dim,))


class MessagePassing(nn.Layer):
    def __init__(self, hidden_dim, frac_freq):
        super().__init__()
        self.lat_embed_dim = 9
        self.frac_embed_dim = 6 * frac_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + self.lat_embed_dim + self.frac_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, node_embed, edge_embed, graph_embed):
        bs, N_m, h_dim = node_embed.shape
        l_dim, f_dim = self.lat_embed_dim, self.frac_embed_dim

        src = node_embed.unsqueeze(2).tile([1, 1, N_m, 1]).reshape([bs, N_m * N_m, h_dim])
        dst = node_embed.unsqueeze(1).tile([1, N_m, 1, 1]).reshape([bs, N_m * N_m, h_dim])
        graph = graph_embed.unsqueeze(1).tile([1, N_m * N_m, 1]).reshape([bs, N_m * N_m, l_dim])
        edge = edge_embed.reshape([bs, N_m * N_m, f_dim])

        fc_graph_edges = paddle.concat([src, dst, graph, edge], axis=-1)
        fc_graph_edges = fc_graph_edges.reshape([bs, N_m, N_m, 2 * h_dim + l_dim + f_dim])

        edge_msg = self.mlp(fc_graph_edges)
        node_msg = edge_msg.mean(axis=2)
        return node_msg


class MLPBlock(nn.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, node_embed, node_msg):
        h = self.mlp(paddle.concat([node_embed, node_msg], axis=-1))
        return h


class CSPNetLightBlock(nn.Layer):
    def __init__(self, hidden_dim, frac_freq):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.message_passing = MessagePassing(hidden_dim, frac_freq)
        self.mlp = MLPBlock(hidden_dim)

    def forward(self, node_embed, edge_embed, graph_embed):
        normed_node_embed = self.norm(node_embed)
        node_msg = self.message_passing(normed_node_embed, edge_embed, graph_embed)
        unnormed_node_embed = normed_node_embed + self.mlp(normed_node_embed, node_msg)
        return unnormed_node_embed


class CSPNetLight(nn.Layer):
    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        num_layers=4,
        max_atoms=100,
        num_freqs=10,
        **kwargs
    ):
        super().__init__()
        self.N_types = max_atoms
        self.hidden_dim = hidden_dim
        self.num_blocks = num_layers
        self.frac_freq = num_freqs
        self.time_dim = time_dim

        self.atom_embedding = nn.Linear(self.N_types, self.hidden_dim)
        self.frac_diff_encoding = SinusoidsEmbeddingLight(n_frequencies=self.frac_freq, n_space=3)
        self.node_embedding = nn.Linear(hidden_dim + self.time_dim, self.hidden_dim)

        self.blocks = nn.LayerList([
            CSPNetLightBlock(self.hidden_dim, self.frac_freq) for _ in range(self.num_blocks)
        ])
        self.final_layer_norm = nn.LayerNorm(self.hidden_dim)

        self.lattice_predictor = nn.Linear(self.hidden_dim, 9)
        self.fractional_predictor = nn.Linear(self.hidden_dim, 3)
        self.atom_predictor = nn.Linear(self.hidden_dim, self.N_types, bias_attr=False)

    def forward(self, t, t_embed, atom_types, fractional, lattice, num_atoms, node2graph):
        bs, h_dim = t.shape[0], self.hidden_dim
        N_m = atom_types.shape[0] // bs

        lattice_embed = (lattice @ lattice.transpose([0, 2, 1])).reshape([bs, 9])

        f = fractional.reshape([bs, N_m, 3])
        frac_diff = f.unsqueeze(1) - f.unsqueeze(2)
        frac_embed = self.frac_diff_encoding(frac_diff)

        atom_embed = self.atom_embedding(atom_types.reshape([bs, N_m, self.N_types]))
        t_per_atom = paddle.repeat_interleave(t_embed, num_atoms, axis=0).reshape([bs, N_m, -1])
        node_features = paddle.concat([atom_embed, t_per_atom], axis=-1)
        node_embed = self.node_embedding(node_features)

        for block in self.blocks:
            node_embed = block(node_embed=node_embed, edge_embed=frac_embed, graph_embed=lattice_embed)

        node_embed = self.final_layer_norm(node_embed)

        lattice_pred = self.lattice_predictor(node_embed.mean(axis=1))
        lattice_mean = paddle.einsum('bij,bjk->bik', lattice_pred.reshape([bs, 3, 3]), lattice)

        fractional_pred = self.fractional_predictor(node_embed)
        fractional_shift = fractional_pred.reshape([bs * N_m, 3])

        atom_pred = self.atom_predictor(node_embed)
        atom_type = atom_pred.reshape([bs * N_m, self.N_types])

        return lattice_mean, fractional_shift, atom_type
