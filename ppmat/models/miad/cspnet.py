# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import math
from paddle_scatter import scatter

class SinusoidsEmbedding(nn.Layer):
    def __init__(self, n_frequencies=128, n_space=3):
        super(SinusoidsEmbedding, self).__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(self.n_frequencies, dtype='float32')
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = paddle.reshape(emb, [-1, self.n_frequencies * self.n_space])
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        return emb.detach()

class CSPLayer(nn.Layer):
    def __init__(self, hidden_dim=128, act_fn=None, dis_emb=None, ln=False, ip=True):
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if act_fn is None:
            act_fn = nn.SiLU()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        hi = node_features[edge_index[0]]
        hj = node_features[edge_index[1]]
        if frac_diff is None:
            xi = frac_coords[edge_index[0]]
            xj = frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose([0, 2, 1])
        else:
            lattice_ips = lattices
        lattice_ips_flatten = paddle.reshape(lattice_ips, [-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat([hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim=0, dim_size=node_features.shape[0], reduce='mean')
        agg = paddle.concat([node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output

class CSPNet(nn.Layer):
    def __init__(self, num_atom_types=100, hidden_channels=512, num_layers=6, cutoff=6.0, max_num_atoms=25):
        super(CSPNet, self).__init__()
        self.num_atom_types = num_atom_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Create distance embedding
        self.dis_emb = SinusoidsEmbedding(n_frequencies=128, n_space=3)

        self.node_embedding = nn.Linear(100, hidden_channels, bias_attr=True)
        self.atom_latent_emb = nn.Linear(768, hidden_channels, bias_attr=True)
        for i in range(num_layers):
            self.add_module(f"csp_layer_{i}", CSPLayer(hidden_dim=hidden_channels, dis_emb=self.dis_emb, ln=True, ip=True))
        self.final_layer_norm = nn.LayerNorm(hidden_channels)
        self.coord_out = nn.Linear(hidden_channels, 3, bias_attr=False)
        self.lattice_out = nn.Linear(hidden_channels, 9, bias_attr=False)
        self.type_out = nn.Linear(hidden_channels, num_atom_types, bias_attr=True)

    def forward(self, t, t_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        """
        Forward pass of CSPNet.

        Args:
            t: timestep scalar
            t_emb: [batch_size, 256] time embeddings
            atom_types: [num_atoms_total, num_atom_types] one-hot atom features
            frac_coords: [num_atoms_total, 3] fractional coordinates
            lattices: [batch_size, 3, 3] lattice matrices
            num_atoms: [batch_size] number of atoms per crystal
            node2graph: [num_atoms_total] mapping from node to graph index
        """
        batch_size = lattices.shape[0]

        # Node embedding: one-hot atom types -> hidden features
        h = self.node_embedding(atom_types)

        # Concatenate time embeddings and project
        t_per_atom = paddle.repeat_interleave(t_emb, num_atoms, axis=0)
        h = paddle.concat([h, t_per_atom], axis=1)
        h = self.atom_latent_emb(h)

        # Generate fully-connected edges
        edge_index, edge2graph = self._gen_fc_edges(num_atoms, node2graph)

        # Message passing through CSP layers
        for i in range(self.num_layers):
            csp_layer = getattr(self, f"csp_layer_{i}")
            h = csp_layer(h, frac_coords, lattices, edge_index, edge2graph)

        h = self.final_layer_norm(h)

        # Coordinate prediction per atom
        coord_pred = self.coord_out(h)

        # Lattice prediction: aggregate node features per graph
        graph_features = scatter(h, node2graph, dim=0, dim_size=batch_size, reduce='mean')
        lattice_pred = paddle.reshape(self.lattice_out(graph_features), [batch_size, 3, 3])

        # Apply inner product transformation
        lattice_pred = paddle.einsum('bij,bjk->bik', lattice_pred, lattices)

        # Type prediction per atom
        type_pred = self.type_out(h)

        return lattice_pred, coord_pred, type_pred

    def _gen_fc_edges(self, num_atoms, node2graph):
        """Generate fully-connected edges within each graph."""
        src_list, dst_list, graph_list = [], [], []
        offset = 0
        for g in range(len(num_atoms)):
            n = num_atoms[g].item()
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src_list.append(offset + i)
                        dst_list.append(offset + j)
                        graph_list.append(g)
            offset += n
        edge_index = paddle.to_tensor([src_list, dst_list], dtype='int64')
        edge2graph = paddle.to_tensor(graph_list, dtype='int64')
        return edge_index, edge2graph

    @staticmethod
    def load_pytorch_weights(checkpoint_path):
        """
        Load pretrained PyTorch weights with proper transpose for PaddlePaddle.

        Args:
            checkpoint_path: Path to converted PaddlePaddle checkpoint (.pdparams)

        Returns:
            CSPNet model with loaded weights

        Note:
            PyTorch and PaddlePaddle use different Linear weight conventions:
            - PyTorch: weight.shape = (out_features, in_features)
            - PaddlePaddle: weight.shape = (in_features, out_features)
            This function transposes Linear layer weights during loading.
        """
        # Create default model instance
        model = CSPNet(num_atom_types=100, hidden_channels=512, num_layers=6)

        # Load checkpoint
        checkpoint = paddle.load(checkpoint_path)
        model_state = model.state_dict()

        # Load weights with transpose for Linear layers
        for key in checkpoint.keys():
            if key not in model_state:
                continue

            ckpt_param = checkpoint[key]
            ckpt_shape = ckpt_param.shape
            model_shape = model_state[key].shape

            # Check if this is a Linear layer weight (2D, not LayerNorm)
            is_linear_weight = (
                key.endswith('.weight') and
                not key.endswith('layer_norm.weight') and
                len(ckpt_shape) == 2 and
                len(model_shape) == 2
            )

            if is_linear_weight:
                # Transpose Linear weight: (out, in) -> (in, out)
                model_state[key] = ckpt_param.T
            elif ckpt_shape == model_shape:
                # Direct copy for bias, LayerNorm, etc.
                model_state[key] = ckpt_param

        # Set the modified state dict
        model.set_state_dict(model_state)

        return model
