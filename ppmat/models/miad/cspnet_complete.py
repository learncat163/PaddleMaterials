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

"""
CSPNet (Crystal Structure Prediction Network) for MiAD model.
"""

import paddle
import paddle.nn as nn
import math
from paddle_scatter import scatter

from ppmat.models.miad.graph_utils import (
    dense_to_sparse,
    block_diag,
)

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(nn.Layer):
    """Sinusoidal position embedding for fractional coordinates."""

    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        frequencies = 2 * math.pi * paddle.arange(self.n_frequencies)
        self.register_buffer('frequencies', frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)
        return emb.detach()


class CSPLayer(nn.Layer):
    """Message passing layer for CSPNet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=None,
        dis_emb=None,
        ln=False,
        ip=True
    ):
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if act_fn is None:
            act_fn = nn.SiLU()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose([0, 2, 1])
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.reshape([-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat([hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim=0, reduce='mean', dim_size=node_features.shape[0])
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


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    assert len(sizes.shape) == 1
    assert paddle.all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not paddle.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = paddle.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, paddle.Tensor):
            repeats = paddle.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, paddle.Tensor):
            repeat_inc = paddle.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, paddle.Tensor):
        assert paddle.all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = paddle.ones([1], dtype=sizes.dtype)
            zero = paddle.zeros([1], dtype=sizes.dtype)
            sizes = paddle.concat([one, sizes])
            repeats = paddle.concat([one, repeats])
            if isinstance(block_inc, paddle.Tensor):
                block_inc = paddle.concat([zero, block_inc])
            if isinstance(repeat_inc, paddle.Tensor):
                repeat_inc = paddle.concat([zero, repeat_inc])
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = paddle.repeat_interleave(
        paddle.arange(len(sizes)), repeats
    )

    # Get total size of output array
    N = paddle.sum(sizes * repeats)

    # Initialize indexing array
    id_ar = paddle.ones([N], dtype='int64')
    id_ar[0] = 0
    insert_index = paddle.cumsum(sizes[r1[:-1]], axis=0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, paddle.Tensor) and paddle.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = paddle.concat([paddle.zeros([1], dtype=sizes.dtype), paddle.cumsum(diffs, axis=0)])
        if continuous_indexing:
            # Now segment_csr is available from graph_utils
            if isinstance(block_inc, paddle.Tensor):
                insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")
            else:
                # Simplified version for scalar block_inc
                insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, paddle.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, paddle.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, paddle.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index
    id_ar[0] += start_idx

    # Finally index into input array
    res = paddle.cumsum(id_ar, axis=0)
    return res


class CSPNet(nn.Layer):
    """Crystal Structure Prediction Network."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_layers=4,
        max_atoms=100,
        act_fn='silu',
        dis_emb='sin',
        num_freqs=10,
        edge_style='fc',
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        model_name='CSPNet'
    ):
        super(CSPNet, self).__init__()
        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias_attr=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == 'fc':
            # Fully connected graph within each crystal
            # Use our implemented block_diag function
            lis = [paddle.ones([n, n], dtype=num_atoms.dtype) for n in num_atoms]
            fc_graph = block_diag(*lis)
            fc_edges, _ = dense_to_sparse(fc_graph)
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])

    def forward(self, t, t_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        # Handle both discrete atom types and one-hot encoding
        # Discrete: atom_types shape (total_atoms,) with values in [1, max_atoms)
        # One-hot: atom_types shape (total_atoms, num_types)
        if atom_types.ndim > 1:
            # One-hot input: for smooth=True Linear model, directly use one-hot input
            if self.smooth:
                # Linear layer expects one-hot [N, 100] as input
                node_features = self.node_embedding(atom_types.cast('float32'))
            else:
                # Embedding expects indices
                atom_indices = atom_types.argmax(axis=-1)
                node_features = self.node_embedding(atom_indices.cast('int64'))
        else:
            if self.smooth:
                node_features = self.node_embedding(atom_types.cast('float32'))
            else:
                node_features = self.node_embedding(atom_types - 1)

        t_per_atom = paddle.repeat_interleave(t_emb, num_atoms, axis=0)
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])

        if self.ip:
            lattice_out = paddle.einsum('bij,bjk->bik', lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out

        return lattice_out, coord_out
