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

"""
CSPNet model for crystal structure prediction.

This module is migrated from OMG (Open Materials Generation).
Original code: models.diffcsp.cspnet
"""

import math

import paddle
from paddle_scatter import scatter

from .utils import radius_graph_pbc, repeat_blocks, SinusoidalTimeEmbeddings

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(paddle.nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat([emb.sin(), emb.cos()], axis=-1)
        return emb


class CSPLayer(paddle.nn.Layer):
    """Message passing layer for cspnet."""

    def __init__(
        self, hidden_dim=128, act_fn=None, dis_emb=None, ln=False, ip=True
    ):
        super().__init__()
        if act_fn is None:
            act_fn = paddle.nn.Silu()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            paddle.nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            paddle.nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = paddle.nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
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
        agg = scatter(
            edge_features,
            edge_index[0],
            dim=0,
            reduce='mean',
            dim_size=node_features.shape[0],
        )
        agg = paddle.concat([node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(paddle.nn.Layer):
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
        pred_scalar=False,
        time_embed_dim=None,  # If None, don't use time embedding
    ):
        super().__init__()
        self.ip = ip
        self.smooth = smooth
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        self.species_shift = 1  # Offset for species indexing
        if self.smooth:
            self.node_embedding = paddle.nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = paddle.nn.Embedding(max_atoms, hidden_dim)
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_embedder = SinusoidalTimeEmbeddings(time_embed_dim)
            self.atom_latent_emb = paddle.nn.Linear(hidden_dim + time_embed_dim, hidden_dim)
        else:
            self.time_embedder = None
            self.atom_latent_emb = paddle.nn.Linear(hidden_dim + 1, hidden_dim)
        
        if act_fn == 'silu':
            self.act_fn = paddle.nn.Silu()
        else:
            self.act_fn = paddle.nn.Silu()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs, n_space=3)
            dis_dim = num_freqs * 2 * 3
        elif dis_emb == 'none':
            self.dis_emb = None
            dis_dim = 0
        else:
            self.dis_emb = None
            dis_dim = 0
        self.dis_dim = dis_dim
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i,
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        self.num_layers = num_layers
        self.coord_out = paddle.nn.Linear(hidden_dim, 3, bias_attr=False)
        self.coord_out_2 = paddle.nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out = paddle.nn.Linear(hidden_dim, 9, bias_attr=False)
        self.lattice_out_2 = paddle.nn.Linear(hidden_dim, 9, bias_attr=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = paddle.nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = paddle.nn.Linear(hidden_dim, max_atoms)
            self.type_out_2 = paddle.nn.Linear(hidden_dim, max_atoms)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = paddle.nn.Linear(hidden_dim, 1)

    def enable_masked_species(self) -> None:
        """
        Enable a masked species (with token 0) in the encoder.

        For De Novo Generation (DNG) with DiscreteFlowMatchingMask,
        token 0 is reserved as a special "masked" state.
        The embedding vocabulary must be expanded to max_atoms + 1
        to handle this additional token, while type_out remains at max_atoms.

        Original code: omg/model/encoders/cspnet_full.py enable_masked_species()
        """
        self.node_embedding = paddle.nn.Embedding(self.max_atoms + 1, self.hidden_dim)
        self.species_shift = 0

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat([tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        """
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms = mask_same_atoms & cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand([2, -1])].reshape([2, -1])
        edge_index_cat = paddle.concat(
            [
                edge_index_new,
                paddle.stack([edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            paddle.arange(neighbors.size(0)), neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(batch_edge, minlength=neighbors.size(0))
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )
        return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == 'fc':
            lis = [paddle.ones([n, n]) for n in num_atoms.tolist()]
            fc_graph = paddle.block_diag(lis)
            # Convert to edge index using dense_to_sparse logic
            rows, cols = [], []
            offset = 0
            for i, n in enumerate(num_atoms.tolist()):
                for j in range(n):
                    for k in range(n):
                        rows.append(j + offset)
                        cols.append(k + offset)
                offset += n
            fc_edges = paddle.stack([paddle.to_tensor(rows), paddle.to_tensor(cols)])
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        elif self.edge_style == 'knn':
            lattice_nodes = lattices[node2graph]
            cart_coords = paddle.einsum('bi,bij->bj', frac_coords, lattice_nodes)
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                None,
                None,
                num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=num_atoms.place,
                lattices=lattices,
            )
            j_index, i_index = edge_index[0], edge_index[1]
            distance_vectors = frac_coords[j_index] - frac_coords[i_index]
            distance_vectors = distance_vectors + to_jimages
            edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
                edge_index, to_jimages, num_bonds, distance_vectors
            )
            return edge_index_new, -edge_vector_new

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph, prop=None):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - self.species_shift)
        
        # Apply time embedding if configured, otherwise use raw time
        if self.time_embed_dim is not None:
            t_embed = self.time_embedder(t)
        else:
            t_embed = t
        
        # Handle both 1D (single sample) and 2D (batched) time input
        # repeat_interleave needs proper handling for single vs multi-graph batches
        total_atoms = int(num_atoms.sum())  # Total number of atoms across all graphs
        if t_embed.ndim == 1:
            t_embed = t_embed.unsqueeze(0)  # Add batch dim if needed
        t_per_atom = t_embed.repeat_interleave(num_atoms, axis=0)  # Repeat for each atom
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        coord_out_2 = self.coord_out_2(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce='mean')

        if self.pred_scalar:
            return self.scalar_out(graph_features)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.einsum('bij,bjk->bik', lattice_out, lattices)
        lattice_out_2 = self.lattice_out_2(graph_features)
        lattice_out_2 = lattice_out_2.reshape([-1, 3, 3])
        if self.ip:
            lattice_out_2 = paddle.einsum('bij,bjk->bik', lattice_out_2, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            type_out_2 = self.type_out_2(node_features)
            return lattice_out, coord_out, type_out, lattice_out_2, coord_out_2, type_out_2

        return lattice_out, coord_out
