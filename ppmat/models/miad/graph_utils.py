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
Graph utility functions for CSPNet.
Converted from PyTorch Geometric to PaddlePaddle.
"""

import paddle
import paddle.nn.functional as F


def to_dense_adj(edge_index, edge_attr=None, max_num_nodes=None):
    """
    Convert edge index to dense adjacency matrix.

    Args:
        edge_index: Edge indices of shape (2, num_edges)
        edge_attr: Optional edge attributes of shape (num_edges, num_edge_features)
        max_num_nodes: Maximum number of nodes (if None, inferred from edge_index)

    Returns:
        adj: Dense adjacency matrix of shape (batch_size, max_num_nodes, max_num_nodes)
               or (max_num_nodes, max_num_nodes) if edge_attr is None
    """
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, num_edges), got {edge_index.shape}")

    num_edges = edge_index.shape[1]

    if max_num_nodes is None:
        max_num_nodes = int(edge_index.max()) + 1

    # Determine batch size from edge_attr if provided
    if edge_attr is not None:
        batch_size = edge_attr.shape[0]
        adj_shape = (batch_size, max_num_nodes, max_num_nodes)
    else:
        batch_size = 1
        adj_shape = (max_num_nodes, max_num_nodes)

    # Initialize adjacency matrix with zeros
    if edge_attr is not None:
        num_edge_features = edge_attr.shape[-1]
        adj = paddle.zeros(adj_shape + (num_edge_features,), dtype=edge_attr.dtype)
    else:
        adj = paddle.zeros(adj_shape, dtype='float32')

    # Fill adjacency matrix
    if batch_size == 1:
        # Single graph case
        for i in range(num_edges):
            source, target = int(edge_index[0, i]), int(edge_index[1, i])
            if edge_attr is not None:
                adj[source, target] = edge_attr[0, i] if edge_attr.ndim > 1 else edge_attr[i]
            else:
                adj[source, target] = 1.0
    else:
        # Batch case - need to handle batching
        # This requires additional batch information
        # For now, implement simple version
        for i in range(num_edges):
            source, target = int(edge_index[0, i]), int(edge_index[1, i])
            if edge_attr is not None:
                # This needs batch assignment info
                adj[:, source, target] = edge_attr[i] if edge_attr.ndim > 1 else edge_attr[i]
            else:
                adj[:, source, target] = 1.0

    return adj


def dense_to_sparse(adj):
    """
    Convert dense adjacency matrix to edge index and edge attributes.

    Args:
        adj: Dense adjacency matrix of shape (num_nodes, num_nodes) or
             (batch_size, num_nodes, num_nodes)

    Returns:
        edge_index: Edge indices of shape (2, num_edges)
        edge_attr: Edge attributes (optional)
    """
    if adj.ndim == 2:
        # Single graph case: use nonzero for vectorized extraction
        nonzero = paddle.nonzero(adj.cast('float32'))
        if nonzero.shape[0] == 0:
            return paddle.zeros([2, 0], dtype='int64'), paddle.zeros([0], dtype='float32')
        edge_index = nonzero.t()
        return edge_index, paddle.ones([edge_index.shape[1]], dtype='float32')

    # Batch case
    batch_size, num_nodes, _ = adj.shape
    edge_indices_list = []
    edge_attrs_list = []

    for b in range(batch_size):
        adj_b = adj[b]
        nonzero = paddle.nonzero(adj_b.cast('float32'))
        if nonzero.shape[0] == 0:
            continue
        edge_indices_list.append(nonzero.t())
        edge_attrs_list.append(paddle.ones([nonzero.shape[0]], dtype='float32'))

    if not edge_indices_list:
        return paddle.zeros([2, 0], dtype='int64'), paddle.zeros([0], dtype='float32')

    edge_index = paddle.concat(edge_indices_list, axis=1)
    edge_attr = paddle.concat(edge_attrs_list, axis=0)
    return edge_index, edge_attr


def block_diag(*inputs):
    """
    Create a block diagonal matrix from input tensors.

    Args:
        *inputs: Variable number of 2D tensors (square matrices for fc graph)

    Returns:
        output: Block diagonal matrix of shape (total_size, total_size)
    """
    # Get dimensions
    sizes = [tensor.shape[0] for tensor in inputs]
    total_size = sum(sizes)

    # All inputs should be square matrices (n, n)
    # Output should be (total_size, total_size)
    output = paddle.zeros([total_size, total_size], dtype=inputs[0].dtype)

    # Fill block diagonal
    row_offset = 0
    for tensor in inputs:
        rows, cols = tensor.shape
        output[row_offset:row_offset + rows, row_offset:row_offset + cols] = tensor
        row_offset += rows

    return output


def segment_csr(data, indptr, reduce="sum"):
    """
    Compute reduction over segments using CSR format.

    Args:
        data: Input data of shape (N, *)
        indptr: Index pointers of shape (M+1,)
        reduce: Reduction operation ('sum', 'mean', 'max', 'min')

    Returns:
        output: Reduced data of shape (M, *)
    """
    if indptr[0] != 0:
        raise ValueError("indptr must start with 0")

    num_segments = len(indptr) - 1
    output_shape = (num_segments,) + data.shape[1:]

    if reduce == "sum":
        output = paddle.zeros(output_shape, dtype=data.dtype)
        for i in range(num_segments):
            start, end = int(indptr[i]), int(indptr[i + 1])
            if start < end:
                output[i] = paddle.sum(data[start:end], axis=0)
    elif reduce == "mean":
        output = paddle.zeros(output_shape, dtype=data.dtype)
        for i in range(num_segments):
            start, end = int(indptr[i]), int(indptr[i + 1])
            if start < end:
                output[i] = paddle.mean(data[start:end], axis=0)
    elif reduce == "max":
        output = paddle.zeros(output_shape, dtype=data.dtype)
        for i in range(num_segments):
            start, end = int(indptr[i]), int(indptr[i + 1])
            if start < end:
                output[i] = paddle.max(data[start:end], axis=0)
    elif reduce == "min":
        output = paddle.zeros(output_shape, dtype=data.dtype)
        for i in range(num_segments):
            start, end = int(indptr[i]), int(indptr[i + 1])
            if start < end:
                output[i] = paddle.min(data[start:end], axis=0)
    else:
        raise ValueError(f"Unknown reduction: {reduce}")

    return output


def coalesce(edge_index, edge_attr=None):
    """
    Coalesce duplicate edges by summing their attributes.

    Args:
        edge_index: Edge indices of shape (2, num_edges)
        edge_attr: Optional edge attributes of shape (num_edges, num_edge_features)

    Returns:
        edge_index: Coalesced edge indices
        edge_attr: Coalesced edge attributes
    """
    if edge_index.shape[1] == 0:
        return edge_index, edge_attr

    # Create unique edge keys
    edge_keys = edge_index[0] * paddle.max(edge_index[1]) + edge_index[1]

    # Sort by edge keys
    perm = paddle.argsort(edge_keys)
    edge_index_sorted = edge_index[:, perm]

    if edge_attr is not None:
        edge_attr_sorted = edge_attr[perm]
    else:
        edge_attr_sorted = paddle.ones(edge_index.shape[1], dtype='float32')

    # Find unique edges and their counts
    unique_mask = paddle.concat([
        paddle.to_tensor([True], dtype='bool'),
        edge_keys[perm[1:]] != edge_keys[perm[:-1]]
    ])

    edge_index_coalesced = edge_index_sorted[:, unique_mask]

    if edge_attr is not None:
        # Sum attributes for duplicate edges
        num_edges = edge_index.shape[1]
        num_unique = int(unique_mask.sum())
        num_edge_features = edge_attr_sorted.shape[-1] if edge_attr_sorted.ndim > 1 else 1

        if edge_attr_sorted.ndim > 1:
            edge_attr_coalesced = paddle.zeros([num_unique, num_edge_features], dtype=edge_attr_sorted.dtype)
        else:
            edge_attr_coalesced = paddle.zeros([num_unique], dtype=edge_attr_sorted.dtype)

        idx = 0
        for i in range(num_unique):
            # Find all duplicate edges
            if i < num_unique - 1:
                end = perm[paddle.nonzero(~unique_mask[i + 1:])[0] + i + 1]
                if len(end) > 0:
                    end_idx = end[0] + 1
                else:
                    end_idx = i + 1
            else:
                end_idx = num_edges

            if edge_attr_sorted.ndim > 1:
                edge_attr_coalesced[i] = paddle.sum(edge_attr_sorted[idx:end_idx], axis=0)
            else:
                edge_attr_coalesced[i] = paddle.sum(edge_attr_sorted[idx:end_idx])

            idx = end_idx

        return edge_index_coalesced, edge_attr_coalesced
    else:
        return edge_index_coalesced, None
