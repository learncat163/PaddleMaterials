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

"""Base model classes for MatterChat."""

import os

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.matterchat.utils.weight_init import get_device
from ppmat.utils import logger
from ppmat.utils.misc import all_gather


class BaseModel(nn.Layer):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return get_device(self)

    def load_checkpoint(self, url_or_filename):
        """Load from a finetuned checkpoint."""
        if os.path.isfile(url_or_filename):
            checkpoint = paddle.load(url_or_filename)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.set_state_dict(state_dict)
        logger.info("load checkpoint from %s" % url_or_filename)

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot


class BaseEncoder(nn.Layer):
    """Base class for primitive encoders."""

    def __init__(self):
        super().__init__()

    def forward_features(self, samples, **kwargs):
        raise NotImplementedError

    @property
    def device(self):
        return get_device(self)


class GatherLayer(paddle.autograd.PyLayer):
    """Gather tensors from all workers with backward propagation support."""

    @staticmethod
    def forward(ctx, x):
        if not paddle.distributed.is_initialized():
            return (x,)
        world_size = paddle.distributed.get_world_size()
        output = [paddle.zeros_like(x) for _ in range(world_size)]
        paddle.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if not paddle.distributed.is_initialized():
            return grads[0]
        all_gradients = paddle.stack(list(grads))
        paddle.distributed.all_reduce(all_gradients)
        return all_gradients[paddle.distributed.get_rank()]


@paddle.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors."""
    return all_gather(tensor, concat=True, axis=0)


def tile(x, dim, n_tile):
    init_dim = x.shape[dim]
    repeat_idx = [1] * len(x.shape)
    repeat_idx[dim] = n_tile
    x = x.tile(repeat_idx)
    order_index = paddle.to_tensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]),
        dtype=paddle.int64,
    )
    idx = order_index.reshape([1, -1]).expand([x.shape[0], -1])
    return paddle.gather(x, dim, idx)
