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

"""Blip2Base model for MatterChat."""

import contextlib

import paddle
import paddle.nn as nn

from ppmat.models.matterchat.chgnet.model.model_embedding import CHGNet
from ppmat.models.matterchat.q_former.q_former_base import BertConfig, BertLMHeadModel


class LayerNorm(nn.LayerNorm):
    """Subclass of paddle LayerNorm to support mixed precision."""

    def forward(self, x):
        orig_dtype = x.dtype
        x = super().forward(x.cast(paddle.float32))
        return x.cast(orig_dtype)


class Blip2Base(nn.Layer):
    """Base class for Blip2 models."""

    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        """Initialize tokenizer. Returns None - tokenizer is managed externally."""
        return None

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        config = BertConfig()
        config.encoder_width = vision_width
        config.add_cross_attention = True
        config.cross_attention_freq = cross_attention_freq
        config.query_length = num_query_token

        Qformer = BertLMHeadModel(config)

        query_tokens = paddle.create_parameter(
            shape=[1, num_query_token, config.hidden_size],
            dtype=paddle.float32,
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=config.initializer_range
            ),
        )
        return Qformer, query_tokens

    def init_material_encoder(self):
        """Initialize CHGNet material encoder without pretrained weights."""
        return CHGNet()

    def maybe_autocast(self, dtype=paddle.float16):
        """Context manager for mixed precision."""
        if not paddle.is_compiled_with_cuda():
            return contextlib.nullcontext()
        return paddle.amp.auto_cast(dtype=dtype)
