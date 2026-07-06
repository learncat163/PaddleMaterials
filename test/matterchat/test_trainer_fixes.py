# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for matterchat trainer fixes.

Covers the fixes applied in the merge-cleanup commit:
- LoRA dtype follows base weight (not hardcoded float16)
- merge_into_base correctness (no erroneous .T transpose)
- Stage 1 raises NotImplementedError (not silent zero loss)
- apply_lora_to_mistral cleanup (no dead parent_name variable)
"""

import os

import numpy as np
import paddle
import paddle.nn as nn
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ppmat.models.matterchat.trainer import (
    LoRALinear,
    MISTRAL_LORA_TARGETS,
    apply_lora_to_mistral,
)


# =============================================================================
# LoRA dtype fix
# =============================================================================


class TestLoRADtype:
    """Verify LoRA parameters follow base linear dtype instead of float16."""

    def test_dtype_follows_base_fp32(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        assert lora.lora_A.dtype == paddle.float32
        assert lora.lora_B.dtype == paddle.float32
        assert lora.lora_A.dtype == base.weight.dtype

    def test_dtype_follows_base_fp16(self):
        # Build a base linear whose weight is genuinely float16 by setting
        # the parameter dtype directly (astype returns a copy and may not
        # mutate the original in-place depending on Paddle version).
        base = nn.Linear(64, 32)
        base.weight = paddle.create_parameter(
            shape=base.weight.shape, dtype=paddle.float16
        )
        if base.bias is not None:
            base.bias = paddle.create_parameter(
                shape=base.bias.shape, dtype=paddle.float16
            )
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        assert lora.lora_A.dtype == paddle.float16
        assert lora.lora_B.dtype == paddle.float16

    def test_shapes_correct(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        assert list(lora.lora_A.shape) == [64, 8]
        assert list(lora.lora_B.shape) == [8, 32]
        assert lora.scaling == 2.0  # alpha/r = 16/8


# =============================================================================
# merge_into_base correctness
# =============================================================================


class TestMergeIntoBase:
    """Verify merged linear produces identical output to LoRA forward."""

    def test_merge_matches_forward(self):
        paddle.seed(42)
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        lora.eval()
        x = paddle.randn([4, 64])
        out_lora = lora(x)
        merged = lora.merge_into_base()
        out_merged = merged(x)
        diff = float(paddle.max(paddle.abs(out_lora - out_merged)))
        assert diff < 1e-5, f"merge mismatch: max diff {diff}"

    def test_merge_no_transpose_bug(self):
        """Regression: previously .T caused shape mismatch [in,out] vs [out,in]."""
        base = nn.Linear(128, 16)
        lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
        merged = lora.merge_into_base()
        assert list(merged.weight.shape) == [128, 16]

    def test_merge_preserves_bias(self):
        base = nn.Linear(64, 32, bias_attr=True)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        merged = lora.merge_into_base()
        assert merged.bias is not None
        np.testing.assert_allclose(
            merged.bias.numpy(), base.bias.numpy(), atol=1e-6
        )

    def test_merge_no_bias_when_base_has_none(self):
        base = nn.Linear(64, 32, bias_attr=False)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        merged = lora.merge_into_base()
        assert merged.bias is None


# =============================================================================
# LoRA forward correctness
# =============================================================================


class TestLoRAForward:
    def test_forward_shape(self):
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        lora.eval()
        x = paddle.randn([2, 64])
        out = lora(x)
        assert list(out.shape) == [2, 32]

    def test_lora_B_zero_init_no_change(self):
        """lora_B is zero-initialized, so initial LoRA output == base output."""
        base = nn.Linear(64, 32)
        lora = LoRALinear(base, r=8, alpha=16, dropout=0.0)
        lora.eval()
        x = paddle.randn([2, 64])
        out_lora = lora(x)
        out_base = base(x)
        diff = float(paddle.max(paddle.abs(out_lora - out_base)))
        assert diff < 1e-6, f"zero-init lora_B should not change output, diff={diff}"

    def test_base_params_frozen(self):
        base = nn.Linear(64, 32)
        LoRALinear(base, r=8, alpha=16, dropout=0.0)
        for p in base.parameters():
            assert p.stop_gradient is True


# =============================================================================
# apply_lora_to_mistral cleanup
# =============================================================================


class _FakeDecoderLayer(nn.Layer):
    """Mimics a Mistral decoder layer's projection structure."""

    def __init__(self, dim=16):
        super().__init__()
        self.self_attn = nn.Layer()
        self.self_attn.q_proj = nn.Linear(dim, dim)
        self.self_attn.k_proj = nn.Linear(dim, dim)
        self.self_attn.v_proj = nn.Linear(dim, dim)
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Layer()
        self.mlp.gate_proj = nn.Linear(dim, dim * 2)
        self.mlp.up_proj = nn.Linear(dim, dim * 2)
        self.mlp.down_proj = nn.Linear(dim * 2, dim)
        self.unrelated_linear = nn.Linear(dim, dim)


class _FakeMistral(nn.Layer):
    def __init__(self, num_layers=2, dim=16):
        super().__init__()
        self.layers = nn.LayerList(
            [_FakeDecoderLayer(dim) for _ in range(num_layers)]
        )
        self.embed_tokens = nn.Embedding(100, dim)


class TestApplyLora:
    def test_replaces_all_target_projections(self):
        model = _FakeMistral(num_layers=2)
        n = apply_lora_to_mistral(model, r=4, alpha=8, dropout=0.0)
        # 7 targets * 2 layers = 14
        assert n == 14, f"expected 14 replacements, got {n}"

    def test_unrelated_linear_not_replaced(self):
        model = _FakeMistral(num_layers=1)
        apply_lora_to_mistral(model, r=4, alpha=8, dropout=0.0)
        # unrelated_linear should remain a plain nn.Linear
        assert isinstance(model.layers[0].unrelated_linear, nn.Linear)
        assert not isinstance(model.layers[0].unrelated_linear, LoRALinear)

    def test_replaced_layers_are_lora(self):
        model = _FakeMistral(num_layers=1)
        apply_lora_to_mistral(model, r=4, alpha=8, dropout=0.0)
        attn = model.layers[0].self_attn
        assert isinstance(attn.q_proj, LoRALinear)
        assert isinstance(attn.k_proj, LoRALinear)
        assert isinstance(attn.v_proj, LoRALinear)
        assert isinstance(attn.o_proj, LoRALinear)
        mlp = model.layers[0].mlp
        assert isinstance(mlp.gate_proj, LoRALinear)
        assert isinstance(mlp.up_proj, LoRALinear)
        assert isinstance(mlp.down_proj, LoRALinear)

    def test_custom_target_names(self):
        model = _FakeMistral(num_layers=1)
        n = apply_lora_to_mistral(
            model, target_names=("q_proj",), r=4, alpha=8, dropout=0.0
        )
        assert n == 1
        assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
        assert isinstance(model.layers[0].self_attn.k_proj, nn.Linear)
        assert not isinstance(model.layers[0].self_attn.k_proj, LoRALinear)


def _get_nested(obj, name):
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


# =============================================================================
# Stage 1 NotImplementedError
# =============================================================================


class TestStage1NotImplemented:
    """Verify Stage 1 raises rather than silently returning zero loss."""

    def test_stage1_raises(self):
        from ppmat.models.matterchat.trainer import MatterChatModule

        # Build a minimal stub to call _forward_stage1 without full init.
        # _forward_stage1 does not access self, so we can invoke via
        # the unbound method on a bare object.
        class _Stub:
            pass

        stub = _Stub()
        stub.__class__ = MatterChatModule
        with pytest.raises(NotImplementedError, match="Stage 1"):
            MatterChatModule._forward_stage1(stub, {})


# =============================================================================
# Tokenizer return_tensors
# =============================================================================


class TestTokenizerReturnTensors:
    """Verify tokenizer wrapper returns paddle tensors regardless of
    return_tensors value."""

    def _make_tokenizer(self, tmp_path):
        # Build a minimal tokenizer.json for HF tokenizers library.
        import json

        vocab = {f"<unk_{i}>": i for i in range(10)}
        vocab.update({"<s>": 1, "</s>": 2, "a": 3, "b": 4})
        tok_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": 1, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": 2, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            ],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "<unk_0>",
            },
        }
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps(tok_json))
        from ppmat.models.matterchat.utils.tokenizer import MistralTokenizerWrapper

        return MistralTokenizerWrapper(str(path))

    def test_returns_paddle_tensors(self, tmp_path):
        tok = self._make_tokenizer(tmp_path)
        result = tok("a b", return_tensors="pd")
        assert isinstance(result.input_ids, paddle.Tensor)
        assert isinstance(result.attention_mask, paddle.Tensor)

    def test_ignores_return_tensors_value(self, tmp_path):
        """Wrapper should return paddle tensors even if 'pt' is passed."""
        tok = self._make_tokenizer(tmp_path)
        result = tok("a b", return_tensors="pt")
        assert isinstance(result.input_ids, paddle.Tensor)

    def test_batch_encoding(self, tmp_path):
        tok = self._make_tokenizer(tmp_path)
        result = tok(["a b", "a"], return_tensors="pd", padding="longest")
        assert result.input_ids.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
