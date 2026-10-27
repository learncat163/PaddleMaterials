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

"""Integration tests for the MatterChat inference model."""

import json

import numpy as np
import paddle
import pytest
from cinn_workflow_harness import gpu_cinn_or_skip
from cinn_workflow_harness import requires_gpu_cinn
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.models import MODEL_REGISTRY
from ppmat.models.matterchat.q_former.q_former_complete import Blip2MistralInstruct

# Tiny Mistral config; the default 7B setting does not fit on one GPU.
_SMALL_LLM_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
}

# Minimal WordLevel tokenizer.json accepted by MistralTokenizerWrapper.
_TOK_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 1,
            "content": "<s>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 2,
            "content": "</s>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk_0>": 0,
            "<s>": 1,
            "</s>": 2,
            "a": 3,
            "b": 4,
            "Si": 5,
        },
        "unk_token": "<unk_0>",
    },
}


def _write_tokenizer_json(path):
    path.write_text(json.dumps(_TOK_JSON))


def _make_tokenizer(tmp_path):
    from ppmat.models.matterchat.utils.tokenizer import MistralTokenizerWrapper

    tok_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tok_path)
    return MistralTokenizerWrapper(str(tok_path))


def _build_tiny_model(tokenizer_path=None):
    paddle.seed(0)
    return Blip2MistralInstruct(
        num_query_token=4,
        prompt="test",
        max_txt_len=32,
        max_output_txt_len=32,
        qformer_text_input=False,
        llm_tokenizer=None,
        llm_model=None,
        tokenizer_path=tokenizer_path or "",
        llm_config=dict(_SMALL_LLM_CONFIG),
    )


@pytest.fixture(scope="module")
def small_model():
    """Build a tiny Blip2MistralInstruct without an LLM tokenizer."""
    model = _build_tiny_model()
    model.eval()
    return model


@pytest.fixture(scope="module")
def chat_model(tmp_path_factory):
    """Tiny Blip2MistralInstruct with a generated tokenizer for chat()."""
    tok_dir = tmp_path_factory.mktemp("tokenizer")
    tok_path = tok_dir / "tokenizer.json"
    _write_tokenizer_json(tok_path)
    model = _build_tiny_model(tokenizer_path=str(tok_path))
    model.eval()
    return model


@pytest.fixture
def si_structure():
    return Structure(Lattice.cubic(5.43), ["Si"] * 2, [[0, 0, 0], [0.25, 0.25, 0.25]])


@pytest.fixture
def gan_structure():
    return Structure(Lattice.cubic(4.5), ["Ga", "N"], [[0, 0, 0], [0.5, 0.5, 0.5]])


class TestMatterChatBuildAndForward:
    """Model can be built from config and the core data flow runs."""

    def test_model_builds_from_config(self, small_model):
        from dataclasses import is_dataclass

        from omegaconf import OmegaConf

        from ppmat.models.chgnet.chgnet import CHGNet
        from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter

        assert isinstance(small_model.material_encoder, CHGNet)
        assert isinstance(
            small_model.material_encoder.graph_converter, CHGNetGraphConverter
        )
        assert small_model.Qformer is not None
        assert small_model.query_tokens is not None
        assert small_model.llm_proj is not None
        assert small_model.llm_model is not None

        full_config = OmegaConf.load(
            "structure_generation/configs/matterchat/matterchat_full.yaml"
        )
        settings = OmegaConf.to_container(
            full_config.Model.__init_params__.llm_config, resolve=True
        )
        assert is_dataclass(small_model.llm_model.config)
        assert small_model._build_llm_config(settings) == small_model._build_llm_config(
            None
        )
        with pytest.raises(TypeError, match="hidden_szie"):
            small_model._build_llm_config({"hidden_szie": 64})
        with pytest.raises(ValueError, match="num_key_value_heads"):
            small_model._build_llm_config({"num_key_value_heads": 3})

    def test_chgnet_encodes_structure_to_embedding(self, small_model, si_structure):
        emb = small_model.material_encoder.predict_structure_embedding(si_structure)
        assert isinstance(emb, paddle.Tensor)
        # CHGNet produces per-atom embeddings; Si2 has 2 atoms
        assert emb.ndim == 2
        assert emb.shape[0] == 2

    def test_qformer_forward_from_embedding(self, small_model, si_structure):
        emb = small_model.material_encoder.predict_structure_embedding(si_structure)
        mask = paddle.ones([emb.shape[0]], dtype=paddle.int64)
        query_tokens = small_model.query_tokens.expand([1, -1, -1])
        out = small_model._run_qformer(
            query_tokens,
            emb.unsqueeze(0),
            mask.unsqueeze(0),
            None,
        )
        # Output: [batch=1, num_query_tokens=4, qformer_hidden=768]
        assert out.last_hidden_state.shape == [1, 4, 768]

    def test_llm_proj_maps_to_llm_hidden(self, small_model, si_structure):
        emb = small_model.material_encoder.predict_structure_embedding(si_structure)
        mask = paddle.ones([emb.shape[0]], dtype=paddle.int64)
        query_tokens = small_model.query_tokens.expand([1, -1, -1])
        out = small_model._run_qformer(
            query_tokens,
            emb.unsqueeze(0),
            mask.unsqueeze(0),
            None,
        )
        inputs_llm = small_model.llm_proj(
            out.last_hidden_state[:, : query_tokens.shape[1], :]
        )
        # llm_proj maps qformer hidden (768) -> mistral hidden size
        assert inputs_llm.ndim == 3
        assert inputs_llm.shape[0] == 1
        assert inputs_llm.shape[1] == 4


class TestChatInference:
    """chat() runs the full dialogue pipeline on a single structure."""

    def test_chat_returns_string(self, chat_model, si_structure):
        answer = chat_model.chat(
            si_structure, "what is the chemical formula of this material?"
        )
        assert isinstance(answer, str)

    def test_chat_runs_on_different_structures(self, chat_model, gan_structure):
        answer = chat_model.chat(gan_structure, "what is the chemical formula?")
        assert isinstance(answer, str)

    def test_encoder_and_llm_frozen_for_inference(self, chat_model):
        for p in chat_model.material_encoder.parameters():
            assert p.stop_gradient is True
        for p in chat_model.llm_model.parameters():
            assert p.stop_gradient is True


class TestTokenizerIntegration:
    """Tokenizer returns paddle tensors usable in the inference path."""

    def test_encode_returns_paddle_tensors(self, tmp_path):
        tok = _make_tokenizer(tmp_path)
        result = tok("a b", return_tensors="pd")
        assert isinstance(result.input_ids, paddle.Tensor)
        assert isinstance(result.attention_mask, paddle.Tensor)

    def test_batch_encode_with_padding(self, tmp_path):
        tok = _make_tokenizer(tmp_path)
        result = tok(["a b", "a"], return_tensors="pd", padding="longest")
        assert result.input_ids.shape[0] == 2

    def test_decode_roundtrip(self, tmp_path):
        tok = _make_tokenizer(tmp_path)
        enc = tok("a b", return_tensors="pd")
        decoded = tok.decode(enc.input_ids[0])
        # WordLevel vocab maps a->3, b->4; decode returns a non-empty string
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_special_tokens_present(self, tmp_path):
        tok = _make_tokenizer(tmp_path)
        assert tok.bos_token == "<s>"
        assert tok.eos_token == "</s>"
        assert tok.bos_token_id == 1
        assert tok.eos_token_id == 2


def _registry_layout(tmp_path, model):
    """Materialize a MODEL_REGISTRY-style directory holding `model` weights."""
    import os

    config = {
        "Model": {
            "__class_name__": "Blip2MistralInstruct",
            "__init_params__": {
                "num_query_token": 4,
                "prompt": "test",
                "max_txt_len": 32,
                "max_output_txt_len": 32,
                "qformer_text_input": False,
                "llm_tokenizer": None,
                "llm_model": None,
                "tokenizer_path": "",
                "llm_config": dict(_SMALL_LLM_CONFIG),
            },
        }
    }
    try:
        from omegaconf import OmegaConf

        OmegaConf.save(config, str(tmp_path / "matterchat_full.yaml"))
    except ImportError:
        import yaml

        with open(tmp_path / "matterchat_full.yaml", "w") as f:
            yaml.safe_dump(config, f)
    os.makedirs(tmp_path / "checkpoints", exist_ok=True)
    paddle.save(model.state_dict(), str(tmp_path / "checkpoints" / "best.pdparams"))
    return tmp_path


class TestFromPretrained:
    """from_pretrained parses registry directory structure."""

    def test_raises_on_missing_yaml(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="yaml"):
            Blip2MistralInstruct.from_pretrained(str(tmp_path))

    def test_load_weights_raises_on_missing_weights(self, small_model, tmp_path):
        """_load_weights_from_registry raises when no checkpoint found."""
        with pytest.raises(FileNotFoundError, match="No weights"):
            small_model._load_weights_from_registry(str(tmp_path))

    def test_from_pretrained_roundtrip(self, small_model, tmp_path):
        """from_pretrained rebuilds the model and restores saved weights."""
        path = _registry_layout(tmp_path, small_model)
        loaded = Blip2MistralInstruct.from_pretrained(str(path))
        ref_sd = small_model.state_dict()
        loaded_sd = loaded.state_dict()
        assert set(ref_sd.keys()) == set(loaded_sd.keys())
        for key in ("llm_proj.weight", "llm_proj.bias", "query_tokens"):
            assert paddle.allclose(ref_sd[key], loaded_sd[key])

    def test_registry_entry_exists(self):
        assert "matterchat_full" in MODEL_REGISTRY
        assert MODEL_REGISTRY["matterchat_full"].startswith("http")


class TestConcatText:
    """concat_text_input_output splices input and output token sequences."""

    def test_concat_preserves_input_then_output(self, small_model):
        input_ids = paddle.to_tensor([[1, 2, 3, 0, 0]])
        input_atts = paddle.to_tensor([[1, 1, 1, 0, 0]])
        output_ids = paddle.to_tensor([[4, 5, 6]])
        output_atts = paddle.to_tensor([[1, 1, 1]])
        res, lens = small_model.concat_text_input_output(
            input_ids, input_atts, output_ids, output_atts
        )
        ids = res["input_ids"][0].numpy().tolist()
        # input(1,2,3) + output(4,5,6) + pad(0,0)
        assert ids == [1, 2, 3, 4, 5, 6, 0, 0]
        assert int(lens[0]) == 3

    def test_concat_batch_shape(self, small_model):
        input_ids = paddle.to_tensor([[1, 2, 0, 0], [1, 2, 3, 4]])
        input_atts = paddle.to_tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        output_ids = paddle.to_tensor([[0, 5, 6], [0, 7, 8]])
        output_atts = paddle.to_tensor([[0, 1, 1], [0, 1, 1]])
        res, lens = small_model.concat_text_input_output(
            input_ids, input_atts, output_ids, output_atts
        )
        assert res["input_ids"].shape[0] == 2
        assert res["attention_mask"].shape == res["input_ids"].shape
        assert len(lens) == 2


class TestMultiStructureAndGeneration:
    """CHGNet handles varied structures; generation entry points exist."""

    def test_encode_multi_element_structure(self, small_model, gan_structure):
        emb = small_model.material_encoder.predict_structure_embedding(gan_structure)
        assert emb.ndim == 2
        assert emb.shape[0] == 2  # GaN has 2 atoms

    def test_encode_different_structures_give_different_embeddings(
        self, small_model, si_structure, gan_structure
    ):
        emb_si = small_model.material_encoder.predict_structure_embedding(si_structure)
        emb_gan = small_model.material_encoder.predict_structure_embedding(
            gan_structure
        )
        assert not paddle.allclose(emb_si.mean(), emb_gan.mean())

    def test_encode_batch_of_structures(self, small_model, si_structure, gan_structure):
        embs, masks = [], []
        for s in [si_structure, gan_structure]:
            e = small_model.material_encoder.predict_structure_embedding(s)
            embs.append(e)
            masks.append(paddle.ones([e.shape[0]], dtype=paddle.int64))
        stacked = paddle.stack(embs)
        assert stacked.shape[0] == 2

    def test_generate_entry_point_exists(self, small_model):
        assert hasattr(small_model, "generate")
        assert hasattr(small_model, "generate_followup")
        assert hasattr(small_model, "_encode_for_generate")

    def test_llm_proj_output_dtype_matches_embedding(self, small_model, si_structure):
        emb = small_model.material_encoder.predict_structure_embedding(si_structure)
        mask = paddle.ones([emb.shape[0]], dtype=paddle.int64)
        qt = small_model.query_tokens.expand([1, -1, -1])
        out = small_model._run_qformer(qt, emb.unsqueeze(0), mask.unsqueeze(0), None)
        proj = small_model.llm_proj(out.last_hidden_state[:, : qt.shape[1], :])
        assert proj.dtype == emb.dtype


def test_cinn_chat_dispatch_preserves_eager_output(
    chat_model, si_structure, monkeypatch
):
    keys = tuple(chat_model.state_dict())
    expected = chat_model.chat(si_structure, "a b", max_new_tokens=4)
    recorded = []

    def run_runtime(owner, name, function, *args, **kwargs):
        recorded.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(type(chat_model.llm_model.model), "_run_runtime", run_runtime)
    try:
        chat_model.set_execution_backend("cinn")
        assert chat_model.execution_backend == "cinn"
        assert chat_model.llm_model.model.execution_backend == "cinn"
        assert chat_model.chat(si_structure, "a b", max_new_tokens=4) == expected
        assert recorded and set(recorded) == {"forward"}
        assert tuple(chat_model.state_dict()) == keys
    finally:
        chat_model.set_execution_backend("eager")


@requires_gpu_cinn
def test_gpu_cinn_chat_matches_eager(chat_model, si_structure):
    gpu_cinn_or_skip()
    chat_model.to(paddle.get_device())
    owner = chat_model.llm_model.model
    observed = []

    def collect_hidden(layer, inputs, output):
        observed.append(output[0].numpy().copy())

    hook = owner.register_forward_post_hook(collect_hidden)
    try:
        keys = tuple(chat_model.state_dict())
        chat_model.set_execution_backend("eager")
        expected = chat_model.chat(si_structure, "a b", max_new_tokens=4)
        eager_hidden = list(observed)
        observed.clear()
        chat_model.set_execution_backend("cinn")
        actual = chat_model.chat(si_structure, "a b", max_new_tokens=4)
        assert actual == expected
        assert len(observed) == len(eager_hidden)
        for compiled, eager in zip(observed, eager_hidden):
            np.testing.assert_allclose(compiled, eager, atol=2e-5, rtol=2e-5)
        assert ("cinn", "eval", "forward") in owner._runtime_cache
        assert tuple(chat_model.state_dict()) == keys
    finally:
        hook.remove()
        chat_model.set_execution_backend("eager")
        chat_model.to("cpu")
        paddle.set_device("cpu")


def test_official_trainer_training_and_resume(tmp_path, si_structure, monkeypatch):
    from cinn_workflow_harness import assert_state_dict_close
    from cinn_workflow_harness import trainer_config
    from omegaconf import OmegaConf

    from ppmat.datasets import build_dataloader
    from ppmat.models import build_model
    from ppmat.optimizer import build_optimizer
    from ppmat.trainer import BaseTrainer

    config = OmegaConf.load(
        "structure_generation/configs/matterchat/matterchat_train.yaml"
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer_json(tokenizer_path)
    config.Model.__init_params__.tokenizer_path = str(tokenizer_path)
    llm_config = dict(_SMALL_LLM_CONFIG, vocab_size=32769)
    config.Model.__init_params__.llm_config = llm_config
    config.Model.__init_params__.num_query_token = 4
    model_config = OmegaConf.to_container(config.Model, resolve=True)
    paddle.seed(2026)
    with paddle.utils.unique_name.guard():
        model = build_model(model_config)
    model.eval()
    larger_structure = si_structure.copy()
    larger_structure.make_supercell([2, 1, 1])
    with paddle.no_grad():
        features = [
            model.material_encoder.predict_structure_embedding(structure).numpy()
            for structure in (si_structure, larger_structure)
        ]
    data_path = tmp_path / "train.npz"
    np.savez(
        data_path,
        material_embedding=np.concatenate(features),
        num_atoms=np.array([len(feature) for feature in features]),
        text_input=np.array(["a", "a b"]),
        text=np.array(["b", "a b"]),
    )
    config.Dataset.train.dataset.__init_params__.path = str(data_path)
    config.Dataset.train.sampler.__init_params__.shuffle = False
    loader = build_dataloader(
        OmegaConf.to_container(config.Dataset.train, resolve=True)
    )
    batch = next(iter(loader))
    assert batch["num_atoms"].tolist() == [2, 4]
    assert batch["material_embedding"].shape == [6, 64]
    model.train()
    captured_labels = []
    llm_forward = model.llm_model.forward

    def capture_labels(*args, **kwargs):
        captured_labels.append(kwargs["labels"].numpy().copy())
        return llm_forward(*args, **kwargs)

    monkeypatch.setattr(model.llm_model, "forward", capture_labels)
    loss = model(batch)["loss_dict"]["loss"]
    monkeypatch.setattr(model.llm_model, "forward", llm_forward)
    np.testing.assert_array_equal(
        captured_labels[0],
        [
            [-100, -100, -100, -100, -100, 4, 2, -100, -100],
            [-100, -100, -100, -100, -100, -100, 3, 4, 2],
        ],
    )
    assert bool(paddle.isfinite(loss))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.stop_gradient:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert bool(paddle.isfinite(parameter.grad).all()), name
    model.clear_gradients()
    initial_projection = model.llm_proj.weight.numpy().copy()
    frozen_embedding = model.llm_model.model.embed_tokens.weight.numpy().copy()
    optimizer_config = OmegaConf.to_container(config.Optimizer, resolve=True)
    optimizer, scheduler = build_optimizer(optimizer_config, model, 2, len(loader))
    trainer = BaseTrainer(
        trainer_config(tmp_path / "first", 1),
        model,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        execution_config={"backend": "eager"},
    )
    with paddle.utils.unique_name.guard():
        trainer.train()
    assert trainer.state.global_step == 1
    assert not np.array_equal(initial_projection, model.llm_proj.weight.numpy())
    np.testing.assert_array_equal(
        frozen_embedding, model.llm_model.model.embed_tokens.weight.numpy()
    )
    checkpoint = tmp_path / "first" / "checkpoints" / "latest"
    for suffix in ("pdparams", "pdopt", "pdstates"):
        assert checkpoint.with_suffix("." + suffix).is_file()
    for name, parameter in model.named_parameters():
        assert bool(paddle.isfinite(parameter).all()), name
    expected_keys = tuple(model.state_dict())
    with paddle.utils.unique_name.guard():
        restored = build_model(model_config)
    restored_optimizer, restored_scheduler = build_optimizer(
        optimizer_config, restored, 2, len(loader)
    )
    resumed = BaseTrainer(
        trainer_config(tmp_path / "resumed", 2),
        restored,
        train_dataloader=loader,
        val_dataloader=loader,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        execution_config={"backend": "eager"},
    )
    # Zero epochs after restoration isolate checkpoint restoration from updates.
    resumed.max_epochs = 1
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 1
    assert tuple(restored.state_dict()) == expected_keys
    assert_state_dict_close(restored.state_dict(), model.state_dict(), 0)
    assert_state_dict_close(restored_optimizer.state_dict(), optimizer.state_dict(), 0)
    resumed.max_epochs = 2
    with paddle.utils.unique_name.guard():
        resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 2
    for name, parameter in restored.named_parameters():
        assert bool(paddle.isfinite(parameter).all()), name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
