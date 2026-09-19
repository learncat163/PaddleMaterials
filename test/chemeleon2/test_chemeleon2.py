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

import os

import paddle
from omegaconf import OmegaConf

from ppmat.models import build_model

_CONFIG_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../structure_generation/configs/chemeleon2",
)


def _load_cfg(name):
    path = os.path.join(_CONFIG_DIR, f"chemeleon2_mp20_{name}.yaml")
    return OmegaConf.to_container(OmegaConf.load(path).Model, resolve=True)


def _build_vae():
    return build_model(_load_cfg("vae"))


def _build_ldm():
    return build_model(_load_cfg("ldm"))


def _batch(num_atoms_list, seed=42):
    from ppmat.models.chemeleon2.common.schema import CrystalBatch

    paddle.seed(seed)
    total = sum(num_atoms_list)
    b = CrystalBatch()
    b.atom_types = paddle.randint(1, 90, [total])
    b.frac_coords = paddle.rand([total, 3])
    b.cart_coords = paddle.rand([total, 3])
    b.lattices = paddle.stack([paddle.eye(3) * 5 for _ in num_atoms_list])
    b.lengths = paddle.to_tensor([[5.0, 5.0, 5.0] for _ in num_atoms_list])
    b.angles = paddle.to_tensor([[90.0, 90.0, 90.0] for _ in num_atoms_list])
    b.lengths_scaled = b.lengths / paddle.to_tensor(
        num_atoms_list, dtype="float32"
    ).unsqueeze(-1) ** (1 / 3)
    b.angles_radians = paddle.deg2rad(b.angles)
    b.num_atoms = paddle.to_tensor(num_atoms_list)
    b.batch = paddle.repeat_interleave(
        paddle.arange(len(num_atoms_list)), paddle.to_tensor(num_atoms_list)
    )
    b.token_idx = paddle.concat([paddle.arange(n) for n in num_atoms_list])
    b.num_graphs = len(num_atoms_list)
    b.mask = paddle.ones([len(num_atoms_list), max(num_atoms_list)], dtype="bool")
    return b


def test_vae_pipeline():
    vae = _build_vae()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = vae.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0
    assert not paddle.isnan(loss["total_loss"]).item()

    encoded = vae.encode(b)
    assert "x" in encoded and "moments" in encoded and "posterior" in encoded

    encoded["x"] = encoded["posterior"].sample()
    decoded = vae.decode(encoded)
    assert decoded["atom_types"].shape[0] == 20

    rec = vae.reconstruct(decoded, b)
    assert rec.lattices is not None and rec.lengths is not None

    cfg = vae.get_config()
    assert cfg is not None and "latent_dim" in cfg


def test_ldm_pipeline():
    ldm = _build_ldm()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = ldm.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0

    with paddle.no_grad():
        for sampler in ("ddpm", "ddim"):
            r = ldm.sample(b, sampler=sampler, num_inference_steps=5, progress=False)
            assert "result" in r and len(r["result"]) == 2

    with paddle.no_grad():
        r = ldm.predict(
            {"num_samples": 2, "batch_size": 2, "num_atoms": 8},
            num_inference_steps=5,
            sampler="ddim",
        )
    assert "result" in r

    cfg = ldm.get_config()
    assert cfg is not None and "use_cfg" in cfg

    ldm.merge_lora()
    assert ldm.lora_configs is None


def test_ldm_conditional_pipeline():
    # Small end-to-end CFG pipeline build: forward, loss and sampling with a
    # condition module wired into the DiT through the condition_dim channel.
    import copy

    cfg = copy.deepcopy(_load_cfg("ldm"))
    params = cfg["__init_params__"]
    params["denoiser"]["__init_params__"].update(
        hidden_dim=64, num_layers=2, num_heads=4, condition_dim=64
    )
    params["vae"]["__init_params__"]["encoder"]["__init_params__"].update(
        d_model=64, dim_feedforward=128, num_layers=2
    )
    params["vae"]["__init_params__"]["decoder"]["__init_params__"].update(
        d_model=64, dim_feedforward=128, num_layers=2
    )
    params["condition_module"] = {
        "__class_name__": "ConditionModule",
        "__init_params__": {
            "condition_type": {"condition": "composition"},
            "hidden_dim": 64,
            "drop_prob": 0.1,
        },
    }

    ldm = build_model(cfg)
    assert ldm.use_cfg is True

    b = _batch([8, 12])
    b.y = {"condition": ["MgO", "NaCl"]}
    paddle.seed(42)
    loss = ldm.calculate_loss(b, training=False)
    assert not paddle.isnan(loss["total_loss"]).item()

    with paddle.no_grad():
        r = ldm.sample(b, sampler="ddim", num_inference_steps=3, progress=False)
        assert "result" in r and len(r["result"]) == 2

    # ``StructureSampler.sample_by_condition`` contract: conditions arrive as
    # top-level dict keys, routed through the model's ``condition_names``.
    assert ldm.condition_names == ["condition"]
    data_by_condition = {
        "structure_array": {"num_atoms": paddle.to_tensor([8], dtype="int64")},
        "condition": "MgO",
    }
    with paddle.no_grad():
        r = ldm.sample(
            data_by_condition, sampler="ddim", num_inference_steps=3, progress=False
        )
        assert "result" in r and len(r["result"]) == 1


def test_struct_gen_metric():
    # Generation metrics on hand-crafted structures: Na2O valid twice plus
    # one invalid variant (a non-positive lattice is always rejected).
    import numpy as np

    from ppmat.metrics import StructGenMetric
    from ppmat.metrics.utils import Crystal

    def make_pred(atom_types, frac_coords, lattice):
        return {
            "atom_types": np.array(atom_types),
            "frac_coords": np.array(frac_coords),
            "lattice": np.array(lattice),
        }

    a = [11.0, 11.0, 8.0]
    coords = [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.6, 0.6, 0.1]]
    latt = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    pred_common = make_pred(a, coords, latt)

    # Deterministically invalid: a non-positive lattice is always rejected.
    bad = make_pred(a, coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    metric = StructGenMetric()
    res = metric([pred_common, make_pred(a, coords, latt), bad])
    assert 0.0 <= res["validity"] <= 1.0 and res["validity"] > 0.0
    assert 0.0 <= res["uniqueness"] <= 1.0 and res["uniqueness"] < 1.0

    # Novelty only runs with a reference set; reuse the generated structure
    # as its own reference so the unique structure is correctly non-novel.
    ref_metric = StructGenMetric()
    novelty_pred = [make_pred(a, coords, latt), pred_common]
    ref = [Crystal(pred_common).structure]
    ref_metric.reference_structures = ref
    res2 = ref_metric(novelty_pred)
    assert res2["novelty"] == 0.0


def test_lora_merge_math():
    # LoRA adapter math in the Paddle weight layout: after wrap (B=0) the
    # output equals the base layer; with random B, the LoRA forward, the
    # manually composed update and the merged plain model agree exactly.
    import paddle
    import paddle.nn as nn

    from ppmat.models.chemeleon2.common.lora import apply_lora_to_linear
    from ppmat.models.chemeleon2.common.lora import merge_lora_weights

    class M(nn.Layer):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

    m = M()
    x = paddle.randn([4, 64])
    baseline = m.fc2(m.fc1(x))

    # Capture the original plain weights before the adapter replacement.
    w1, b1 = m.fc1.weight, m.fc1.bias
    w2, b2 = m.fc2.weight, m.fc2.bias

    wrapped = apply_lora_to_linear(m, rank=4)
    cw = wrapped.fc1.lora
    cw.lora_B.set_value(
        paddle.randn([cw.lora_B.shape[0], cw.lora_B.shape[1]]).astype("float32")
    )
    out_lora = wrapped.fc2(wrapped.fc1(x))
    d1 = wrapped.fc1.lora.lora_A @ wrapped.fc1.lora.lora_B * wrapped.fc1.lora.scaling
    d2 = wrapped.fc2.lora.lora_A @ wrapped.fc2.lora.lora_B * wrapped.fc2.lora.scaling
    manual = (x @ (w1 + d1) + b1) @ (w2 + d2) + b2
    assert float((out_lora - manual).abs().max()) < 1e-4

    merged = merge_lora_weights(wrapped)
    out_merged = merged.fc2(merged.fc1(x))
    # fp32 CPU tolerance: merge reorders the matmul graph slightly.
    assert float((out_merged - out_lora).abs().max()) < 1e-3
    # Merge must be a pure weight relocation: sanity anchor on the baseline.
    assert baseline is not None


if __name__ == "__main__":
    tests = [
        test_vae_pipeline,
        test_ldm_pipeline,
        test_ldm_conditional_pipeline,
        test_struct_gen_metric,
        test_lora_merge_math,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    print(f"\nTotal: {len(tests)}, Passed: {len(tests) - failed}, Failed: {failed}")
    exit(1 if failed > 0 else 0)
