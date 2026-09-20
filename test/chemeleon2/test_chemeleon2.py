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

import copy
import os

import numpy as np
import paddle
import pytest
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

    # Generation-to-evaluation contract: the sample() output schema must be
    # directly consumable by the generation metrics layer. Deterministic
    # structures are used because the tiny random model can emit degenerate
    # lattices that stall the structure matcher.
    from ppmat.metrics import StructGenMetric
    from ppmat.metrics.utils import Crystal

    def make_pred(atom_types, frac_coords, lattice):
        return {
            "atom_types": np.array(atom_types),
            "frac_coords": np.array(frac_coords),
            "lattice": np.array(lattice),
        }

    schema_pred = {
        k: np.asarray(v)
        for k, v in r["result"][0].items()
        if k in ("atom_types", "frac_coords", "lattice")
    }
    assert sorted(schema_pred) == ["atom_types", "frac_coords", "lattice"]

    a = [11.0, 11.0, 8.0]
    coords = [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.6, 0.6, 0.1]]
    latt = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    valid_pred = make_pred(a, coords, latt)
    # A deterministically invalid variant: a non-positive lattice is always
    # rejected by the metric layer.
    bad_pred = make_pred(a, coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    metric = StructGenMetric()
    res = metric([valid_pred, valid_pred, bad_pred])
    assert 0.0 <= res["validity"] <= 1.0 and res["validity"] > 0.0
    assert 0.0 <= res["uniqueness"] <= 1.0 and res["uniqueness"] < 1.0

    ref_metric = StructGenMetric()
    ref_metric.reference_structures = [Crystal(valid_pred).structure]
    res2 = ref_metric([valid_pred])
    assert res2["novelty"] == 0.0

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
        "__class_name__": "Chemeleon2ConditionModule",
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


def test_ldm_cinn_dispatch_and_parity(monkeypatch):
    # Single runtime-contract check: hook protocol, weight invariance under
    # backend switches, boundary funneling and eager/cinn numeric parity.
    from ppmat.models.common.runtime import RuntimeMixin

    ldm = _build_ldm()
    assert isinstance(ldm, RuntimeMixin)
    for hook in (
        "set_execution_backend",
        "set_runtime_options",
        "validate_execution_backend",
    ):
        assert callable(getattr(ldm, hook, None))

    keys_eager = sorted(ldm.state_dict().keys())
    ldm.set_execution_backend("cinn")
    assert ldm.execution_backend == "cinn"
    assert sorted(ldm.state_dict().keys()) == keys_eager
    ldm.set_execution_backend("eager")
    assert ldm.execution_backend == "eager"

    ldm.eval()
    batch = _batch([8, 12])

    paddle.seed(7)
    with paddle.no_grad():
        eager_result = ldm.sample(
            batch, sampler="ddim", num_inference_steps=4, progress=False
        )

    seen_boundaries = []

    def eager_dispatch(name, function, *args, **kwargs):
        seen_boundaries.append(name)
        return function(*args, **kwargs)

    monkeypatch.setattr(ldm, "_run_runtime", eager_dispatch)
    ldm.set_execution_backend("cinn")
    paddle.seed(7)
    with paddle.no_grad():
        cinn_result = ldm.sample(
            batch, sampler="ddim", num_inference_steps=4, progress=False
        )

    assert seen_boundaries and set(seen_boundaries) == {"denoise_step"}
    for eager_struct, cinn_struct in zip(eager_result["result"], cinn_result["result"]):
        for key in ("atom_types", "frac_coords", "lattice"):
            assert np.array_equal(
                np.asarray(eager_struct[key]), np.asarray(cinn_struct[key])
            )


@pytest.mark.skipif(
    os.environ.get("PPMAT_RUN_CINN_WORKFLOW_TESTS") != "1",
    reason="set PPMAT_RUN_CINN_WORKFLOW_TESTS=1 to run the CINN GPU compile test",
)
def test_ldm_cinn_gpu_compile_parity():
    # Real CINN compilation on GPU: one compiled denoise step must match eager.
    if not (paddle.is_compiled_with_cuda() and paddle.base.is_compiled_with_cinn()):
        pytest.skip("CUDA Paddle built with CINN is required")

    cfg = copy.deepcopy(_load_cfg("ldm"))
    params = cfg["__init_params__"]
    params["denoiser"]["__init_params__"].update(
        hidden_dim=64, num_layers=2, num_heads=4
    )
    for part in ("encoder", "decoder"):
        params["vae"]["__init_params__"][part]["__init_params__"].update(
            d_model=64, dim_feedforward=128, num_layers=2
        )

    paddle.set_device("gpu")
    try:
        ldm = build_model(cfg)
        ldm.eval()
        x = paddle.randn([2, 8, 8])
        t = paddle.to_tensor([3, 3], dtype="int64")
        mask = paddle.ones([2, 8], dtype="bool")
        with paddle.no_grad():
            eager = ldm._denoise_step(x, t, mask=mask)
            ldm.set_execution_backend("cinn")
            cinn = ldm._denoise_step(x, t, mask=mask)
    finally:
        paddle.set_device("cpu")

    assert bool(paddle.allclose(eager, cinn, atol=2e-5).item())


if __name__ == "__main__":
    # Mirror the pytest conftest device pin so both entry points agree.
    paddle.set_device("cpu")
    tests = [
        test_vae_pipeline,
        test_ldm_pipeline,
        test_ldm_conditional_pipeline,
        test_ldm_cinn_dispatch_and_parity,
    ]

    import inspect

    class PatchStub:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    def test_needs_patch(function):
        return any(
            param.default is inspect.Parameter.empty and param.name == "monkeypatch"
            for param in inspect.signature(function).parameters.values()
        )

    def run_with_patch_stub(function):
        function(PatchStub())

    failed = 0
    for t in tests:
        try:
            if test_needs_patch(t):
                run_with_patch_stub(t)
            else:
                t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    print(f"\nTotal: {len(tests)}, Passed: {len(tests) - failed}, Failed: {failed}")
    exit(1 if failed > 0 else 0)
