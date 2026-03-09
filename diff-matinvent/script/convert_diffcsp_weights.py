#!/usr/bin/env python3
"""
DiffCSP 权重转换脚本: PyTorch checkpoint → Paddle pdparams (smooth=True)
========================================================================
目标精度: DiffCSP 前向输出 max_diff < 1e-4

架构约定:
  PyTorch (jwchen25/MatInvent): smooth=True, pred_type=True
    - node_embedding = nn.Linear(100, 512, bias=True)
    - 输入 atom_types 是 float soft/one-hot 向量 [N, 100]
    - decoder 返回 (pred_l, pred_x, pred_t)
  
  Paddle (平行对齐目标, smooth=True): 
    - node_embedding = nn.Linear(100→512, bias_attr=True)
    - 输入 atom_types 是 float 向量 [N, 100] (与 PyTorch 一致)
    - decoder 只返回 (pred_l, pred_x), pred_type=False

权重转换规则:
  - Linear.weight: PyTorch [out, in] → Paddle [in, out] (转置)
  - Linear.bias:   不转置, 直接复制
  - LayerNorm:     直接复制
  - coord_out, lattice_out: 无 bias (bias=False), 只转置 weight
  - node_embedding.bias: 直接复制 (smooth=True 两边都有 bias)
  跳过:
    - decoder.type_out.*: Paddle pred_type=False 无此层
    - beta_scheduler.*, sigma_scheduler.*, type_sigma_scheduler.*: Paddle 不存储
  保留原 Paddle 值:
    - decoder.csp_layer_N.prop_mlp.*: PyTorch 无此层, 推理时 property_emb=None 不会被调用

输出文件: /home/cao/code/github/PaddleMaterials/tmp/diffcsp_mp20_converted.pdparams

运行环境: matinvent (PyTorch) 用于读取 checkpoint, ppmat (Paddle) 用于存储
         本脚本在 ppmat 环境下运行, 只用 numpy/pickle 读取 PyTorch checkpoint

用法:
  /home/cao/miniconda3/envs/ppmat/bin/python diff-matinvent/script/convert_diffcsp_weights.py
"""

import sys
import pickle
import struct
import zipfile
import io
import logging
from pathlib import Path

import numpy as np
import paddle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── 路径配置 ───────────────────────────────────────────────────────────────────
PT_CKPT = Path("/home/cao/.cache/huggingface/hub/models--jwchen25--MatInvent/snapshots/b192d079a52d16403fcbbb26b73078d23f9e86c2/diffcsp_mp20/last.ckpt")
PD_CKPT_ORIG = Path("/home/cao/.paddlemat/weights/diffcsp_mp20/diffcsp_mp20/checkpoints/latest.pdparams")
OUT_CKPT = Path(PROJECT_ROOT) / "tmp" / "diffcsp_mp20_converted.pdparams"
OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

# ── 读取 PyTorch checkpoint (通过 pickle 直接解析, 无需 torch) ──────────────────

def load_torch_checkpoint_numpy(ckpt_path: Path) -> dict:
    """
    使用 pickle + numpy 直接读取 PyTorch checkpoint.
    PyTorch .ckpt 是 zip 格式, 里面包含 data.pkl + tensors.
    用 torch 的方式读更简单, 这里改用 torch (matinvent 不可用时的备用).
    
    实际上本脚本在 ppmat 环境运行, torch 不可用, 改用 pickle 手动解析.

    原理: torch.load 实际上是:
    1. 打开 zip 文件
    2. 解析 data.pkl
    3. 读取 tensor 数据

    由于 ppmat 环境没有 torch, 使用纯 Python 实现
    """
    # PyTorch checkpoint 是 zip 文件
    with zipfile.ZipFile(ckpt_path, 'r') as zf:
        names = zf.namelist()
        logger.info(f"Checkpoint zip contents: {names[:5]}...")
        
        # 读取 data.pkl (使用自定义 Unpickler 处理 torch tensors)
        with zf.open('archive/data.pkl') as f:
            data_bytes = f.read()
        
        # 读取所有 tensor 数据文件
        tensor_data = {}
        for name in names:
            if name.startswith('archive/data/') and not name.endswith('/'):
                with zf.open(name) as f:
                    tensor_data[name] = f.read()
    
    return data_bytes, tensor_data


class TorchTensorStub:
    """Stub for unpickling torch tensors without torch"""
    def __init__(self):
        self.storage = None
        self.offset = 0
        self.shape = ()
        self.stride = ()
        self.requires_grad = False
        self.storage_offset = 0
    
    def __reduce_ex__(self, proto):
        pass


class TorchUnpickler(pickle.Unpickler):
    """Custom unpickler that handles torch.Tensor objects"""
    def __init__(self, file, tensor_data):
        super().__init__(file)
        self.tensor_data = tensor_data
        self.persistent_objects = {}
    
    def find_class(self, module, name):
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self._rebuild_tensor_v2
        if module == 'torch.storage' and name == '_load_from_bytes':
            return self._load_from_bytes
        if module == 'torch' and name == 'Size':
            return tuple
        if module == 'torch' and name in ('float32', 'float16', 'int32', 'int64', 'bool'):
            return lambda: np.float32
        # Return a dummy for everything else
        return lambda *args, **kwargs: None
    
    def _rebuild_tensor_v2(self, storage, offset, shape, stride, requires_grad, backward_hooks, metadata=None):
        return {'storage': storage, 'offset': offset, 'shape': shape, 'stride': stride}
    
    def _load_from_bytes(self, b):
        return b
    
    def persistent_load(self, saved_id):
        return saved_id


def load_pt_state_dict(ckpt_path: Path) -> dict:
    """
    Load PyTorch state dict as numpy arrays.
    Falls back to using the matinvent torch environment if available.
    """
    # Try direct numpy reading first
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd = ckpt['state_dict']
        result = {}
        for k, v in sd.items():
            result[k] = v.numpy()
        logger.info(f"Loaded {len(result)} weights using torch")
        return result
    except ImportError:
        logger.info("torch not available, trying alternative loading...")
    
    # Alternative: use npz file if already exported
    npz_path = Path(PROJECT_ROOT) / "tmp" / "diffcsp_pytorch_all_weights.npz"
    if npz_path.exists():
        logger.info(f"Loading from pre-exported NPZ: {npz_path}")
        data = np.load(npz_path)
        # Convert keys back: replace _ separator to .
        # But this is ambiguous, so we stored with a special separator
        result = {}
        for k in data.files:
            original_key = k.replace('__DOT__', '.')
            result[original_key] = data[k]
        logger.info(f"Loaded {len(result)} weights from NPZ")
        return result
    
    raise RuntimeError(
        "Cannot load PyTorch checkpoint: torch not available and NPZ not found.\n"
        f"Please run: export_pt_weights_to_npz.py first, or run in matinvent env"
    )


def convert_diffcsp_weights(pt_sd: dict, pd_orig: dict) -> dict:
    """
    Convert PyTorch DiffCSP state dict to Paddle format (smooth=True target).
    
    原始 PyTorch 代码: raw-matinvent/models/diffcsp/cspnet.py (smooth=True, pred_type=True)
    目标 Paddle 代码: ppmat/models/diffcsp/diffcsp.py (smooth=True 重建, pred_type=False)
    
    转换策略:
      1. 从 pd_orig 拷贝 prop_mlp 等 Paddle 独有的层 (推理时不用, 不影响精度)
      2. 从 pt_sd 转换所有可匹配的层 (decoder.*)
      3. decoder.node_embedding.bias: PT 有 (Linear bias), Paddle smooth=True 也有 → 直接复制
      4. decoder.type_out.*: 跳过 (pred_type=False)
      5. scheduler.*: 跳过 (Paddle 运行时动态计算)
    
    Returns:
        New Paddle state dict with converted weights
    """
    # 先把 pd_orig 中 Paddle 独有的层保留 (prop_mlp)
    new_pd = {}
    for k, v in pd_orig.items():
        if 'prop_mlp' in k:
            new_pd[k] = v
            logger.info(f"  KEEP(prop_mlp) {k}: {list(v.shape)}")

    # Keys to SKIP (not in Paddle architecture):
    SKIP_KEYS = {
        'decoder.node_embedding.bias',  # 单独处理 (添加到 new_pd)
    }
    skip_prefixes = [
        'beta_scheduler.',
        'sigma_scheduler.',
        'type_sigma_scheduler.',
        'decoder.type_out.',
    ]
    
    # 建立 Paddle key 到期望 shape 的映射 (从 pd_orig 获取)
    pd_shapes = {k: list(v.shape) for k, v in pd_orig.items()}
    
    converted_count = 0
    skipped_count = 0
    special_count = 0
    
    for pt_key, pt_val in pt_sd.items():
        # Check if should skip
        if pt_key in SKIP_KEYS:
            # 特殊处理: node_embedding.bias → 直接添加 (smooth=True Paddle 也有此 key)
            new_pd['decoder.node_embedding.bias'] = paddle.to_tensor(pt_val.astype(np.float32))
            logger.info(f"  ADD_BIAS decoder.node_embedding.bias: {list(pt_val.shape)}")
            special_count += 1
            continue
        
        should_skip = any(pt_key.startswith(p) for p in skip_prefixes)
        if should_skip:
            skipped_count += 1
            logger.debug(f"  SKIP: {pt_key}")
            continue
        
        # Paddle key name 与 PyTorch 相同 (都在 decoder.* 下)
        pd_key = pt_key
        pt_shape = list(pt_val.shape)
        
        if pt_key.endswith('.weight') and len(pt_shape) == 2:
            # Linear weight 转置: PyTorch [out, in] → Paddle [in, out]
            converted = pt_val.T  # [in, out]
            expected_shape = [pt_shape[1], pt_shape[0]]
            logger.info(f"  TRANSPOSE {pt_key}: {pt_shape} → {expected_shape}")
        else:
            # bias, LayerNorm: 直接复制
            converted = pt_val
            expected_shape = pt_shape
            logger.info(f"  COPY      {pt_key}: {pt_shape}")
        
        # 验证 shape 与原始 Paddle checkpoint 一致 (除了 node_embedding.bias 是新增的)
        if pd_key in pd_shapes and expected_shape != pd_shapes[pd_key]:
            logger.error(
                f"  SHAPE MISMATCH {pt_key}: converted={expected_shape} vs paddle_orig={pd_shapes[pd_key]}"
            )
            skipped_count += 1
            continue
        
        new_pd[pd_key] = paddle.to_tensor(converted.astype(np.float32))
        converted_count += 1
    
    logger.info(
        f"\nConversion summary: {converted_count} converted, "
        f"{special_count} special, {skipped_count} skipped, "
        f"total={len(new_pd)}"
    )
    return new_pd


def verify_conversion(pt_sd: dict, new_pd: dict):
    """Quick sanity check: verify a few key weights match"""
    logger.info("\n=== Verification ===")
    
    checks = [
        # (pt_key, pd_key, need_transpose)
        ('decoder.csp_layer_0.edge_mlp.0.weight', 'decoder.csp_layer_0.edge_mlp.0.weight', True),
        ('decoder.csp_layer_0.edge_mlp.0.bias',   'decoder.csp_layer_0.edge_mlp.0.bias',   False),
        ('decoder.csp_layer_0.layer_norm.weight',  'decoder.csp_layer_0.layer_norm.weight',  False),
        ('decoder.lattice_out.weight',             'decoder.lattice_out.weight',              True),
        ('decoder.node_embedding.weight',          'decoder.node_embedding.weight',           True),
        ('decoder.atom_latent_emb.weight',         'decoder.atom_latent_emb.weight',          True),
        ('decoder.atom_latent_emb.bias',           'decoder.atom_latent_emb.bias',            False),
    ]
    
    all_pass = True
    for pt_key, pd_key, need_t in checks:
        if pt_key not in pt_sd or pd_key not in new_pd:
            logger.warning(f"  MISSING: {pt_key}")
            continue
        pt_val = pt_sd[pt_key]
        pd_val = new_pd[pd_key].numpy() if hasattr(new_pd[pd_key], 'numpy') else new_pd[pd_key]
        
        if need_t:
            pt_compare = pt_val.T
        else:
            pt_compare = pt_val
        
        if pt_compare.shape != pd_val.shape:
            logger.error(f"  SHAPE MISMATCH {pt_key}: {pt_compare.shape} vs {pd_val.shape}")
            all_pass = False
            continue
        
        diff = np.abs(pt_compare - pd_val).max()
        status = "PASS" if diff < 1e-5 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"  [{status}] {pt_key}: max_diff={diff:.4e}")
    
    return all_pass


def main():
    logger.info("=" * 60)
    logger.info("DiffCSP Weight Conversion: PyTorch → Paddle (smooth=True)")
    logger.info("=" * 60)
    
    # ── 加载 PyTorch state dict ──────────────────────────────────────────────
    logger.info(f"\n1. Loading PyTorch checkpoint: {PT_CKPT}")
    pt_sd = load_pt_state_dict(PT_CKPT)
    logger.info(f"   Found {len(pt_sd)} PyTorch keys")
    
    # ── 加载原始 Paddle checkpoint (用于保存 prop_mlp 等不可转换的权重) ────────
    logger.info(f"\n2. Loading original Paddle checkpoint: {PD_CKPT_ORIG}")
    pd_orig = paddle.load(str(PD_CKPT_ORIG))
    logger.info(f"   Found {len(pd_orig)} Paddle keys")
    
    # ── 执行转换 ──────────────────────────────────────────────────────────────
    logger.info(f"\n3. Converting weights...")
    new_pd = convert_diffcsp_weights(pt_sd, pd_orig)
    
    # ── 验证 ───────────────────────────────────────────────────────────────────
    logger.info(f"\n4. Verifying conversion...")
    all_pass = verify_conversion(pt_sd, new_pd)
    
    if all_pass:
        logger.info("   All checks PASSED ✓")
    else:
        logger.warning("   Some checks FAILED, please review the conversion")
    
    # ── 保存 ───────────────────────────────────────────────────────────────────
    logger.info(f"\n5. Saving converted checkpoint to: {OUT_CKPT}")
    paddle.save(new_pd, str(OUT_CKPT))
    logger.info(f"   Saved {len(new_pd)} keys")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done! Next step: run benchmark_diffcsp_converted.py")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
