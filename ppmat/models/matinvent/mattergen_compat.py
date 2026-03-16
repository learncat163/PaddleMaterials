#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

"""MatInvent MatterGen PBC 兼容子类。直接用ppmat自带的mattergen，有部分兼容问题

自定义子类原因
----------
ppmat/models/mattergen/mattergen.py 中的 ``radius_graph_pbc`` 函数存在
以下两个 Paddle 兼容性问题，但由于该文件是 ppmat 公共库文件，不应直接修改：

**问题 1 — pbc tensor shape 导致 paddle.all 异常**
  原始代码（ppmat/models/mattergen/mattergen.py, radius_graph_pbc）：
      pbc = paddle.to_tensor([True, True, True], dtype="float32").to("bool")
  shape 为 [3]，经 ``atleast_2d`` 后变为 [1, 3]，对其取列 ``pbc[:, i]``
  得到 shape=[1] 的 bool tensor。Paddle 的 ``paddle.all``/``paddle.any``
  在 shape-1 的 1D bool tensor 上存在发布方式和MatInvent 的使用方式不匹配 ，导致 PBC 判断逻辑失效。

**问题 2 — 非连续 bool tensor 导致 paddle.all 异常**
  对 2D tensor 取列（``pbc[:, i]``）产生非连续（non-contiguous）内存布局。
  Paddle（某些版本）的 ``paddle.all``/``paddle.any`` 在非连续 bool tensor
  上存在计算错误，需要先 ``.contiguous()`` 才能得到正确结果。

"""

from __future__ import annotations

from typing import Optional

import paddle

from ppmat.models.mattergen.mattergen import (
    GemNetT,
    GemNetTCtrl,
    GemNetTDenoiser,
    MatterGen,
    radius_graph_pbc_ocp,
)

def _radius_graph_pbc_fixed(
    cart_coords: paddle.Tensor,
    lattice: paddle.Tensor,
    num_atoms: paddle.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    max_cell_images_per_dim: int = 10,
) -> tuple:
    orig_paddle_all = paddle.all
    orig_paddle_any = paddle.any

    def _patched_all(*args, **kwargs):
        x = kwargs.get("x", None)
        if x is None and len(args) > 0:
            x = args[0]
        if isinstance(x, paddle.Tensor) and x.dtype == paddle.bool:
            x = x.contiguous()
            if "x" in kwargs:
                kwargs["x"] = x
            elif len(args) > 0:
                args = (x,) + args[1:]
        return orig_paddle_all(*args, **kwargs)

    def _patched_any(*args, **kwargs):
        x = kwargs.get("x", None)
        if x is None and len(args) > 0:
            x = args[0]
        if isinstance(x, paddle.Tensor) and x.dtype == paddle.bool:
            x = x.contiguous()
            if "x" in kwargs:
                kwargs["x"] = x
            elif len(args) > 0:
                args = (x,) + args[1:]
        return orig_paddle_any(*args, **kwargs)

    # 构造 [batch_size, 3] 的 pbc tensor
    batch_size = int(num_atoms.shape[0])
    pbc_tensor = paddle.ones(shape=[batch_size, 3], dtype="bool").to(cart_coords.place)

    # 临时修复 paddle.all/any 的 non-contiguous
    paddle.all = _patched_all
    paddle.any = _patched_any
    try:
        edge_index, unit_cell, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
            pos=cart_coords,
            cell=lattice,
            natoms=num_atoms,
            pbc=pbc_tensor,
            radius=radius,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
            max_cell_images_per_dim=max_cell_images_per_dim,
        )
    finally:
        paddle.all = orig_paddle_all
        paddle.any = orig_paddle_any

    return edge_index, unit_cell, num_neighbors_image


class MatinventGemNetT(GemNetT):
    def generate_interaction_graph(
        self,
        cart_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        num_atoms: paddle.Tensor,
        edge_index: Optional[paddle.Tensor] = None,
        to_jimages: Optional[paddle.Tensor] = None,
        num_bonds: Optional[paddle.Tensor] = None,
    ):
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = _radius_graph_pbc_fixed(
                cart_coords=cart_coords,
                lattice=lattice,
                num_atoms=num_atoms,
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_neighbors,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
        orig_otf_graph = self.otf_graph
        self.otf_graph = False
        try:
            result = super().generate_interaction_graph(
                cart_coords, lattice, num_atoms,
                edge_index=edge_index,
                to_jimages=to_jimages,
                num_bonds=num_bonds,
            )
        finally:
            self.otf_graph = orig_otf_graph
        return result

class MatinventGemNetTCtrl(GemNetTCtrl):
    def generate_interaction_graph(
        self,
        cart_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        num_atoms: paddle.Tensor,
        edge_index: Optional[paddle.Tensor] = None,
        to_jimages: Optional[paddle.Tensor] = None,
        num_bonds: Optional[paddle.Tensor] = None,
    ):
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = _radius_graph_pbc_fixed(
                cart_coords=cart_coords,
                lattice=lattice,
                num_atoms=num_atoms,
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_neighbors,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )

        orig_otf_graph = self.otf_graph
        self.otf_graph = False
        try:
            result = super().generate_interaction_graph(
                cart_coords, lattice, num_atoms,
                edge_index=edge_index,
                to_jimages=to_jimages,
                num_bonds=num_bonds,
            )
        finally:
            self.otf_graph = orig_otf_graph
        return result
class MatinventGemNetTDenoiser(GemNetTDenoiser):
    def __init__(self, gemnet_cfg: dict, gemnet_type: str = "GemNetT", **kwargs):
        super().__init__(gemnet_cfg=gemnet_cfg, gemnet_type=gemnet_type, **kwargs)

        # 父类已按 gemnet_type 创建了 self.gemnet（使用原始有 bug 的类），
        # 此处替换为修复版子类。权重通过外部 set_state_dict 加载，不受影响。
        if gemnet_type == "GemNetT":
            self.gemnet = MatinventGemNetT(**gemnet_cfg)
        elif gemnet_type == "GemNetTCtrl":
            self.gemnet = MatinventGemNetTCtrl(**gemnet_cfg)



class MatinventMatterGen(MatterGen):
    def __init__(self, decoder_cfg: dict, **kwargs):
        super().__init__(decoder_cfg=decoder_cfg, **kwargs)
        # 替换父类创建的 GemNetTDenoiser 为修复版，保留所有其他初始化成果
        self.model = MatinventGemNetTDenoiser(**decoder_cfg)