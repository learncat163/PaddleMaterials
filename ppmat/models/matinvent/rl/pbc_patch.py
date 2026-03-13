#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

"""MatInvent runtime compatibility patch for MatterGen PBC graph building.

The patch is applied only from MatInvent entry points to avoid modifying
ppmat core files. It replaces `radius_graph_pbc` at runtime so that the `pbc`
tensor shape matches batch size.
"""

from __future__ import annotations

import paddle


_ORIG_PADDLE_ALL = None


def apply_mattergen_pbc_patch() -> None:
	"""Apply a MatInvent-scoped patch to MatterGen's `radius_graph_pbc`."""
	import ppmat.models.mattergen.mattergen as mattergen_mod

	global _ORIG_PADDLE_ALL
	if _ORIG_PADDLE_ALL is None:
		_ORIG_PADDLE_ALL = paddle.all

		def _patched_paddle_all(*args, **kwargs):
			x = kwargs.get("x", None)
			if x is None and len(args) > 0:
				x = args[0]
			if isinstance(x, paddle.Tensor) and x.dtype == paddle.bool:
				x = x.contiguous()
				if "x" in kwargs:
					kwargs["x"] = x
				elif len(args) > 0:
					args = (x,) + args[1:]
			return _ORIG_PADDLE_ALL(*args, **kwargs)

		paddle.all = _patched_paddle_all

	if getattr(mattergen_mod, "_matinvent_pbc_patch_applied", False):
		return

	def _patched_radius_graph_pbc(
		cart_coords: paddle.Tensor,
		lattice: paddle.Tensor,
		num_atoms: paddle.Tensor,
		radius: float,
		max_num_neighbors_threshold: int,
		max_cell_images_per_dim: int = 10,
		topk_per_pair: paddle.Tensor | None = None,
	):
		assert topk_per_pair is None, "non None values of topk_per_pair is not supported"

		# Use a >1-length all-True mask to avoid single-element reduction edge cases
		# while keeping periodic graph construction enabled.
		pbc_tensor = paddle.ones(shape=[3, 3], dtype="bool").to(cart_coords.place)

		edge_index, unit_cell, num_neighbors_image, _, _ = mattergen_mod.radius_graph_pbc_ocp(
			pos=cart_coords,
			cell=lattice,
			natoms=num_atoms,
			pbc=pbc_tensor,
			radius=radius,
			max_num_neighbors_threshold=max_num_neighbors_threshold,
			max_cell_images_per_dim=max_cell_images_per_dim,
		)
		return edge_index, unit_cell, num_neighbors_image

	mattergen_mod.radius_graph_pbc = _patched_radius_graph_pbc
	mattergen_mod._matinvent_pbc_patch_applied = True

