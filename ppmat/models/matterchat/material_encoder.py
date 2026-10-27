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

"""Checkpoint-compatible MatterChat features from the framework CHGNet."""

import paddle
from paddle import nn

from ppmat.models.chgnet.chgnet import CHGNet
from ppmat.models.chgnet.chgnet import GatedMLP
from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter


class MatterChatEncoder(CHGNet):
    """Reuse CHGNet graph construction and message passing with dialogue weights."""

    _ATOM_GRAPH_CUTOFF = 5
    _BOND_GRAPH_CUTOFF = 3

    def __init__(self):
        super().__init__(
            num_radial=9,
            num_angular=9,
            mlp_hidden_dims=(64, 64),
            atom_graph_cutoff=self._ATOM_GRAPH_CUTOFF,
            bond_graph_cutoff=self._BOND_GRAPH_CUTOFF,
            cutoff_coeff=5,
            mlp_out_bias=True,
            property_names=["energy_per_atom"],
        )
        # Released MatterChat weights contain neither gated nor readout norms.
        for layer in self.sublayers():
            if isinstance(layer, GatedMLP):
                layer.bn1 = nn.Identity()
                layer.bn2 = nn.Identity()
        self.readout_norm = None
        self.graph_converter = CHGNetGraphConverter(
            atom_graph_cutoff=self._ATOM_GRAPH_CUTOFF,
            bond_graph_cutoff=self._BOND_GRAPH_CUTOFF,
        )

    @paddle.no_grad()
    def predict_structure_embedding(self, structure):
        """Return final per-atom features, preserving the dialogue encoder API."""
        features = []

        def capture_features(layer, inputs, output):
            features.append(output)

        hook = self.atom_conv_layers[-1].register_forward_post_hook(capture_features)
        try:
            self._forward({"graph": self.graph_converter(structure)})
        finally:
            hook.remove()
        return features[0]
