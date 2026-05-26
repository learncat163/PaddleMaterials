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

from ppmat.models.miad.miad import MiAD
from ppmat.models.miad.cspnet_complete import CSPNet as MiadCSPNet
from ppmat.models.miad.cspnet_light import CSPNetLight as MiadCSPNetLight
from ppmat.models.miad.crystal_diffusion import CrystalGen as MiadCrystalGen, DiffCSP as MiadDiffCSP
from ppmat.models.miad.crystal_diffusion import init_diffusion, parse_batch
from ppmat.models.miad.lattice_diffusion import DDPM as MiadDDPM, FM as MiadFM, FM_LenAng as MiadFM_LenAng
from ppmat.models.miad.frac_diffusion import WrappedNormal as MiadWrappedNormal, PFM as MiadPFM
from ppmat.models.miad.type_diffusion import DDPM_onehot as MiadDDPM_onehot, D3PM as MiadD3PM
from ppmat.models.miad.diffusion_utils import SinusoidalTimeEmbeddings as MiadSinusoidalTimeEmbeddings, TimeDistribution as MiadTimeDistribution
from ppmat.models.miad.graph_utils import (
    to_dense_adj,
    dense_to_sparse,
    block_diag,
    segment_csr,
    coalesce,
)
from ppmat.models.miad.scheduler import scheduler

# Pipeline classes
from ppmat.models.miad.train import MiADConfig, MiADTrainer, create_miad_trainer_from_config
from ppmat.models.miad.generate import GenerationConfig, MiADGenerator, create_generator_from_checkpoint

# Collate classes
from ppmat.models.miad.collate import MiADCollator, CrystalBatch, create_miad_dataloader, create_sampling_batch

__all__ = [
    # Model
    'MiAD',
    'MiadCSPNet',
    'MiadCSPNetLight',
    'MiadCrystalGen',
    'MiadDiffCSP',
    'init_diffusion',
    'parse_batch',
    # Diffusion components
    'MiadDDPM',
    'MiadFM',
    'MiadFM_LenAng',
    'MiadWrappedNormal',
    'MiadPFM',
    'MiadDDPM_onehot',
    'MiadD3PM',
    'MiadSinusoidalTimeEmbeddings',
    'MiadTimeDistribution',
    'scheduler',
    # Graph utils
    'to_dense_adj',
    'dense_to_sparse',
    'block_diag',
    'segment_csr',
    'coalesce',
    # Pipeline
    'MiADConfig',
    'MiADTrainer',
    'create_miad_trainer_from_config',
    'GenerationConfig',
    'MiADGenerator',
    'create_generator_from_checkpoint',
    # Collate
    'MiADCollator',
    'CrystalBatch',
    'create_miad_dataloader',
    'create_sampling_batch',
]