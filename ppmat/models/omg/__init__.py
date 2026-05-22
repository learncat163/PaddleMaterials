# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""OMG: Open Materials Generation for crystal structure prediction.
Based on Stochastic Interpolants (ICML 2025, NeurIPS 2025).
"""

import os
import os.path as osp

from omegaconf import OmegaConf

from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load

from ppmat.models.omg.datamodule.structure import Structure
from ppmat.models.omg.datamodule.omg_data import OMGData
from ppmat.models.omg.si import (
    StochasticInterpolants,
    SingleStochasticInterpolant,
    Interpolant,
    Corrector,
    TimeChecker,
)
from ppmat.models.omg.sampler import (
    IndependentSampler,
    PositionDistribution,
    CellDistribution,
    SpeciesDistribution,
    MirrorPosition,
    NormalPositionDistribution,
    UniformPositionDistribution,
    MirrorCell,
    NormalCellDistribution,
    InformedLatticeDistribution,
    MirrorSpecies,
    UniformSpeciesDistribution,
    WeightedSpeciesDistribution,
)
from ppmat.models.omg.model import (
    CSPNet,
    CSPLayer,
    SinusoidsEmbedding,
    BetaScheduler,
    SigmaScheduler,
    lattice_params_to_matrix_paddle,
    frac_to_cart_coords,
    cart_to_frac_coords,
    radius_graph_pbc,
)
from ppmat.models.omg.pipeline import (
    ReinL,
    MatInvent,
)

OMATG_WEIGHTS = {
    "perov_5_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-SDE.pdparams",
    },
    "mpts_52_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-SDE-Gamma.pdparams",
    },
    "mp_20_dng": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-SDE.pdparams",
    },
    "mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-SDE.pdparams",
    },
    "alex_mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-SDE.pdparams",
    },
}

__all__ = [
    # datamodule
    "Structure",
    "OMGData",
    # si
    "StochasticInterpolants",
    "SingleStochasticInterpolant",
    "Interpolant",
    "Corrector",
    "TimeChecker",
    # sampler
    "IndependentSampler",
    "PositionDistribution",
    "CellDistribution",
    "SpeciesDistribution",
    "MirrorPosition",
    "NormalPositionDistribution",
    "UniformPositionDistribution",
    "MirrorCell",
    "NormalCellDistribution",
    "InformedLatticeDistribution",
    "MirrorSpecies",
    "UniformSpeciesDistribution",
    "WeightedSpeciesDistribution",
    # model
    "CSPNet",
    "CSPLayer",
    "SinusoidsEmbedding",
    "BetaScheduler",
    "SigmaScheduler",
    "lattice_params_to_matrix_paddle",
    "frac_to_cart_coords",
    "cart_to_frac_coords",
    "radius_graph_pbc",
    # pipeline
    "ReinL",
    "MatInvent",
    # weights
    "OMATG_WEIGHTS",
    "build_omatg_model",
]


def build_omatg_model(dataset: str, variant: str, weights_name: str = None):
    """Build OMatG model with automatic weight downloading.

    This function follows the same pattern as build_model_from_name but is
    specifically designed for OMatG models which use direct .pdparams files
    without separate configuration files.

    Args:
        dataset: Dataset name, e.g., "mp_20_csp", "perov_5_csp", "mpts_52_csp",
                      "mp_20_dng", "alex_mp_20_csp"
        variant: Model variant, e.g., "linear_ode", "trig_ode_gamma", "encdec_sde_gamma"
        weights_name: Specific weight file name (optional). If None, uses the default
                     weight file corresponding to the variant.

    Returns:
        Loaded model and configuration dictionary

    Example:
        >>> model, config = build_omatg_model("mp_20_csp", "linear_ode")
        >>> model, config = build_omatg_model("perov_5_csp", "trig_ode_gamma", "custom.pdparams")
    """
    # Validate dataset and variant
    if dataset not in OMATG_WEIGHTS:
        available_datasets = list(OMATG_WEIGHTS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {available_datasets}"
        )

    if variant not in OMATG_WEIGHTS[dataset]:
        available_variants = list(OMATG_WEIGHTS[dataset].keys())
        raise ValueError(
            f"Unknown variant: {variant} for dataset {dataset}. "
            f"Available variants: {available_variants}"
        )

    # Get weight URL
    weight_url = OMATG_WEIGHTS[dataset][variant]

    logger.info(f"Building OMatG model: {dataset} / {variant}")
    logger.info(f"Weight URL: {weight_url}")

    # Download weight automatically (uses same mechanism as build_model_from_name)
    weight_path = download.get_weights_path_from_url(weight_url)
    logger.info(f"Weight saved to: {weight_path}")

    # If custom weights_name is specified, use it; otherwise use the variant name
    if weights_name is None:
        # Extract weight filename from URL (e.g., "Linear-ODE.pdparams")
        weight_filename = weight_url.split("/")[-1]
        weights_name = weight_filename.replace(".pdparams", "")
        logger.info(f"Using default weights: {weights_name}")

    # Create default configuration for OMatG model
    # Note: OMatG models don't have separate yaml config files like other models
    # The configuration is defined in the model architecture itself
    config = {
        "Model": {
            "__init_params__": {
                # CSPNet default parameters
                "hidden_dim": 128,
                "latent_dim": 256,
                "num_layers": 4,
                "max_atoms": 100,
                "act_fn": "silu",
                "dis_emb": "sin",
                "num_freqs": 10,
                "edge_style": "fc",
                "cutoff": 6.0,
                "max_neighbors": 20,
                "ln": False,
                "ip": True,
                "smooth": False,
                "pred_type": False,
                "pred_scalar": False,
                "time_embed_dim": None,  # If None, don't use time embedding
            }
        },
        # OMatG-specific configuration
        "dataset": dataset,
        "variant": variant,
        "weights_name": weights_name,
    }

    # Build model using the configuration
    try:
        # Lazy import to avoid circular dependency
        from ppmat.models.omg.model import CSPNet

        model_config = config.get("Model", None)

        # Use the imported class directly instead of build_model
        cls = CSPNet
        init_params = model_config.get("__init_params__", {})
        model = cls(**init_params)

        # Load pre-trained weights
        save_load.load_pretrain(model, weight_path, weights_name)

        logger.info(f"Successfully built and loaded OMatG model: {dataset}/{variant}")

        return model, config

    except Exception as e:
        logger.error(f"Failed to build OMatG model: {e}")
        raise


def get_omatg_model_url(dataset: str, variant: str) -> str:
    """Get the weight URL for an OMatG model variant.

    This is a convenience function for quickly getting the weight URL
    without building the full model.

    Args:
        dataset: Dataset name
        variant: Model variant

    Returns:
        Weight URL string

    Example:
        >>> url = get_omatg_model_url("mp_20_csp", "linear_ode")
        >>> weight_path = download.get_weights_path_from_url(url)
    """
    if dataset not in OMATG_WEIGHTS:
        available_datasets = list(OMATG_WEIGHTS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {available_datasets}"
        )

    if variant not in OMATG_WEIGHTS[dataset]:
        available_variants = list(OMATG_WEIGHTS[dataset].keys())
        raise ValueError(
            f"Unknown variant: {variant} for dataset {dataset}. "
            f"Available variants: {available_variants}"
        )

    return OMATG_WEIGHTS[dataset][variant]
