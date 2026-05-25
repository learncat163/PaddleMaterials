#!/bin/bash
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

# MiAD Generation Script
# Usage: bash scripts/miad_generate.sh [OPTIONS]
#   -g, --gpus       GPU IDs (default: 0)
#   -c, --checkpoint  Model checkpoint path
#   -d, --data       Test data path
#   -n, --num_steps  Number of inference steps (default: 1000)
#   -s, --batch      Batch size for generation (default: 128)
#   -o, --output     Output directory

set -e

PYTHON=~/miniconda3/envs/ppmat/bin/python
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="0"
CHECKPOINT=".historys/maid/2-weight/pretrained-pd/miad_mp20_epoch8000.pdparams"
DATA="./data/mp_20/test.csv"
NUM_STEPS=1000
BATCH=128
OUTPUT="./output/miad_generation"

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)       GPUS="$2"; shift 2 ;;
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -d|--data)       DATA="$2"; shift 2 ;;
        -n|--num_steps)  NUM_STEPS="$2"; shift 2 ;;
        -s|--batch)      BATCH="$2"; shift 2 ;;
        -o|--output)     OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPUS

echo "========================================="
echo "MiAD Generation"
echo "========================================="
echo "Checkpoint:  $CHECKPOINT"
echo "Data:        $DATA"
echo "Inf steps:   $NUM_STEPS"
echo "Output:      $OUTPUT"
echo "========================================="

$PYTHON -c "
import paddle
import os
import json
from ppmat.datasets.mp20_dataset import MP20Dataset
from ppmat.datasets.collate_fn import DefaultCollator
from paddle.io import DataLoader
from ppmat.models.miad.miad import MiAD

# Load model
model = MiAD(
    model_cfg={
        'hidden_dim': 512, 'latent_dim': 256, 'num_layers': 6,
        'smooth': True, 'pred_type': True, 'num_freqs': 128,
        'ln': True, 'ip': True, 'max_atoms': 100,
    },
    diffusion_cfg={
        'task': 'gen_mp20', 'method': 'DiffCSP',
        'cont_time': False, 'num_steps': 1000,
        'lat_diffusion': {'method': 'fm', 'scheduler': 'diffcsp_cosine', 'parameterization': 'eps'},
        'frac_diffusion': {'method': 'wrapped_normal', 'scheduler': 'default_wrapped_normal'},
        'type_diffusion': {'method': 'ddpm_onehot', 'scheduler': 'diffcsp_cosine'},
    },
)

# Load pretrained
from ppmat.models.miad.cspnet_complete import CSPNet
pretrained = CSPNet.load_pytorch_weights('$CHECKPOINT')
decoder_state = model.decoder.state_dict()
for key in pretrained.state_dict():
    if key in decoder_state:
        decoder_state[key] = pretrained.state_dict()[key]
model.decoder.set_state_dict(decoder_state)
model.eval()

# Load data
dataset = MP20Dataset(path='$DATA', build_structure_cfg={'format': 'cif_str', 'num_cpus': 4})
loader = DataLoader(dataset, batch_size=$BATCH, collate_fn=DefaultCollator(), num_workers=0)

# Generate
os.makedirs('$OUTPUT', exist_ok=True)
all_results = []
for i, batch in enumerate(loader):
    result = model.sample(batch, num_inference_steps=$NUM_STEPS)
    all_results.extend(result['result'])
    print(f'Generated batch {i}: {len(result[\"result\"])} crystals')
    if i >= 5:
        break

# Save
with open(os.path.join('$OUTPUT', 'generated.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Saved {len(all_results)} crystals to $OUTPUT/generated.json')
"
