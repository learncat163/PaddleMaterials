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

# MiAD S.U.N. Evaluation Script
# Usage: bash scripts/miad_sun.sh [OPTIONS]
#   -g, --gpus       GPU IDs (default: 0)
#   -i, --input      Generated structures JSON file
#   -r, --ref        Reference CSV file (training set for novelty)
#   -e, --energy     Energy above hull JSON file (optional)
#   --chgnet        Use ppmat CHGNet for energy prediction (no extra deps)
#   --no-relax      Skip structure relaxation (energy only)
#   -o, --output    Output directory

set -e

PYTHON=~/miniconda3/envs/ppmat/bin/python
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="0"
INPUT="./output/miad_generation/generated.json"
REF="./data/mp_20/train.csv"
ENERGY=""
USE_CHGNET=false
NO_RELAX=false
OUTPUT="./output/miad_sun_results"

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)   GPUS="$2"; shift 2 ;;
        -i|--input)  INPUT="$2"; shift 2 ;;
        -r|--ref)    REF="$2"; shift 2 ;;
        -e|--energy) ENERGY="$2"; shift 2 ;;
        --chgnet)    USE_CHGNET=true; shift ;;
        --no-relax)  NO_RELAX=true; shift ;;
        -o|--output) OUTPUT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPUS

echo "========================================="
echo "MiAD S.U.N. Evaluation"
echo "========================================="
echo "Input:      $INPUT"
echo "Reference:  $REF"
echo "Energy:     ${ENERGY:-not provided}"
echo "CHGNet:     $USE_CHGNET"
echo "Relax:      $( [ "$NO_RELAX" = true ] && echo 'no' || echo 'yes' )"
echo "Output:     $OUTPUT"
echo "========================================="

$PYTHON -c "
import json
import os
import numpy as np
from pymatgen.core import Structure, Lattice
from ppmat.metrics.sun_metric import SUNMetric

# Load generated structures
with open('$INPUT', 'r') as f:
    generated = json.load(f)
print(f'Loaded {len(generated)} generated structures')

# Convert to dict format for SUNMetric
struct_dicts = []
for item in generated:
    try:
        struct_dicts.append({
            'lattice': item['lattice'],
            'atom_types': item['atom_types'],
            'frac_coords': item['frac_coords'],
        })
    except Exception:
        struct_dicts.append(None)

valid_dicts = [d for d in struct_dicts if d is not None]
print(f'Valid structures: {len(valid_dicts)}/{len(generated)}')

# Load energy above hull if provided
energy_above_hull = None
energy_path = '${ENERGY}'
if energy_path and os.path.exists(energy_path):
    with open(energy_path, 'r') as f:
        energy_data = json.load(f)
    energy_above_hull = energy_data.get('energy_above_hull_per_atom', None)
    if energy_above_hull:
        print(f'Loaded {len(energy_above_hull)} energy values')

# Determine if CHGNet should be used
use_chgnet = ${USE_CHGNET}
do_relax = not ${NO_RELAX}

# Run S.U.N. evaluation
metric = SUNMetric(
    gt_file_path='$REF',
    stability_threshold=0.0,
    use_chgnet=use_chgnet,
    chgnet_relax=do_relax,
)
results = metric(valid_dicts, energy_above_hull=energy_above_hull)

# Print results
print()
print('=' * 50)
print('           S.U.N. Evaluation Results           ')
print('=' * 50)
print(f'  Total structures:        {results[\"total\"]}')
print(f'  Valid structures:        {results[\"valid\"]}')
print(f'  Non-trivial (>1 elem):   {results[\"non_trivial\"]}')
print(f'  Stability rate:          {results[\"stability_rate\"]:.2f}%')
print(f'  Uniqueness rate:         {results[\"uniqueness_rate\"]:.2f}%')
print(f'  Novelty rate:            {results[\"novelty_rate\"]:.2f}%')
print(f'  S.U.N. rate:             {results[\"sun_rate\"]:.2f}%')
print(f'  S.U.N. count:            {results[\"sun_count\"]}')
if 'avg_energy_per_atom' in results:
    print(f'  Avg energy/atom:         {results[\"avg_energy_per_atom\"]:.4f} eV')
if 'msun_rate' in results:
    print(f'  Metastable rate:         {results[\"metastable_rate\"]:.2f}%')
    print(f'  M.S.U.N. rate:           {results[\"msun_rate\"]:.2f}%')
print('=' * 50)

# Save results
os.makedirs('$OUTPUT', exist_ok=True)
with open(os.path.join('$OUTPUT', 'sun_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to $OUTPUT/sun_results.json')
"
