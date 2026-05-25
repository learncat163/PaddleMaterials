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

# MiAD Training Script
# Usage: bash scripts/miad_train.sh [OPTIONS]
#   -g, --gpus       GPU IDs (default: 0)
#   -c, --config     Config file path
#   -e, --epochs     Max epochs (default: 8000)
#   -b, --batch      Batch size (default: 256)
#   -o, --output     Output directory
#   -r, --resume     Resume from checkpoint

set -e

PYTHON=~/miniconda3/envs/ppmat/bin/python
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
GPUS="0"
CONFIG="structure_generation/configs/miad/miad_mp20.yaml"
EPOCHS=8000
BATCH=256
OUTPUT=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)    GPUS="$2"; shift 2 ;;
        -c|--config)  CONFIG="$2"; shift 2 ;;
        -e|--epochs)  EPOCHS="$2"; shift 2 ;;
        -b|--batch)   BATCH="$2"; shift 2 ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -r|--resume)  RESUME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPUS

EXTRA_ARGS="Trainer.max_epochs=${EPOCHS}"

if [ -n "$OUTPUT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS Trainer.output_dir=${OUTPUT}"
fi
if [ -n "$RESUME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS Trainer.resume_from_checkpoint=${RESUME}"
fi

echo "========================================="
echo "MiAD Training"
echo "========================================="
echo "Config:  $CONFIG"
echo "GPUs:    $GPUS"
echo "Epochs:  $EPOCHS"
echo "========================================="

$PYTHON structure_generation/train.py \
    -c "$CONFIG" \
    $EXTRA_ARGS
