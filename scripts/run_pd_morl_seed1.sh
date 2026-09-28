#!/usr/bin/env bash
set -u

code=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?output directory required}
gpu=${2:?physical GPU index required}

test ! -e "$out" || { echo "output already exists: $out" >&2; exit 2; }
mkdir -p "$out"
export PYTHONPATH="$code"
export CUDA_VISIBLE_DEVICES="$gpu"

nvidia-smi \
  --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used \
  --format=csv,noheader -i "$gpu" -l 30 > "$out/gpu_monitor.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT

cd "$code"
/home/qiuquanj/miniforge3/envs/evorl/bin/python scripts/train.py \
  --config-name experiment/pd_morl_walker_reproduction \
  hydra.run.dir="$out" > "$out/launcher.log" 2>&1
status=$?
echo "$status" > "$out/run.exit"
exit "$status"
