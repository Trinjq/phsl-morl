#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
  echo "usage: $0 GPU SEED [SEED ...]" >&2
  exit 2
fi

gpu=$1
shift
repo=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-/home/qiuquanj/miniforge3/envs/evorl/bin/python}
export CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled
cd "$repo"

for seed in "$@"; do
  key_dir="outputs/key_source_like_w05/seed_$seed"
  a_dir="outputs/pd_morl_schedule_stability/source_like/seed_$seed"
  b_dir="outputs/pd_morl_schedule_stability/gpu_native/seed_$seed"
  for dir in "$key_dir" "$a_dir" "$b_dir"; do
    [[ ! -e "$dir" ]] || { echo "refusing to overwrite $dir" >&2; exit 1; }
  done

  mkdir -p "$key_dir"
  "$python" scripts/train_brax_key_solutions.py \
    --key-index 1 --seed "$seed" --source-like --output "$key_dir/result.json" \
    >"$key_dir/train.log" 2>&1

  "$python" scripts/train.py --config-name experiment/pd_morl_schedule_source_like \
    seed="$seed" output_dir="$repo/$a_dir" hydra.run.dir="$repo/$a_dir"

  "$python" scripts/train.py --config-name experiment/pd_morl_brax_gpu_native_v2 \
    seed="$seed" output_dir="$repo/$b_dir" hydra.run.dir="$repo/$b_dir"
done
