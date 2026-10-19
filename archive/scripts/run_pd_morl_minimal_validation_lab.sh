#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-/home/qiuquanj/miniforge3/envs/evorl/bin/python}
cd "$repo"

for pid in ${WAIT_PIDS:-}; do
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

smoke_root=outputs/pd_morl_schedule_stability/smoke
[[ ! -e "$smoke_root" ]] || { echo "refusing to overwrite $smoke_root" >&2; exit 1; }
mkdir -p "$smoke_root"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  "$python" scripts/train.py --config-name experiment/pd_morl_schedule_source_like \
  seed=0 total_timesteps=201280 run_final_evaluation=false checkpoint.enable=false \
  log_interval=1000 output_dir="$repo/$smoke_root/source_like" \
  hydra.run.dir="$repo/$smoke_root/source_like" >"$smoke_root/source_like.log" 2>&1 &
a=$!
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  "$python" scripts/train.py --config-name experiment/pd_morl_brax_gpu_native_v2 \
  seed=0 total_timesteps=201280 run_final_evaluation=false checkpoint.enable=false \
  log_interval=1000 output_dir="$repo/$smoke_root/gpu_native" \
  hydra.run.dir="$repo/$smoke_root/gpu_native" >"$smoke_root/gpu_native.log" 2>&1 &
b=$!
wait "$a" "$b"

"$python" scripts/validate_pd_morl_schedule_smoke.py \
  "$smoke_root/source_like" "$smoke_root/gpu_native" \
  --output "$smoke_root/validation.json"

scripts/run_pd_morl_minimal_validations.sh 0 1 3 &
a=$!
scripts/run_pd_morl_minimal_validations.sh 2 2 &
b=$!
wait "$a" "$b"
