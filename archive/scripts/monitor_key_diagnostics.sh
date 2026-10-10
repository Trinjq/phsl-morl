#!/usr/bin/env bash
set -u

cd "${1:?repository path required}"
shift
interval="${MONITOR_INTERVAL_SECONDS:-1800}"
log=reports/key_diagnostics/monitor_30m.log
mkdir -p "$(dirname "$log")"

while true; do
  date '+%F %T %Z' >> "$log"
  alive=0
  for spec in "$@"; do
    seed="${spec%%:*}"
    pid="${spec##*:}"
    dir="outputs/key_diagnostics/seed_$seed"
    if kill -0 "$pid" 2>/dev/null; then
      state=running
      alive=1
    elif [[ -f "$dir/counters_summary.json" ]]; then
      state=complete
    else
      state=failed
    fi
    rows=0
    [[ -f "$dir/key_replacement_diagnostics.csv" ]] &&
      rows=$(($(wc -l < "$dir/key_replacement_diagnostics.csv") - 1))
    steps=$(grep 'sampled_timesteps:' "$dir/GPU-native PD-MORL_walker2d_brax.log" 2>/dev/null | tail -1 | awk '{print $2}')
    echo "seed=$seed pid=$pid state=$state rows=$rows logged_steps=${steps:-0}" >> "$log"
  done
  ((alive == 0)) && exit 0
  sleep "$interval"
done
