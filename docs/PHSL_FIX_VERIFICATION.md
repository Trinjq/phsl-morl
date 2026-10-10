# PHSL Fix Verification

- Remote repository: /home/qiuquanj/projects/phsl-stability-hv-history
- Branch: fix/phsl-stability-hv-history
- Upstream main checked at f39bc3a; latest key refresh checked at f4972a0.
- Corrected formal Artifact: configs/artifacts/interp_objs_walker2d_brax_v3.txt
- Corrected Artifact SHA-256:
  e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93

## Code and data checks

- Hydra compose for pd_morl_brax_gpu_native_v2: passed.
- Remote GPU focused suite: 63 passed.
- Key smoke: 313 Critic updates, 156 Actor updates, finite nonzero Actor loss.
- Corrected three-Key full run: 3 keys, each 2,000,128 transitions,
  25,088 random steps, 1,999,929 Critic updates, 999,964 Actor updates.
- Corrected Artifact validator: passed finite, metadata, key-order,
  endpoint/midpoint, and JAX/SciPy interpolation checks.
- Historical JSONL export: passed; legacy runs without transition fields derive
  the x-axis from resolved_config.yaml without re-evaluation.
- HV plotting: passed; output includes hv_history.csv, hv_curve.png, and
  hv_summary.json.
- HV writes occur only after existing episode-triggered 201x3 full evaluations;
  no additional evaluator or rollout was added.
- Resume guard: resume requires save_replay_buffer=true and skips prefill;
  external checkpoint Artifact SHA is checked before restore.

## Formal training gate

Formal training starts only after the corrected Artifact and all checks above
pass. Formal runs use the v3 Artifact and one visible GPU per process.
