# PHSL-MORL Fix Baseline

- Remote host: lab4090 (172.24.52.209), repository:
  /home/qiuquanj/projects/phsl-stability-hv-history
- Work branch: fix/phsl-stability-hv-history
- Latest upstream main: f39bc3afaae2870c919db3d3f414f63bbe0e8fde
- Latest upstream experiment/key-artifact-refresh-20261009:
  f4972a0032c1771c5484823f10c3a3b5df06c869
- Integrated branch baseline before this change: 6f3ef94
- Active versioned Brax Artifact:
  configs/artifacts/interp_objs_walker2d_brax_v2.txt
- Active Artifact SHA-256:
  954adcddd8a97c7d3b0fbfe5e931a0291409d1a5a5005f76e515fe1408225870
- Formal GPU-native config:
  configs/experiment/pd_morl_brax_gpu_native_v2.yaml
- Historical evaluation data is read-only under:
  /home/qiuquanj/projects/evorl/outputs
- New run outputs belong under:
  /home/qiuquanj/projects/phsl-stability-hv-history/outputs

The training budget counts base environment transitions only. Existing full evaluations
remain episode-triggered (eval_freq=100); hv_history.csv records those results and
does not create a new evaluator or rollout.
