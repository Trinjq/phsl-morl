> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PHSL-MORL Fix Baseline

- Remote host: lab4090 (172.24.52.209), repository:
  /home/qiuquanj/projects/phsl-stability-hv-history
- Work branch: fix/phsl-stability-hv-history
- Latest upstream main: f39bc3afaae2870c919db3d3f414f63bbe0e8fde
- Latest upstream experiment/key-artifact-refresh-20261009:
  f4972a0032c1771c5484823f10c3a3b5df06c869
- Integrated branch baseline before this change: 6f3ef94
- Active versioned Brax Artifact:
  configs/artifacts/interp_objs_walker2d_brax_v3.txt
- Active Artifact SHA-256:
  e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93
- Formal GPU-native config:
  configs/experiment/pd_morl_brax_gpu_native_v2.yaml
- Historical evaluation data is read-only under:
  /home/qiuquanj/projects/evorl/outputs
- New run outputs belong under:
  /home/qiuquanj/projects/phsl-stability-hv-history/outputs

The training budget counts base environment transitions only. Existing full evaluations
remain episode-triggered (eval_freq=100); hv_history.csv records those results and
does not create a new evaluator or rollout.

## Corrected Artifact promotion

The corrected three-Key retraining completed on the remote GPU 0. The candidate
passed the validator with finite values, exact training counters, nonzero Actor
loss, and JAX/SciPy interpolation agreement. Formal configurations now point to
the versioned v3 Artifact; v2 and the withdrawn unversioned candidate remain
available for historical comparison.
