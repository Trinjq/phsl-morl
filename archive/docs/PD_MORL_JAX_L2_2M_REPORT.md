> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PHSL Pure-JAX RBF + Unified L2: 2M Remote Experiment Report

## Worthwhile problem and contribution

This work removes the production SciPy/host round trip from PD-MORL's RBF fitting and makes the Key-objective normalization consistent across initialization and online refits, while preserving the original linear-RBF control semantics.

## Reproducibility

- Repository branch: `codex/jax-rbf-unified-l2`
- Target commit: `03a93b59a5732abf64cfce7c197cf158adf407f0`
- Artifact: `configs/artifacts/interp_objs_walker2d_brax.txt`
- Artifact SHA-256: `989e74d631ba74572884c4f32b19e12c9317af847ffbc6c02bdde5ad39bfa209`
- Environment: Walker2d Brax, GPU-native PD-MORL, `total_timesteps=2,000,960` (prefill plus complete rollout chunks), same frozen configuration for A/B/C.
- All training and evaluation commands were executed on `lab4090`; no local training or tests were run.

## Implementation and verification

- Production interpolator fitting, L2 normalization, and forward evaluation are JAX-only.
- Online Key evaluation returns a fixed-shape JAX array; replacement and refit remain on-device.
- `initial` and `online` objective normalization both use L2; preference vectors retain their original L1 simplex logic.
- Linear kernel `-r`, degree 0, zero smoothing, strict `>` replacement, and refit-without-replacement are preserved.
- Run metadata records `interpolator_backend: jax`, `key_objective_normalization: l2`, commit, artifact path/SHA, GPU, JAX/Brax versions, and configuration.
- Remote focused CPU suite: `62 passed, 1 warning`.
- Remote broader suite with GPU selection: `94 passed` after the float32 tolerance was made explicit; targeted GPU checks: `2 passed`.
- Remote `git diff --check` and compile checks passed.

## 2M remote experiment summary

Metrics are means over the recorded three-repeat final evaluation unless noted. A is the original SciPy-L1 behavior, B is SciPy-L2, and C is JAX-L2. The fixed Artifact SHA above was verified for every run, including A/B runs whose legacy metadata schema did not serialize the field.

| Variant | Seed | GPU | HV | Sparsity | Pareto count | Key eval / refit | Replacements | Wall-clock (s) | Steady-state transitions/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A SciPy-L1 | 0 | 2 | 5,703,518.58 | 1,465.81 | 146.67 | 69 / 69 | 0 | 1,090.5 | 2,177.9 |
| B SciPy-L2 | 0 | 0 | 6,374,613.59 | 3,884.79 | 104.33 | 73 / 73 | 0 | 2,295.8 | 949.6 |
| B SciPy-L2 | 1 | 0 | 5,886,449.97 | 2,166.27 | 139.67 | 72 / 72 | 0 | 2,339.6 | 949.0 |
| C JAX-L2 | 0 | 2 | 5,884,091.70 | 1,411.64 | 95.00 | 71 / 71 | 0 | 1,015.7 | 2,280.1 |
| C JAX-L2 | 1 | 1 | 5,941,648.83 | 5,567.22 | 69.00 | 70 / 70 | 0 | 3,018.8 | 725.5 |

All runs reached exactly `2,000,960` environment transitions and reported finite diagnostics. No Key replacements occurred in these runs, so the online path was exercised through evaluation and refit but not through a positive replacement event.

Representative final mean returns at preferences `(0,1)`, `(0.5,0.5)`, and `(1,0)`:

| Variant/seed | `(0,1)` | `(0.5,0.5)` | `(1,0)` |
|---|---|---|---|
| A/0 | `[519.88, 2452.84]` | `[2367.02, 1841.12]` | `[1394.00, 403.94]` |
| B/0 | `[45.30, 360.22]` | `[29.49, 17.68]` | `[29.81, -8.04]` |
| B/1 | `[498.21, 2403.46]` | `[2453.67, 1835.45]` | `[2605.68, 694.06]` |
| C/0 | `[756.26, 2371.13]` | `[72.65, 87.08]` | `[8.57, -0.87]` |
| C/1 | `[667.43, 2333.10]` | `[68.76, 54.88]` | `[1983.35, 552.75]` |

## Interpretation and limits

The implementation and numerical regression goals are met: the production fitting path is JAX-only, unified L2 is recorded in metadata, fixed-shape device-side refit works, and SciPy-L2/JAX-L2 both complete the 2M workflow without non-finite diagnostics. The data do not support a claim of Pareto-quality improvement: HV, sparsity, and Pareto count vary substantially by seed and evaluation repeat.

The three RTX 4090 devices were shared with unrelated lab processes, so wall-clock values are not a controlled hardware benchmark and should not be used to claim a speedup. Peak per-process GPU memory was not captured as a run-level high-water mark; the run metadata records total device memory only. A/B/C were matched by artifact, configuration, and seed, but not by an exclusive physical GPU. The A/B/C comparison is therefore an execution and normalization audit, not a claim of universal performance superiority.

Remote output directories:

- A: `/home/qiuquanj/projects/phsl-scipy-l1/outputs/pd_morl_scipy_l1_2m_seed0_20261009`
- B seed 0: `/home/qiuquanj/projects/phsl-scipy-l2/outputs/pd_morl_scipy_l2_2m_seed0_20261009`
- B seed 1: `/home/qiuquanj/projects/phsl-scipy-l2/outputs/pd_morl_scipy_l2_2m_seed1_20261009`
- C seed 0: `/home/qiuquanj/projects/phsl-jax-rbf-l2/outputs/pd_morl_jax_l2_2m_seed0_20261009`
- C seed 1: `/home/qiuquanj/projects/phsl-jax-rbf-l2/outputs/pd_morl_jax_l2_2m_seed1_20261009`
