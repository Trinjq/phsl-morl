> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# Archived Step8 data-parallel investigation

This document records the experimental multi-GPU PD-MORL data-parallel work.
The implementation is preserved on branch
`archive/pd-morl-step8-data-parallel` and is intentionally not imported by the
production single-GPU training path. The supported experiment architecture is
one complete K=10 PD-MORL training run per GPU, with independent training seed,
replay, optimizer, PRNG, interpolator, and checkpoint.

Do not use this archived document as a production launch recipe.

## STEP8 DATA-PARALLEL REVERT AUDIT

The complete pre-revert snapshot is commit `2ec7266` on the archive branch.

| Scope | Files | Disposition |
|---|---|---|
| Distributed helpers | `evorl/distributed/pd_morl.py`, `evorl/distributed/{__init__,gradients}.py` | KEEP_ARCHIVED |
| Multi-GPU smoke/diagnosis | `tests/pd_morl_multi_gpu_smoke.py`, `tests/test_pd_morl_{integration,numerical_diagnosis,production_shape_smoke,real_walker_reproduction,three_gpu_masked,two_gpu_balanced}.py` | KEEP_ARCHIVED |
| Distributed environment/evaluator hooks | `evorl/envs/brax.py`, `evorl/envs/wrappers/preference_wrapper.py`, `evorl/evaluators/{__init__,pd_morl}.py` | REMOVE_FROM_PRODUCTION_PATH |
| Distributed loss/workflow edits | `evorl/algorithms/mo_td3.py`, `evorl/algorithms/{offpolicy_utils,td3}.py`, `configs/agent/mo-td3.yaml` | RESTORE_STEP7_2 |
| Documentation and auxiliary tests | `docs/guide/step7*.md`, `docs/PD_MORL_*`, `tests/test_pd_morl_{alignment,parallel}.py`, `pyproject.toml` | KEEP_ARCHIVED |

On `main`, the production config and algorithm are the Step7.2 single-GPU
versions. The only new production-facing change is the entrypoint guard in
`scripts/train.py`, which rejects a PD-MORL process that sees more than one
GPU; the archived distributed modules are not imported by it.
