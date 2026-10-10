> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PHSL stability recheck

This recheck compares the refreshed v3 key artifact with the previous v2 artifact under the same 2M-transition training configuration. Each seed used one visible RTX 4090 on the remote host and the same code commit (`49c1ef8`).

Artifact hashes:

- v3: `e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93`
- v2: `954adcddd8a97c7d3b0fbfe5e931a0291409d1a5a5005f76e515fe1408225870`

## Paired final evaluation at 2,000,960 environment transitions

| Seed | Artifact | Source HV | Source sparsity | Pareto points | Key replacements |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 15 | v3 | 6,695,690.45 | 1,467.45 | 204 | 0 |
| 15 | v2 | 5,771,612.71 | 941.30 | 198 | 0 |
| 651 | v3 | 6,178,406.85 | 2,725.75 | 162 | 0 |
| 651 | v2 | 6,931,216.07 | 7,723.49 | 123 | 1 |
| 0 | v3 | 6,082,916.64 | 717.72 | 164 | 0 |
| 0 | v2 | 6,695,110.71 | 4,075.11 | 145 | 0 |

The v3 artifact increases the final Pareto-point count for all three seeds (+6, +39, and +19), while the early source HV is mixed: v3 is higher for seed 15 and lower for seeds 651 and 0. The mean source-HV difference is -146,975.18 for v3 minus v2. Therefore the refreshed artifact is not claimed to provide unconditional early-HV superiority; the relevant observed change is broader Pareto coverage, with a seed-dependent HV trade-off.

The 2M runs produced `training_final` evaluations only. The training-full evaluator is episode-count based and had not crossed its 100-episode trigger for these runs, so no `hv_history.csv` was emitted at this horizon. The formal 10M runs are the first runs expected to provide the full HV history and plot artifacts.

## Verification

- All six runs reached exactly 2,000,960 environment transitions.
- Critic-step counters matched the rollout-derived expectation in every run.
- Each run recorded the intended artifact path and SHA in `run_metadata.json`.
- Resume helper regression: 3 focused tests passed, including latest-checkpoint selection and rejection of checkpoints without replay-buffer state.
