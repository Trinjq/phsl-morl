> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PD-MORL Schedule Stability Report

**状态：进行中；本文件是可验证的阶段性结果，不是最终三问题结论。**  
**分支基线：** `codex/pd-morl-exp8-k320-v2`，commit `d96be276639fc15b1ec0584ac9e714d0fd8927bc`  
**实验环境：** 实验室 `lab4090`，`/home/qiuquanj/miniforge3/envs/evorl`

## 核心问题

在算法、Brax Walker2d、`interp_objs_walker2d_brax_v2.txt`、RBF、HER、损失、网络、optimizer、reward、evaluation protocol、base-transition budget 和 seeds 全部固定时，仅改变 rollout/更新 schedule，判断当前 GPU-native schedule 是否导致 seed instability、Pareto front 断裂和高 Sparsity。

## 两组 schedule

| 设置 | A: Source-like | B: GPU-native |
|---|---:|---:|
| process_count | 10 | 10 |
| envs_per_preference_group | 1 | 16 |
| num_envs | 10 | 160 |
| rollout_length | 1 | 4 |
| fold_iters | 1 | 4 |
| batch / replay batch | 256 / 256 | 512 / 512 |
| critic updates / rollout | 10 | 320 |
| actor update interval | 10 | 10 |
| total base-transition budget | 10,000,960 | 10,000,960 |

## Smoke test

两组 smoke 均通过，结果保存在：

`outputs/pd_morl_schedule_stability/smoke/validation.json`

关键证据：

- 两组均为 JAX GPU backend，单进程仅见一个 CUDA device。
- loss finite；B 组另用一个独立单-chunk检查确认 Actor/Critic loss 无 NaN/Inf。
- 两组均越过 HER base-transition threshold；共用相同 `PDMORLGPUWorkflow._add_to_replay_buffer` 路径。
- A/B 的 key evaluation 与 RBF refit 分别为 `598/598` 和 `36/36`。
- A 组 `197,120` 个 post-prefill transitions 对应 `197,120` 个 critic updates，即每 10 个新 transitions 对应 10 个 critic updates。
- counter 派生关系全部一致，Actor update count 等于 target update count。

## 已完成的 paired seed 结果

当前只有 seed 1 的 A/B 均已完成。seed 2 和 seed 3 尚未形成完整配对，因此不能计算可靠的三-seed波动、overlay 或最终结论。

| Schedule | Seed | HV | Sparsity | Pareto points | R1 range | R2 range | w=[0.5,0.5] return | Critic / Actor updates | Runtime |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| A: Source-like | 1 | 6,729,753.256 | 8,729.925 | 89 | [704.890, 3,218.396] | [1,605.961, 2,250.037] | [2,972.540, 1,947.162] | 9,996,800 / 999,680 | 6.71 h |
| B: GPU-native | 1 | 6,815,353.579 | 7,486.165 | 88 | [536.649, 3,148.103] | [1,410.178, 2,448.898] | [2,659.076, 1,967.590] | 4,998,400 / 499,840 | 1.57 h |

在 seed 1 上，A 的 HV 略低、Sparsity 更高、Pareto point count 近似相同。单个 seed 不支持“Source-like 明显改善稳定性或前沿连续性”的结论；稳定性本身必须由 seeds 1/2/3 的跨-seed比较判断。

## 当前运行状态

- Seed 1：A/B 已完成。
- Seed 2：A 正在运行；之后自动运行 B。
- Seed 3：Key sanity 已完成，A 已启动；之后自动运行 B。
- 所有正式结果写入独立目录：`outputs/pd_morl_schedule_stability/{source_like,gpu_native}/seed_{1,2,3}/`。

## 三个最终问题

1. Source-like schedule 是否明显降低 seed 间 Pareto front 波动？**等待 seeds 2/3 完整配对。**
2. Source-like schedule 是否明显降低 Sparsity 或减少前沿断裂？**等待 seeds 2/3 完整配对。**
3. 当前 GPU-native schedule 是否可能是 instability 的主要来源？**当前证据不足，不提前判断。**

完成条件是三组 paired seeds 均产生统一 final evaluation JSON；在此之前不根据单 seed 临时调参，也不重新训练 30 个 Key seeds。
