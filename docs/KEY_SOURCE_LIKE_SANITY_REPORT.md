# Key [0.5, 0.5] Source-Like Sanity Report

**状态：已完成（seeds 1, 2, 3）**  
**分支基线：** `codex/pd-morl-exp8-k320-v2`，commit `d96be276639fc15b1ec0584ac9e714d0fd8927bc`  
**实验环境：** 实验室 `lab4090`，`/home/qiuquanj/miniforge3/envs/evorl`

## 核心问题

本实验判断：在不更换现有 Key Artifact、不启用 HER、RBF 或 directional-angle loss 的前提下，恢复官方 PD-MORL Key-training 的 preference noise、早期学习和按 completed episodes 评估协议，是否会明显改变 Brax Walker2d 的中间 Key Solution。

## 协议核对

- Brax Walker2d；MO-TD3；key preference `[0.5, 0.5]`。
- `total_env_steps = 2,000,128`，为 256-transition chunk 下最接近约 2M 的完整预算。
- `batch_size = 100`，replay capacity `500000`，`gamma = 0.99`，`tau = 0.005`，`policy_freq = 2`，hidden `[400, 400]`，learning rate `3e-4`。
- random action 持续 25,088 transitions；首个 chunk 写入 256 条 replay 后即开始 Actor/Critic 学习，满足 replay size ≥ 200。
- 每条训练 transition 的 preference 加入截断到 ±0.05 的 Gaussian noise，再做 L1 normalize。
- 每累计 100 个 completed episodes 评估一次；每次 10 episodes；评估严格使用原始 `[0.5, 0.5]`。
- 以 `0.5*R1 + 0.5*R2` 保存训练期间最优 vector return。
- 未修改 `configs/artifacts/interp_objs_walker2d_brax_v2.txt`。

## 三个 seed 的结果

| Seed | Best R1 | Best R2 | 0.5R1+0.5R2 | J / ||J||2 | Best step | Eval 次数 | Runtime |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 2,784.706 | 2,080.604 | 2,432.655 | [0.8011, 0.5985] | 1,927,936 | 117 | 43.43 min |
| 2 | 2,828.031 | 1,892.597 | 2,360.314 | [0.8311, 0.5562] | 1,769,216 | 114 | 37.11 min |
| 3 | 2,719.774 | 2,009.542 | 2,364.658 | [0.8043, 0.5943] | 1,860,096 | 105 | 22.15 min |

每个 seed 的完整 evaluation history、runtime 和最终 JSON 位于实验室隔离 worktree：

`/home/qiuquanj/projects/phsl-morl-exp8-k320-v2-minimal-verify/outputs/key_source_like_w05/seed_{1,2,3}/result.json`

## 与当前已有 30-seed `[0.5, 0.5]` 结果比较

已有 30-seed 数据来自正式主策略在 `[0.5, 0.5]` 上的 final evaluation，而不是 30 次独立 Key-policy training。因此以下比较适合作为 sanity check，不能单独归因于 Key-training 协议。

| 指标 | 新 Source-Like 3 seeds（mean ± sample SD） | 已有 30 seeds（mean ± sample SD） |
|---|---:|---:|
| R1 | 2,777.504 ± 54.486 | 2,315.712 ± 770.360 |
| R2 | 1,994.248 ± 94.932 | 1,900.388 ± 403.975 |
| Scalarized return | 2,385.876 ± 40.570 | 2,108.050 ± 512.436 |
| Direction R1 | 0.8121 ± 0.0165 | 0.7391 ± 0.1618 |
| Direction R2 | 0.5830 ± 0.0233 | 0.6440 ± 0.1190 |

## 判断

### A. Mean return 是否明显变化？

新 3-seed scalarized-return mean 为 2,385.876，比已有 30-seed mean 高 13.18%。数值变化明显，但由于比较对象分别是独立 Key-policy training 和主策略 final evaluation，不能把全部增幅归因于协议修正。

### B. Normalized direction 是否明显变化？

两组 mean normalized direction 的夹角为 5.40°。方向存在可见但不大的偏移：新结果更偏向 R1；三个新 seed 仍集中在同一中间方向区域，没有出现一致的大幅旋转。

### C. Seed 间方差是否明显下降？

新 3-seed scalarized-return sample SD 为 40.570，仅为已有 30-seed SD 的 0.079 倍；R1、R2 和方向分量的离散度也均更低。描述性结果支持“方差明显下降”，但 `n=3` 与 `n=30` 不对称，且后者包含主训练坍缩 seed，因此不能据此声称已证明一般性的方差降低。

## 结论

Source-like Key-training 得到稳定、方向一致且 scalarized return 较高的中间 Key Solution。结果说明训练协议差异可能影响 balanced key 的回报水平与稳定性，但没有证据要求替换现有 Key Artifact，也不授权重新训练全部 30 个 Key seeds。

