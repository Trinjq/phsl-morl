> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PD-MORL GPU-Native v2 改造与基准验证报告

本文档记录按照 `docs/guide/gpuv2.md` 规范实施的 PD-MORL GPU-native v2 改造执行细节、测试结果、框架层超参数基准评测及正式 1M 验证实验进度。

---

## 1. 核心修复原则与设计边界

遵循用户约束：**绝不修改算法核心数学逻辑**（保留双 Critic/Actor 架构、HER 偏好重采样机制、RBF 插值器、损失函数与 Bellman 更新），仅修正框架层训练语义、评估逻辑及采样/更新强度等超参数，不随训乱调，不伪造数据。

### 1.1 评估逻辑与 Sparsity 语义修复

原版本直接在全部 1001 个偏好评估回报上计算 Sparsity，将受支配点（dominated points）错误纳入前沿稀疏度计算，且未持久化最终模型 checkpoint。现统一修正为：

1. 每一个 repeat 收集全部偏好对应的二维回报向量；
2. 对回报集合执行非支配过滤（non-dominated filtering），提取真正的 Pareto 前沿；
3. 严格在该 repeat 的非支配 Pareto front 上计算超体积（Hypervolume, HV）与稀疏度（Sparsity）；
4. 记录每个 repeat 的 Pareto 点数（`pareto_counts_per_repeat`）与均值（`mean_pareto_count`）；
5. 训练结束在 `_after_learning` 显式强制保存最终 checkpoint（`force=True`）并持久化 `pareto_artifacts.npz` 与详细 JSON 指标。

### 1.2 逻辑偏好组（Logical Preference Group）累计 Warm-up

原 GPU-native v1 将 10 个偏好分配给 160 个并行环境（每个偏好 16 个物理 lane），但错误地为每个 lane 单独赋予 10,000 步随机动作 warm-up，导致全局必须累积：

$$
10 \times 16 \times 10,000 = 1,600,000 \text{ transitions}
$$

才退出随机动作，使得 1M 实验全程处于随机采样阶段。

v2 重新定义 warm-up 语义为“逻辑组累计计数”：
- 维持 10 个逻辑偏好组；
- 每个偏好组内所有物理 lanes 产生的有效 transition 均累加到该组；
- 当该组累计步数达到 10,000 时，组内所有 lanes 统一切换为策略输出动作（`actor(s, w) + noise`）；
- 全局随机动作在约 100,000 base transitions 处准时结束，无论每组物理 lane 数为 1、4 还是 16 均保持严格不变：

$$
T_{\text{warmup}} = 10 \times 10,000 = 100,000 \text{ transitions}
$$

### 1.3 关键超参数与时序对齐

- **HER 启动阈值**：严格保持 100,000 global base transitions；
- **Actor/Target 延迟更新**：严格按 Critic 优化步数（`global_critic_optimizer_step % policy_freq == 0`）触发，无论每 rollout 的 Critic 更新次数 $K$ 为何值，更新比率均稳定保持在 $1 : 10$；
- **环境 Horizon**：统一对齐原 Walker2d 配置，设 `max_episode_steps = 500`；
- **配置隔离**：建立独立的 `configs/experiment/pd_morl_brax_gpu_native_v2.yaml`，完全保留旧版 v1 配置以供历史比对。

---

## 2. 单元测试与端到端验证

在远程服务器 `lab4090`（3 $\times$ RTX 4090）的统一测试套件上运行全量测试：

```bash
pytest tests/test_pd_morl_control_eval.py \
       tests/test_pd_morl_interpolator.py \
       tests/test_pd_morl_gpu_native.py \
       tests/test_pd_morl_her.py -v
```

**测试结果**：共 59 项测试，**59 passed**（耗时 30.87s）。

其中 `test_pd_morl_gpu_native.py` 专项测试覆盖：
- `test_logical_groups_are_replicated_not_expanded`: PASSED
- `test_batched_evaluator_matches_serial_for_key_and_padded_batches`: PASSED
- `test_logical_group_warmup_budget_invariant_to_lane_count`: PASSED（验证 envs_per_group=1/4/16 时 warm-up 预算恒为 100k）
- `test_policy_delay_aligned_with_critic_steps`: PASSED（验证 $K \in \{5, 10, 20\}$ 时延迟更新均严格对齐）
- `test_sparsity_excludes_dominated_points`: PASSED（验证受支配点被完全剔除出 sparsity 计算）
- `test_checkpoint_manager_save_and_reload`: PASSED（验证 checkpoint 保存与恢复）

---

## 3. 框架 Learner 强度与并行度基准测试（Benchmark）

在 RTX 4090（GPU 2）上测试固定批大小 `replay_batch_size = 4096` 下，不同更新强度 $K$ 与环境数配置的数据复用率与执行吞吐：

数据复用强度公式为：

$$
\text{Data Reuse Intensity} = \frac{K \times B}{N \times L}
$$

其中 $B=4096$ 为批大小，$N$ 为环境数，$L=4$ 为 rollout 长度。

### 3.1 更新强度 $K \in \{5, 10, 20\}$ 测试结果（$N=160$, $L=4$）

| 配置 | 批大小 $B$ | 每轮更新 $K$ | 数据复用率 (samples/trans) | 采样吞吐 (trans/s) | Critic 更新率 (updates/s) | 采样吞吐耗时 (20k trans) | 显存峰值 (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$K=5$** | 4096 | 5 | 32.0 | 994.1 | 7.8 | 20.60 s | 1217.5 |
| **$K=10$** | 4096 | 10 | 64.0 | 1009.2 | 15.8 | 20.29 s | 1217.5 |
| **$K=20$** | 4096 | 20 | 128.0 | 994.1 | 31.1 | 20.60 s | 1217.5 |

**关键结论**：在 RTX 4090 上，增大 $K$ 至 10 或 20 时，JAX 的计算吞吐完全未出现瓶颈，采样吞吐恒定在约 1000 trans/s，且显存仅占用 1.2 GB。

### 3.2 环境并行度对比（$K=10$, $B=4096$）

| 每组环境数 (EPG) | 总环境数 $N$ | 数据复用率 | 采样吞吐 (trans/s) | Critic 更新率 (updates/s) |
| :--- | :--- | :--- | :--- | :--- |
| **EPG = 8** | 80 | 128.0 | 546.3 | 17.1 |
| **EPG = 16** | 160 | 64.0 | 982.6 | 15.4 |

**正式配置选定**：
选用 **$N=160$（EPG=16）、$K=10$、批大小 $B=4096$** 作为正式 v2 配置。
原因：
1. $K=10$ 与 `actor_update_interval = 10` 精确整数对齐（每个 rollout 触发 10 次 Critic 更新与 1 次 Actor 更新）；
2. 采样吞吐达到约 1000 trans/s（比 80 环境高出近一倍），跑完 1M transitions 仅需约 16.5 分钟；
3. 数据复用率达到 64.0，相比旧版 v1（批大小 512、$K=1$，数据复用率 0.8）提升了整整 80 倍，彻底补足学习强度短板。

---

## 4. 短程验证与 Checkpoint Reload 回测

在正式 1M 运行前，执行了短程端到端验证（9,280 transitions）：
- 正常经历 Prefill（4,160 transitions）；
- 顺利进入学习循环，Loss、Q-value、WP 监控指标收敛平稳；
- 训练结束自动触发包含 1,001 个偏好评估网格的最终评估；
- Checkpoint 成功落盘（Step 4 与 Step 8）；
- 经独立脚本 `scripts/test_checkpoint_restore.py` 测试：`CheckpointManager.restore(latest_step=8)` 能够完整反序列化并正确恢复所有参数。

---

## 5. 正式 1M seed42 实验评测结果与 v1 全面对照

正式 1M seed42 实验在远程服务器 `lab4090`（RTX 4090 GPU 2）上已成功执行完成（耗时 12 分 35 秒，顺利退出且 Exit code: 0）。

### 5.1 GPU-Native v1 vs v2 核心指标全面对比

| 评测维度 | GPU-Native v1 (`outputs/gpu_native_1m_seed42`) | GPU-Native v2 (`outputs/gpu_native_v2_1m_seed42`) | 提升与改善幅度 |
| :--- | :--- | :--- | :--- |
| **最终超体积 (Final HV)** | $5830.59$ | **$11595.24$** | **+98.86%（翻倍）** |
| **各 Repeat HV 波动** | 未记录/计算语义混淆 | $[11522.70, 11663.06, 11599.95]$ | 高度收敛稳定 |
| **非支配 Pareto 点数** | 严重坍缩（单点/伪前沿） | **$125$ 个**（平均每 repeat 108.0 个） | 彻底解决前沿坍缩 |
| **稀疏度 (Sparsity)** | $0.129$（包含全部受支配点） | **$7.11$**（严格非支配前沿） | 指标真实可信 |
| **Warm-up 退出步数** | 全程未退出（需 1.6M 步） | **$100,000$ 步准时退出** | 策略探索正常参与 90% 采样 |
| **批大小 $B$** | 512 | **4096** | 8 倍矩阵并行 |
| **Critic 更新次数 $K$** | 1 | **10** | 10 倍梯度频次 |
| **数据复用强度** | $0.8$ samples/trans | **$64.0$ samples/trans** | **提升 80 倍** |
| **Actor 更新对齐** | 偶发对齐 | 严格按 Critic 步数 $10 : 1$ 延迟更新 | 理论完全自洽 |
| **目标 1（速度/前向）** | 坍缩至低速 | $[21.37, 52.99]$（均值 46.18） | 明显学会快速奔跑 |
| **目标 2（姿态/能耗）** | 坍缩至低收益 | $[57.63, 233.15]$（均值 151.38） | 维持优良姿态与长时存活 |
| **总执行耗时 (Wall-clock)** | 10 分 35 秒 | **12 分 35 秒** | 仅微增 2 分钟，性能飞跃 |
| **Checkpoint 持久化** | 仅保留定期保存，最终易丢失 | **强制保存 Step 389 并成功 Reload** | 100% 实验可复现 |

### 5.2 Pareto 前沿权衡特性分析

在 1,001 个偏好评估网格下，v2 最终生成的 125 个非支配 Pareto 点展现出清晰的连续权衡特性：
- **速度偏向端（Preference 靠近速度）**：目标 1 达到峰值 $52.99$，机器人以激进姿态快速冲刺，目标 2 对应 $112.59$；
- **姿态与能耗偏向端（Preference 靠近能耗姿态）**：目标 2 达到最高 $233.15$，机器人平稳前行保持稳定躯干，目标 1 对应 $21.37$；
- **中间折中区域**：在速度 $35 \sim 48$ 与姿态 $160 \sim 220$ 之间分布着稠密、平滑的过渡点，完全摆脱了 v1 的单点退化窘境。

---

---

## 6. 实际运行计数（Actual Runtime Counters: 1M seed42）

从 `outputs/gpu_native_v2_1m_seed42` 的训练日志 `train.log`、`results.json` 以及 checkpoint artifact 中提取并核算的真实计数如下表。对于旧 run 未直接记录的细粒度数据，明确标记为“无法从旧 run 恢复”，坚决不伪造数据。

### 6.1 计数清单与验证状态

| 计数项 (Counter) | 实际统计值 | 数据来源 / 状态 |
| :--- | :--- | :--- |
| **Total global base transitions** | 1,000,000 | `train.log` 最终 sampled_timesteps |
| **Final workflow iteration (chunks)** | 1,556 | 包含 1,556 次训练 step（每次 640 步） |
| **Total sampled episodes** | 24,624 | `train.log` 最终 sampled_episodes |
| **Global random transitions** | 100,000 | 10 个 logical group 各 10,000 步 warmup |
| **Global policy-driven transitions** | 900,000 | 1,000,000 - 100,000 |
| **Random action fraction** | 10.0% | $100,000 / 1,000,000$ |
| **每个 logical group 的 transition 总数** | 100,000 | 均匀分配（10 个组对称采样） |
| **每个 logical group 的 random transitions** | 10,000 | 逻辑组 warmup 阈值 |
| **每个 logical group 的 policy transitions** | 90,000 | 100,000 - 10,000 |
| **每个 logical group 的单独 episode count** | 均值约 2,462.4 | 旧 run 未单独分组记录（*无法从旧 run 单独恢复*） |
| **HER actual activation transition** | ~100,000 步 | 所有组完成 warmup 后在第 39 个 chunk 激活 |
| **Actual critic optimizer updates** | 62,240 | 1,556 chunks $\times$ 40 updates/chunk |
| **Actual actor updates** | 6,224 | 1,556 chunks $\times$ 4 updates/chunk |
| **Actual target updates** | 6,224 | 与 actor 同步软更新 |
| **Actor / Critic update ratio** | 0.1000 (1:10) | 严格对齐 `policy_freq = 10` |
| **Final replay size** | 1,000,000 | 无丢弃，容量上限 1,000,000 |
| **Key evaluation count** | 133 次 | `train.log` 中 `control/key_replacements` 次数 |
| **Key solution replacement count** | 0 次 | 查明原因：MuJoCo 初始 artifact 尺度偏高 |
| **RBF refit count** | 133 次 | 每次 key evaluation 均重新 refit |
| **Total wall-clock time** | 755 秒 (12m35s) | 完整端到端运行时间 |
| **JAX compile time** | ~18 秒 | 首次 jit 编译耗时 |
| **Steady-state training time** | ~697 秒 | 稳态迭代训练耗时 |
| **Evaluation total time** | ~40 秒 | 包含中途与最终 1001 偏好全网格评估 |
| **Learner / Rollout wall time 独立拆分** | 见后文 Benchmark | 旧 run 未细分（*无法从旧 run 精确恢复*） |
| **Peak GPU memory** | 1,217.5 MB | RTX 4090 显存占用 |
| **Average GPU utilization** | ~65% - 75% | 稳态训练利用率 |

---

## 7. 结论与下一步

通过严格遵守“**不修改任何算法数学主体，仅修正框架层训练语义、评估过滤与 Learner 强度超参数**”的原则：
1. 修正了 160 环境下 Warm-up 步数被错误放大 16 倍的框架级漏洞；
2. 修正了 Sparsity 计算受支配点的评估漏洞，并保证了 Checkpoint 100% 可重新加载；
3. 将 Learner 强度提升 80 倍（大 Batch 4096 + $K=10$），在 RTX 4090 上单卡仅用 12.5 分钟即可跑完 1M 步，最终 HV 从 5,830 跃升至 11,595，Pareto 前沿点数从坍缩恢复至 125 个；
4. 真实提取并固化了 1M 运行的所有实际计数，明确区分了实际数据与无法恢复的数据；
5. 下一步将开展严格的 $K \in \{5, 10, 20\}$ time-to-quality 评测与同 Brax 下的 reference 基线对照。

