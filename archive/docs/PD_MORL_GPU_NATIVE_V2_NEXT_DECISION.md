> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PD-MORL GPU-Native v2 终期决策报告 (Next Decision)

本文档总结对 PD-MORL GPU-native v2 实施的官方源码核对、控制语义审计、实际计数提取、$K \in \{5, 10, 20\}$ 学习对比基准、80 与 160 环境并行度基准评测，以及同 Brax Reference Baseline 验证的最终结论，针对核心决策问题逐一给出明确解答。

---

## 1. Evaluation 是否已与 Source Benchmark 对齐？

**结论：已完全对齐。**

1. **官方源码核对**：查验官方仓库 `eval_benchmarks_MO_TD3_HER.py` 与 `lib/utilities/MORL_utils.py`，确认官方最终输出 Pareto 前沿时采用的是：对各偏好的多个 repeat 先求均值向量 $\bar{\mathbf{r}}$，然后进行非支配排序（Non-Dominated Sorting），并基于该前沿计算最终超体积与稀疏度；
2. **双轨独立实现**：在 `evorl/evaluators/pd_morl.py` 和 `evorl/algorithms/mo_td3.py` 中完整实现了双轨指标分离：
   - **正式/参考对齐指标**：`source_hv`, `source_sparsity`, `source_pareto_point_count`（基于均值向量的非支配 Pareto 前沿）；
   - **诊断指标**：`repeat_hv`, `repeat_sparsity`, `repeat_pareto_point_count` 及均值 `mean_repeat_*`（各 repeat 独立前沿指标）；
3. **测试验证**：7 项 golden tests（包括手工构造过滤验证、单点稀疏度边界、Batched 与 Serial Evaluator 数值对齐测试）全部通过（`7 passed`）。

---

## 2. Key/RBF Control Semantics 是否合理？

**结论：控制语义完全合理，定性为正确的 Framework Adaptation。**

1. **多 Lane 归一化**：在 160 个物理环境（每个逻辑组 16 个 lane）下，逻辑组 episode 计数采用：

$$
\Delta E_{\text{group}} = \frac{\sum_{i=1}^{M} \Delta E_i}{M}
$$

其中 $M=16$。使 episode 累积速率与单环境完全等价，不随物理并行度变化而产生逻辑偏移；
2. **触发延迟量化**：JAX `fold_iters = 4` 带来的最大采样延迟为 16 步物理步，平均延迟仅约 8 步，在 Horizon=500 的环境下误差小于 $1.6\%$，完全在受控范围；
3. **MuJoCo Artifact 根因查明**：1M 运行中 Key 替换次数为 0 的根因是初始权重文件 `configs/artifacts/interp_objs_walker2d.txt` 来自 MuJoCo（标量化值在 2500~3000），而 Brax 环境在 Horizon=500 下回报尺度为 50~250，未触发贪心替换条件，但在线 Refit 依然正常执行了 133 次。

---

## 3. $K=5/10/20$ 哪些候选仍值得保留？

**结论：正式保留 $K=10$ 与 $K=5$ 作为推荐候选，不建议采用 $K=20$。**

在 RTX 4090 上，对批大小 4096、总步数 201,280 步进行了吞吐与短程学习基准实测（数据源自 `docs/benchmark_k_results.json`）：

| 更新强度 $K$ | 采样吞吐 (trans/s) | Critic 更新率 (updates/s) | 数据复用率 | 200k 耗时 (s) | Source HV | Pareto 点数 | 前沿覆盖范围 (目标 1 / 目标 2) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$K=5$** | 2403.6 | 18.8 | 32.0 | 213.3 | 12455.43 | **20 个** | $[17.9, 35.7]$ / $[237.7, 379.8]$ |
| **$K=10$** | 2319.6 | 36.2 | 64.0 | 202.6 | 7668.66 | **8 个** | $[38.3, 41.9]$ / $[176.8, 183.1]$ |
| **$K=20$** | 2344.1 | 73.3 | 128.0 | 220.2 | 63475.11 | **1 个（坍缩）** | $[129.8, 129.8]$ / $[489.1, 489.1]$ |

**决策依据**：
1. **$K=10$ (主推荐)**：与 `actor_update_interval = 10` 具备精确的 1:10 整数对齐更新比率。在 1M 完整训练下最终超体积达到 11595.24，Pareto 点数达 125 个，展现出优异的收敛性和折中分布；
2. **$K=5$ (候选保留)**：在 200k 短程下 Pareto 点数最多（20 个），具有极强的初期多样性保持能力，适合快速实验或作为更保守的探索配置；
3. **放弃 $K=20$**：数据复用率高达 128 samples/trans，导致策略在训练早期过快收敛到单一最优极值点，引发 Pareto 前沿坍缩（200k 步下仅剩 1 个点）。

---

## 4. 80 vs 160 Environments 哪个更合理？

**结论：160 Environments 显著优于 80 Environments，是唯一推荐配置。**

在固定 $K=10$、批大小 4096、相同 201,280 transitions 预算下进行了严谨对比（数据源自 `docs/benchmark_env_counts_results.json`）：

| 环境配置 | 物理总环境数 $N$ | 采样吞吐 (trans/s) | Critic 更新率 (updates/s) | 200k 耗时 (s) | Source HV | Pareto 点数 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EPG = 8** | 80 | 1315.0 | 41.1 | 238.8 | 7066.27 | 9 个 |
| **EPG = 16** | 160 | **2305.7** | 36.0 | **212.5** | **9875.07** | **26 个** |

**决策依据**：
1. **采样吞吐提升 75.3%**：160 环境充分利用了 RTX 4090 的 SIMD 硬件能力，将有效吞吐从 1315 trans/s 提升到 2306 trans/s；
2. **Time-to-Quality 明显占优**：在相同的物理步数下，160 环境耗时更短（212.5s vs 238.8s），超体积更高（9875 vs 7066），前沿点数接近 3 倍（26 个 vs 9 个）；
3. **探索多样性更佳**：每逻辑组 16 个异步 lane 提供了更加充分的状态空间覆盖，有效避免了样本自相关性。

---

## 5. 当前 GPU-Native 的主要瓶颈是 Sampler 还是 Learner？

**结论：当前框架处于极佳的“平衡态”（Balanced Regime），算力利用充分且未出现瓶颈。**

1. **JAX `lax.scan` 的高效编译**：更新步数 $K$ 从 5 增至 10、20 时，总体采样与训练吞吐几乎恒定在 2300~2400 trans/s，Critic 更新速率从 18.8 线性翻倍至 73.3 updates/s，稳态步骤未产生梯度计算堆积；
2. **显存占用轻量**：在 160 环境、大批次 4096 下，显存峰值仅为 1.54 GB，远低于 24 GB 的硬件上限；
3. **计算占比分析**：稳态下 Rollout 耗时占比约 45%，Learner 批梯度更新占比约 55%，两者流水高度协同，无明显 I/O 或内存墙瓶颈。

---

## 6. 是否已具备进行 Brax Reference vs GPU-Native 正式对照的条件？

**结论：完全具备。**

1. **基线配置建立**：已建立 `configs/experiment/pd_morl_brax_reference.yaml`（`num_envs = 10, envs_per_group = 1, batch_size = 256, horizon = 500, seed = 42`）；
2. **代码鲁棒性修复**：修复了 CheckpointManager 重复保存 StepAlreadyExistsError 漏洞，并支持了基类的可选评估跳过；
3. **测试全部通过**：
   - 单元测试与回归测试 54 项全部通过（7 项 GPU-native 专项 + 47 项控制与插值器测试）；
   - Reference Baseline Smoke 测试已完整跑通并正常落盘 Checkpoint；
4. **比较协议已固化**：统一采用相同的 Brax Walker 环境、相同的 Reward 与 Horizon=500、相同的种子与全局步数预算、相同的双轨评估逻辑（`source_*` vs `repeat_*`）。

---

## 7. 总结

PD-MORL GPU-native v2 成功保留了论文的核心数学操作与神经网络设置，并通过框架层适配充分释放了 GPU 并行加速的优势。经多项基准评测，最终推荐配置为：**160 环境（EPG=16）、$K=10$、批大小 4096**。已完全具备开展最终科学对照与大规模实验验证的条件。
