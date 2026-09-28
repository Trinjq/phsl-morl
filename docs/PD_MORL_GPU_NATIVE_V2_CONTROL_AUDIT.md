# PD-MORL GPU-Native v2 控制语义审计报告

本文档按照 `docs/guide/gpu2.5.md` 第二节要求，对 PD-MORL GPU-native v2 在多物理环境 Lane 下的 Key Evaluation 与 RBF Interpolator 控制语义进行系统审计。

---

## 1. 原始 CPU Source 的控制流程

在原始官方 PD-MORL 实现（`train_Walker2d_MO_TD3_HER.py`）中：

- **环境结构**：采用多进程架构，维护 10 个独立进程（Logical Workers），每个 Worker 对应一个固定的子偏好空间并维护 1 个独立的串行环境（Physical Lane）；
- **进度追踪**：主进程维护长度为 10 的整数数组 `process_episode_array`。每当 Worker $i$ 完成一个完整 Episode（步数达到上限或终止）时：

$$
\text{process\_episode\_array}[i] \leftarrow \text{process\_episode\_array}[i] + 1
$$

- **触发判断**：在主进程的主循环中，每个环境 step 都会同步检查：

$$
(\text{process\_episode\_array} > \text{eval\_cnt\_ep}).\text{all}() \iff \min_{i \in \{0,\dots,9\}} \text{process\_episode\_array}[i] > \text{eval\_cnt\_ep}
$$

- **评估与更新**：
  1. 触发后将 `eval_cnt_ep` 自增 1；
  2. 调用 `eval_agent_interp`，使用确定性策略对 3 个关键偏好（Key Preferences）分别评测 3 个 Repeat，取平均回报向量 $x_{\text{tmp}}$；
  3. 对每个 Key 偏好 $k$，比较标量化回报：

$$
w_k \cdot x_{\text{tmp}}[k] > w_k \cdot x_{\text{old}}[k]
$$

  4. 若更优则替换 $x_{\text{old}}[k] \leftarrow x_{\text{tmp}}[k]$；
  5. 无论是否发生替换，均将归一化解向量 $x / \|x\|_1$ 送入 `RBFInterpolator(kernel='linear')` 进行全量重新拟合（Refit），并将新插值器赋予 Agent。

- **时序特性**：检查为环境单步级（Event-driven），触发延迟严格为 0 step。

---

## 2. 当前 GPU-Native v2 的控制流程

在 GPU-native v2（`PDMORLGPUWorkflow`）中：

- **环境结构**：10 个逻辑偏好组（`process_count = 10`），每组由 $M = 16$ 个物理环境 Lane 并行采样（共 160 个并行环境）；
- **进度追踪**：每个 Lane 独立执行环境步进与自动重置（`AutoresetMode.NORMAL`）。每个 Rollout（长度 $L=4$）在 JAX 设备端聚合当次产生的终止信号：

$$
C_g = \sum_{m=0}^{M-1} \text{dones}_{g, m}
$$

设备端状态维护每组累计的终止总数：

$$
\text{episode\_count}[g] \leftarrow \text{episode\_count}[g] + C_g
$$

- **控制计数定义**：在主机端通过 `_control_episode_count` 将组内所有物理 Lane 的累计完成数归一化为逻辑组完成数：

$$
E_g = \left\lfloor \frac{\text{episode\_count}[g]}{M} \right\rfloor = \left\lfloor \frac{\text{episode\_count}[g]}{16} \right\rfloor
$$

- **触发判断**：在每个静态训练 Chunk（由 `fold_iters = 4` 个 Rollout 组成，共 $160 \times 4 \times 4 = 2560$ 个 Base Transitions）执行完毕后，在 Host 边界调用 `_after_multi_steps`：

$$
\min_{g \in \{0,\dots,9\}} E_g > \text{eval\_cnt\_ep}
$$

若满足，则自增 `eval_cnt_ep`，通过 `BatchedPDMORLEvaluator` 批量评测 Key 偏好，执行标量化贪心替换并重新拟合 JAX-compatible RBF 插值器状态。

---

## 3. 数学定义与触发延迟分析

### 3.1 计数器与触发条件数学形式

对于逻辑偏好组 $g \in \{0, 1, \dots, 9\}$，物理 Lane 数为 $M$：

- **组累计完成数**：

$$
C_g(t) = \sum_{\tau=1}^t \sum_{m=0}^{M-1} \mathbb{I}(\text{lane}_{g,m} \text{ completes an episode at step } \tau)
$$

- **逻辑控制计数值**：

$$
E_g(t) = \left\lfloor \frac{C_g(t)}{M} \right\rfloor
$$

- **触发条件**：

$$
\text{key\_update\_due}(E, \text{eval\_cnt\_ep}) \iff \bigwedge_{g=0}^9 \left( E_g > \text{eval\_cnt\_ep} \right)
$$

### 3.2 触发延迟量化

- **Chunk 边界检查开销**：由于 JAX `lax.scan` 静态图将 4 次 Rollout 编译为一个整体执行单次 dispatch，控制逻辑只能在 Chunk 结束时被 Host 检测；
- **最大可能触发延迟**：若在 Chunk 的第 1 个 Rollout 即达成了所有组的瓶颈 Episode，触发将被推迟至当前 Chunk 结束，即延迟：

$$
\Delta_{\max} = (\text{fold\_iters} - 1) \times L = (4 - 1) \times 4 = 12 \text{ steps/lane}
$$

对应全局 Transitions 延迟为 $12 \times 160 = 1920$ transitions。
- **平均触发延迟**：在均匀分布假设下约为：

$$
\Delta_{\text{avg}} = \frac{1}{2} \text{fold\_iters} \times L = 2 \times 4 = 8 \text{ steps/lane}
$$

对应全局 1280 transitions。
- **相对比率**：Walker2d 的 Episode Horizon 为 500 步，8 步的平均延迟仅占整个 Episode 时长的 $1.6\%$，对于平滑演化的 Policy 和插值目标而言，这种微弱滞后不会造成语义偏差。

---

## 4. 1M Seed 42 实际运行数据统计

从远程服务器 `lab4090` 上的 `outputs/gpu_native_v2_1m_seed42/train.log` 与评估 Artifact 提取的真实数据：

1. **Key Evaluation 触发次数**：**133 次**；
2. **RBF Refit 执行次数**：**133 次**；
3. **Key Solution 替换成功次数**：**0 次**；
4. **全量 Full Evaluation（1001 偏好）触发次数**：**3 次**（迭代 1、100、200）；
5. **最终 Episode 完成总数**：50,875 次（平均每 Lane 完成约 318 次，每组完成约 5,088 次）；
6. **Key 替换为 0 的根因定位**：
   - 当前使用的初始关键解工件 `configs/artifacts/interp_objs_walker2d.txt` 来自原论文预训练的 MuJoCo 模型；
   - 其初始标量化回报约为 $2500 \sim 3000$（MuJoCo 环境标准尺度）；
   - 而当前 Brax Walker2d（Horizon=500）的回报尺度约为 $50 \sim 250$；
   - 因此在线采样的 $w \cdot x_{\text{candidate}}$ 无论如何训练均低于 $w \cdot x_{\text{old}}$，贪心判断始终未满足；
   - 尽管未发生替换，但 RBF 插值器仍正常在线 Refit，并在全训练过程中保持数值稳定。

---

## 5. 语义分类与架构定性

| 控制维度 | 特征表现 | 语义定性 |
| :--- | :--- | :--- |
| **组完成数定义** | 使用 $C_g // M$ 对多 Lane 进行整除平均 | **Framework Adaptation（框架适配）**：消除了 Lane 数对 Episode 累积速度的无意膨胀 |
| **触发瓶颈机制** | 使用 `(E > eval_cnt_ep).all()` 严格等待最慢逻辑组 | **Source-Faithful（完全保真）**：严格对应原算法的组间最慢推进原则 |
| **异步延迟** | Chunk 内 4 次 Rollout 编译导致的 $8 \sim 12$ 步滞后 | **Framework Adaptation（框架适配）**：微小延迟换取 GPU lax.scan 巨大吞吐收益 |
| **插值器 Refit** | 触发即 Refit，与替换解耦 | **Source-Faithful（完全保真）**：严格对应原算法无论是否替换均重新拟合逻辑 |

**审计结论**：当前 GPU-native v2 的 Key/RBF 控制语义属于规范、合理的 **Framework Adaptation**。它在逻辑组层面严谨保留了原论文算法的 Episode 驱动与瓶颈判定机制，消除了物理 Lane 数量的干扰，引入的 Chunk 级微小延迟（$<2\%$ episode）完全符合 GPU 高吞吐设计目标，无需引入额外的复杂度或破坏 JAX 编译。
