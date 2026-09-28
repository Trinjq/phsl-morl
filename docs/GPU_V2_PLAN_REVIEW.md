# PD-MORL GPU-native v2 实施计划深度评审报告

## 1. 总体评价

一份结构严密、痛点切中极准、工程与科研边界极其清晰的优秀实施计划。
它直接击中了 GPU-native v1 实验失败的核心根因，没有推倒重来，也没有盲目堆叠不必要的复杂组件，而是通过对齐**训练语义（Training Semantics）**和**学习强度（Learning Intensity）**，打造真正适合 GPU 高吞吐特性的 PD-MORL 训练流。

---

## 2. 核心痛点与设计亮点剖析

### 2.1 彻底根治 Warm-up 被物理并行稀释的致命缺陷

- **原 v1 缺陷**：
  原论文为 10 个 logical worker，每个运行 10,000 步随机动作（全局约 100,000 transitions）。在 GPU 并行环境下（如 160 个环境，每组 16 个 lane），如果按每个 physical lane 累计 10,000 步，全局随机步数将膨胀至：
$$
160 \times 10,000 = 1,600,000 \text{ transitions}
$$
  这导致在 1,000,000 步的短程实验中，**整个训练过程 100% 都在收集随机动作**，Actor 策略从未介入过探索，训练自然不可能学出任何有效前沿。
- **v2 设计亮点**：
  引入**逻辑分组累计计数（Logical Group Cumulative Accounting）**：
$$
N_{\text{group}} = 10, \quad N_{\text{group\_transitions}} = \sum_{m \in \text{group}} \Delta t_m
$$
  每个 logical group 内部的所有物理 lane 共同贡献计数，阈值固定为 10,000。
  无论物理并行度扩展到 4、8 还是 16 lanes/group，全局随机探索预算严格收敛在：
$$
10 \times 10,000 = 100,000 \text{ global transitions}
$$
  向量化实现天然适配 JAX：只需要维护形状为 $(10,)$ 的计数器，通过广播到 $(N_{\text{envs}},)$ 即可使用 `jnp.where` 实现纯向量化分支。

---

### 2.2 科学重构 Learner 强度（Data-Reuse Intensity）

- **原 v1 缺陷**：
  每轮采样产出 $160 \times 4 = 640$ 条新 transition，但只做 1 次 batch=512 的 Critic 更新，重用率仅为：
$$
\frac{1 \times 512}{640} = 0.8 \text{ samples/transition}
$$
  采样速度极快但网络根本来不及学习，处于极端欠拟合状态。
- **v2 设计亮点**：
  拒绝退回 CPU 式 640 次串行循环更新（那将完全丧失 GPU 并行优势），采用：
$$
\text{大 Batch (4096)} + \text{适度多步优化 } K \in \{5, 10, 20\}
$$
  定义严格的重用强度指标：
$$
\text{Replay Samples Per Transition} = \frac{K \times B_{\text{replay}}}{T_{\text{rollout}} \times N_{\text{envs}}}
$$
  在 $N_{\text{envs}}=160, T=4, B_{\text{replay}}=4096$ 下，不同 $K$ 对应的强度为：
  - $K=5 \implies \frac{5 \times 4096}{640} = 32$
  - $K=10 \implies \frac{10 \times 4096}{640} = 64$
  - $K=20 \implies \frac{20 \times 4096}{640} = 128$
  兼顾了 GPU 矩阵算力饱和度与策略训练更新频率。

---

### 2.3 严谨的 Policy Delay 触发对齐

无论 Critic 更新步数 $K$ 为何值，严格以全局实际执行的 `global_critic_optimizer_step` 推进：
$$
\text{Trigger Actor \& Target Update} \iff \text{global\_critic\_optimizer\_step} \pmod{10} == 0
$$
保证了 Actor 与 Critic 之间的相对训练比例严格对齐 TD3 原理。

---

### 2.4 指标计算与实验可复现闭环

1. **Sparsity 规范**：
   严格落实前沿过滤先验：
$$
\mathcal{P}_s = \text{NonDominated}(\mathcal{R}_s)
$$
$$
S_s = \frac{1}{|\mathcal{P}_s| - 1} \sum_{l=1}^{L} \sum_{i=1}^{|\mathcal{P}_s|-1} (\hat{r}_{i+1,l} - \hat{r}_{i,l})^2
$$
   对每个 repeat 独立计算后取均值，消除脏点干扰；
2. **检查点安全**：
   要求训练结束必须持久化 Final Checkpoint，并增加 Reload 回测测试，防止“训完无法评估”；
3. **Horizon 对齐**：
   将 Brax Walker 环境的 `max_episode_steps` 由 1000 统一下调为官方的 500。

---

## 3. 落地实现的细节注意事项

1. **JAX JIT 内部的动态计数处理**：
   `group_random_transition_count` 建议作为 `WorkflowState` 或 `AgentState.extra_state` 中的 JAX Array，类型为 `uint32[10]`。每轮 rollout 中使用 `jnp.add` 更新，避免任何 Python 原生副作用。
2. **K 次更新的编译方式**：
   Critic 的 $K$ 步更新以及对应的延时 Actor 更新，应写为静态循环或 `jax.lax.scan`，避免动态 Python 循环产生编译开销。
3. **Replay Buffer 的并发采样安全性**：
   大 Batch=4096 需要确保 Replay Buffer 在达到采样阈值前有清晰的保护，HER 采样重塑与普通转换采样的比例保持对齐。

---

## 4. 实施就绪性判定

该计划逻辑链条完整、边界清晰，可立即按既定顺序（A $\to$ M）启动执行。
