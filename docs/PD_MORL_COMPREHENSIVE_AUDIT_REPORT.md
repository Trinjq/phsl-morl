# PD-MORL 官方源码与 Trinjq 全部 13 次提交全景对照审计报告

## 审查背景与对照基准

本报告对照位于 `E:\PD-MORL\PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm` 的官方 PD-MORL 原始实现（基于 PyTorch / OpenAI Gym），对作者 **Trinjq** 在 `evorl` 中截至当前的全部 **13 次 commit 记录**及工作区最新代码进行全景对照审计。

### 13 次提交记录全貌
1. `a94fa9a`: Add MORL reward and preference data pipelines
2. `fb6ba7f`: add Preference Sampling
3. `e1821cb`: mo_TD3
4. `f5d46d6`: add HER
5. `271eb38`: add parallel exploration
6. `c1a1432`: add interpolator
7. `24ca728`: add preference-Q alignment
8. `94bfdd9`: add evaluation & control
9. `a1818fa`: restore single-GPU PD-MORL and add independent seed runner
10. `7f06626`: document Step8 data-parallel revert audit
11. `ea8ab13`: isolate duplicate training seed smoke outputs
12. `81d62d2`: log isolated PD-MORL initialization fingerprint
13. `a3c1233`: record independent-run smoke and reproducibility evidence

---

## 一、十个核心维度的全景源码级对照

### 1. 环境与多目标向量奖励（Environment & Vector Reward）
- **官方源码** (`moenvs/MujocoEnvs/walker2d.py:27-30`):
$$
r_{\text{speed}} = \frac{x_{t+1} - x_t}{\Delta t} + 1.0
$$
$$
r_{\text{energy}} = 4.0 - \sum_{i} a_i^2 + 1.0 = 5.0 - \sum_{i} a_i^2
$$
- **Trinjq 实现** (`evorl/envs/brax.py:MOWalker2dAdapter`):
$$
r = \begin{bmatrix} x_{\text{velocity}} + 1.0 \\ 5.0 - \sum a^2 \end{bmatrix}
$$
- **比对结论**：完全对齐。奖励维数为 2，目标语义严格吻合。Brax 步长在配置中锁定了 `max_episode_steps: 500`，对齐了官方 Gym 500 步限制。

---

### 2. 偏好空间与子空间划分（Preference Simplex & Subspaces）
- **官方源码** (`lib/utilities/settings.py`, `train_Walker2d_MO_TD3_HER.py:80-84`):
  - 步长 $w_{\text{step}} = 0.001$，两目标生成 1001 个网格点：
$$
w \in \{ [0.0, 1.0], [0.001, 0.999], \dots, [1.0, 0.0] \}
$$
  - 通过 `np.array_split(w_batch_test, process_count=10)` 切为 10 个子空间（首个 101 点，其余各 100 点）；
  - 每个 Worker 子进程在 episode 开始时从所属子空间随机采样 1 个偏好，并在整个 episode 期间固定，在 episode 结束时重采样。
- **Trinjq 实现** (`evorl/envs/wrappers/preference_wrapper.py`):
  - 实现了 `_preference_subspace_bounds` 和 `official_preference_grid`，划分点与 `np.array_split` 完全一致；
  - `EpisodePreferenceWrapper` 实现了基于 `state.done` 条件重置偏好的 JAX 向量化机制。
- **比对结论**：数学逻辑与切分边界完全对齐。

---

### 3. 网络架构与权重初始化（Network Architecture & Initialization）
- **官方源码** (`lib/models/networks.py:48-120`):
  - `Actor`: 输入维数为 $17 + 2 = 19$，隐藏层为两层 $[400, 400]$，ReLU 激活，输出层使用 Tanh 缩放至 $[-1.0, 1.0]$；
  - `Critic`: Twin Critic 独立双网络，每个网络输入维数为 $17 + 2 + 6 = 25$，隐藏层为两层 $[400, 400]$，输出 2 维向量 $Q \in \mathbb{R}^2$；
  - 初始化：全部权重采用 `xavier_normal_`，偏置全 0 初始化。
- **Trinjq 实现** (`evorl/algorithms/mo_td3.py:PreferenceActor`, `TwinVectorCritic`):
  - Flax linen 模块结构，输入拼接维度严格为 19 和 25；
  - 隐藏层严格为 `(400, 400)`，激活函数为 ReLU，输出为 Tanh 乘 `max_action`；
  - 权重初始化严格为 `jax.nn.initializers.xavier_normal()`。
- **比对结论**：网络维度、层数、激活函数与初始化策略完全一致。

---

### 4. 偏好重标记 HER（Preference Relabeling HER）
- **官方源码** (`lib/common_ptan/experience.py:174-198`):
  - 每个 base transition 采样 $N_w = 3$ 个偏好：
$$
w_{\text{rnd}} \sim \mathcal{N}(0, I), \quad w = \text{round}\left( \frac{|w_{\text{rnd}}|}{\|w_{\text{rnd}}\|_1}, 3 \right)
$$
  - 先写入原始样本；
  - 缓冲区容量条件：
$$
\text{len}(\text{buffer}) > \text{start\_timesteps} \times \text{process\_count} = 10000 \times 10 = 100000
$$
  - 当满足容量条件后，紧随其后追加写入 3 个仅替换偏好为 $w$ 的重标记样本。
- **Trinjq 实现** (`evorl/replay_buffers/her.py:add_her_transitions`):
  - 采用 `lax.scan` 逐样本计算容量阈值与激活掩码；
  - 严格保持“先写原始、超过 100,000 条后追加 3 条重标记”的时序。
- **比对结论**：机制完全一致。

---

### 5. 悲观向量 Bellman 目标（Pessimistic Vector Bellman Target）
- **官方源码** (`lib/common_ptan/agent.py:MO_TD3_HER.learn`):
  - 目标动作平滑加入截断高斯噪声：
$$
a' = \text{clip}\left( \pi_{\text{target}}(s', w) + \text{clip}(\epsilon, -0.5, 0.5), -1.0, 1.0 \right), \quad \epsilon \sim \mathcal{N}(0, 0.2^2)
$$
  - 标量化悲观 Critic 索引选择：
$$
j = \arg\min_{i \in \{1, 2\}} w^T Q_i(s', a', w)
$$
$$
Q_{\text{pessimistic}} = Q_j(s', a', w) \in \mathbb{R}^2
$$
  - 向量 Bellman 目标：
$$
y = r + \gamma (1 - d) Q_{\text{pessimistic}}, \quad \gamma = 0.995
$$
- **Trinjq 实现** (`evorl/algorithms/mo_td3.py`):
  - `select_pessimistic_q_vector`、`add_target_policy_smoothing`、`vector_bellman_target`。
- **比对结论**：公式完全对齐。

---

### 6. 偏好-Q 对齐损失（Preference-Q Alignment Loss）—— Commit `24ca728` 补齐
- **官方源码** (`lib/common_ptan/agent.py:MO_TD3_HER.learn`):
  - 定义方向角：
$$
\text{angle}(u, v) = \arccos\left(\text{clip}\left(\frac{u \cdot v}{\|u\|_2 \|v\|_2}, 0.0, 0.9999\right)\right) \cdot \frac{180^\circ}{\pi}
$$
  - Critic 损失：
$$
\mathcal{L}_{\text{critic}} = \text{SmoothL1}(Q_1, y) + \text{SmoothL1}(Q_2, y) + \mathbb{E}[\text{angle}(w_{\text{proj}}, Q_1)] + \mathbb{E}[\text{angle}(w_{\text{proj}}, Q_2)]
$$
  - Actor 损失：
$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}[w^T Q_1(s, \pi(s, w))] + 10.0 \cdot \mathbb{E}[\text{angle}(w_{\text{proj}}, Q_1(s, \pi(s, w)))]
$$
- **Trinjq 实现** (`evorl/algorithms/mo_td3.py:pd_morl_critic_loss`, `pd_morl_actor_loss`):
  - 在 Commit `24ca728` 中补齐了此核心损失，并在 `tests/golden/test_pd_morl_golden.py` 中通过 PyTorch 源码导出的金标输入输出进行了单精度容差（$< 2 \times 10^{-6}$）比对。
- **比对结论**：完全对齐。

---

### 7. 多维插值器拟合与在线动态更新（Interpolator & Online Refit）—— Commit `94bfdd9`
- **官方源码** (`PD-MORL/train_Walker2d_MO_TD3_HER.py:88-95, 141-150`):
  - 初始基准解：从 `interp_objs_walker2d.txt` 读取 3 个点，行序对应 $[0, 1], [0.5, 0.5], [1, 0]$，做 **$L_2$ 归一化**后拟合线性 `RBFInterpolator`；
  - 在线触发条件：`if (process_episode_array > eval_cnt_ep).all():`（所有 Worker 完成 episode 数均超过计数器）；
  - 候选解评估：确定性模式在 3 个 Key 偏好下各跑 3 个 repeat 并取均值；
  - 替换判定：若 $w_k^T x_{\text{candidate}} > w_k^T x_{\text{old}}$ 则单点替换；
  - 在线重拟合：解集采用 **$L_1$ 归一化**，在 CPU 端重新调用 SciPy 拟合。
- **Trinjq 实现** (`evorl/evaluators/pd_morl.py`, `evorl/algorithms/mo_td3.py`):
  - 完整固化了官方初始点 `configs/artifacts/interp_objs_walker2d.txt`；
  - 在 `MOTD3Workflow._after_multi_steps` 中，实现了 `key_update_due` 判定并在 Host 端重拟合 `RBFInterpolator`，将状态更新回 JAX `AgentState`。
- **比对结论**：动态闭环已完全建立，初始 $L_2$ 与在线 $L_1$ 的范数切换亦严格吻合。

---

### 8. Worker 并行探索与更新节奏（Parallel Schedule）
- **官方源码** (`train_Walker2d_MO_TD3_HER.py:121-137`):
  - 10 个独立进程，每轮各产出 1 个样本（共 10 样本）；
  - 缓冲区超过 $1536$ 后开启学习；
  - 每收集 10 个样本，主进程连续调用 10 次 Critic 更新；
  - 当全局步数满足 `total_it % 10 == 0` 时，执行 1 次 Actor 更新和 Target 软更新；
  - 每个 Worker 在各自的最初 10,000 步输出均匀随机动作。
- **Trinjq 实现** (`evorl/algorithms/mo_td3.py`, `td3.py`):
  - `process_count = 10`, `actor_update_interval = 10`；
  - 通过 `parallel_actor_update_mask` 在 JAX `scan` 中调度每 10 次 Critic 更新触发 1 次 Actor 更新；
  - 每个 Worker 维护独立计数器，在 $< 10000$ 步时使用均匀随机动作掩码。
- **比对结论**：完全对齐。

---

### 9. 评测平面与指标计算（HV, Sparsity, Pareto）—— 存在关键差异
- **官方源码** (`lib/utilities/MORL_utils.py:100-140`):
  - 评估偏好网格：训练期 201 点（Step 0.005，3 repeat），终期 1001 点（Step 0.001，6 repeat）；
  - 评估种子：`eval_ep * 11`；
  - HV：`pymoo` 负收益、0 参考点；
  - **Sparsity**：**必须先对收益集调用 `NonDominatedSorting` 筛选出非支配解集合**，然后再对非支配解在各目标轴上排序计算相邻差分平方和除以 $N_{\text{nondom}} - 1$。
- **Trinjq 实现** (`evorl/evaluators/pd_morl.py:163`):
  - HV 计算、评估网格、评测种子严格吻合；
  - **关键缺陷**：`sparsity(returns)` **未对输入点集执行非支配 Pareto 过滤**，直接对全部 201 个评估点计算了稀疏度！
- **比对结论**：
  将包含大量被支配点的整个点集送入 Sparsity 计算，会导致分母使用 $200$ 而非真实的 Pareto 解个数，计算出的稀疏度数值被严重人为稀释，与官方统计口径不一致。

---

### 10. 执行拓扑（Single-GPU vs Multi-GPU / Multi-Seed）—— Commit `a1818fa`
- **官方源码**：
  纯单卡执行架构，无跨卡通信，跨种子实验通过单卡脚本独立运行。
- **Trinjq 演进**：
  作者在尝试跨卡数据并行后发现与单 Replay 集中式更新冲突，于 Commit `7f06626` 和 `a1818fa` 中果断进行了回滚，建立了单 GPU 独立 seed runner 体系（`scripts/run_pd_morl_seeds.py`）及聚合器（`scripts/aggregate_pd_morl_runs.py`）。
- **比对结论**：回到单卡独立种子架构是完全正确的决定，与官方对齐。

---

## 二、当前存在的具体问题与隐患汇总

1. **Sparsity 缺少非支配过滤（统计口径偏差）**：
   `evorl/evaluators/pd_morl.py:189` 中未先调用 `non_dominated_indices` 过滤点集，使得稀疏度指标偏小。
2. **`evaluate_actions` 缺少动作裁剪保护**：
   在 `mo_td3.py:evaluate_actions` 中直接返回网络输出，未像 `compute_actions` 那样加入 `jnp.clip(actions, self.action_low, self.action_high)`。
3. **Truncation 与 Termination 混合归零**：
   `done = jnp.maximum(termination, truncation)` 导致 500 步超时截断处丢失 Bootstrap。虽然作者在文档中解释这是为了 source-faithful 对齐旧版 Gym，但在现代 RL 标准下存在截断断崖偏差。
4. **外部环境依赖故障**：
   本地 Python 环境的 `orbax-checkpoint` 缺少导出文件，导致依赖工作流的测试无法在外部顺利导入。
