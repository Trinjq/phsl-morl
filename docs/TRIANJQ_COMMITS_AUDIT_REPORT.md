# 基于官方 PD-MORL 源码对照的 Trinjq 提交产物深度审查报告

## 审查说明

本报告对照位于 `E:\PD-MORL\PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm` 的官方 PD-MORL 原始实现（基于 PyTorch / OpenAI Gym），对作者 **Trinjq** 在 `evorl` 中添加的 6 条 commit 记录及其全部产物（算法代码、配置文件、单元测试及文档）进行逐行逐项事实比对，杜绝任何猜测，列出所有不合理、不一致与存在缺陷的部分。

---

## 一、最核心的算法缺陷：PD-MORL 核心对齐损失在主循环中完全缺失（Dead Code）

### 1. 官方 PD-MORL 源码的真实行为
检查官方源码文件 `lib/common_ptan/agent.py` 中的 `MO_TD3_HER.learn` 方法（行 370-435）：
- **Critic 损失**：除了双 Critic 对向量 TD 目标的 Smooth L1 损失外，显式计算了当前 $Q$ 向量与插值器映射偏好向量 $w_{\text{proj}} = I(w)$ 之间的方向角惩罚项：
$$
\text{angle\_term}_1 = \arccos(\text{clip}(\text{cosine\_similarity}(w_{\text{proj}}, Q_1), 0.0, 0.9999)) \cdot \frac{180^\circ}{\pi}
$$
$$
\text{angle\_term}_2 = \arccos(\text{clip}(\text{cosine\_similarity}(w_{\text{proj}}, Q_2), 0.0, 0.9999)) \cdot \frac{180^\circ}{\pi}
$$
$$
\mathcal{L}_{\text{critic}} = \text{SmoothL1}(Q_1, y) + \text{SmoothL1}(Q_2, y) + \mathbb{E}[\text{angle\_term}_1] + \mathbb{E}[\text{angle\_term}_2]
$$
- **Actor 损失**：不仅最大化标量化收益 $-w^T Q_1$，而且必须最小化与插值目标方向的夹角：
$$
\mathcal{L}_{\text{actor}} = -\mathbb{E}[w^T Q_1(s, \pi(s, w))] + \lambda \cdot \mathbb{E}[\text{angle\_term}]
$$
其中系数 $\lambda = \text{actor\_loss\_coeff} = 10.0$（在 `lib/utilities/settings.py` 中严格指定）。

这正是 PD-MORL（Preference-Driven Multi-Objective Reinforcement Learning）的核心灵魂所在：通过插值器 $I(w)$ 将单纯形上的偏好映射到 Pareto 前沿的目标空间方向，并利用方向角损失（Directional Alignment Loss）迫使策略和价值网络朝向该方向收敛。

### 2. Trinjq 提交产物的现状
- 在 `evorl/algorithms/mo_td3.py` 的第 243-301 行中：
  - `critic_loss` 仅计算了 `twin_smooth_l1_loss(q_values, q_target)`，**完全没有计算 Critic 的方向角损失**；
  - `actor_loss` 仅计算了 `-scalarize(q_values[..., 0, :], preference).mean()`，**完全没有计算 Actor 的方向角损失**。
- 尽管 Trinjq 在 Commit 4 中写了 `evorl/utils/morl_math.py`（实现了 `directional_angle`），并在 Commit 6 中写了 `evorl/utils/pd_morl_interpolator.py`（实现了 `interpolate`），但在整个 `evorl/` 算法库中，**没有任何一行训练代码调用了 `interpolate` 或 `directional_angle`**！
- **审查结论**：目前已提交的 `mo_td3.py` 算法在数学与机制上**仅仅是普通的多目标 TD3（对应官方源码中仅用于预训练端点 Key 策略的 `MO_TD3_HER_Key` 简化版本）**，根本没有实现真正的 PD-MORL 核心算法，名不副实，且新引入的插值器和数学函数全部沦为无法发挥作用的死代码（Dead Code）。

---

## 二、插值器模块（Interpolator）的不合理之处

### 1. `key_preferences` 强制使用 `np.unique` 造成排序混乱隐患
- **位置**：`evorl/utils/pd_morl_interpolator.py:33`
- **代码**：
  ```python
  def key_preferences(num_objectives: int) -> np.ndarray:
      if num_objectives < 2:
          raise ValueError("num_objectives must be at least 2")
      keys = np.vstack(
          (np.eye(num_objectives), np.full(num_objectives, 1 / num_objectives))
      )
      return np.unique(keys, axis=0)
  ```
- **对照官方源码**：
  在官方源码 `train_Walker2d_MO_TD3_HER.py:91-94` 中，基准偏好是通过在全局评估测试网格上均匀线性抽样得到的：
  ```python
  idx_w_batch = np.round(np.linspace(0, len(w_batch_test)-1, num=len(x))).astype(int)
  w_batch_interp = w_batch_test[idx_w_batch]
  ```
- **问题分析**：
  Trinjq 自行定义为单位阵加上均匀向量，但在末尾调用了 `np.unique(keys, axis=0)`。`np.unique` 默认会对多维数组按字典序进行升序重排：
  - 在 3 目标场景下，原始拼接顺序为 $[e_1, e_2, e_3, \text{uniform}]$；
  - 经 `np.unique` 排序后，向量顺序变为 $[e_3, e_2, \text{uniform}, e_1]$；
  - 若调用者按照常规的 $1, 2, 3$ 目标顺序提供最优解向量 `key_solutions`，解与偏好向量将**完全错位**；
  - 实际上，对于任意 $m \ge 2$，单位阵与均匀向量之间不存在重复行，调用 `np.unique` 纯属有害冗余。

### 2. 缺少在线更新机制与生命周期闭环
- **官方源码事实**：
  官方源码在主训练循环（`train_Walker2d_MO_TD3_HER.py:141-151`）中：
  - 维护了一个持久评估周期：当所有子进程完成一定 episode 后，调用 `eval_agent_interp` 评估当前策略在各个 Key 偏好下的表现；
  - 比较新解与旧解在对应偏好下的标量化得分；若有提升，则替换对应解并以 $L_1$ 范数重新归一化后，重新拟合 `RBFInterpolator`。
- **Trinjq 实现问题**：
  Trinjq 仅写了一个静态的 `PDMORLInterpolatorState` 类和纯函数，根本没有在 `MOTD3Workflow` 中接入定期评估与动态重拟合插值器的流程。插值器在训练中处于完全静止脱节状态。

---

## 三、动作评估与输出边界缺陷（Action Clipping 缺失）

- **位置**：`evorl/algorithms/mo_td3.py:224-226`
- **代码**：
  ```python
  def evaluate_actions(
      self, agent_state: AgentState, sample_batch: SampleBatch, key: chex.PRNGKey
  ) -> tuple[Action, PolicyExtraInfo]:
      ...
      actions = self.actor_network.apply(
          agent_state.params.actor_params, obs, preference
      )
      return actions, PyTreeDict(preference=preference)
  ```
- **对照官方源码与自身代码**：
  - 在官方源码中，动作前向始终保证严格限制在合法范围内；
  - 在 Trinjq 自己的 `compute_actions`（第 199-201 行）中，显式编写了边界截断：
    ```python
    policy_actions = jnp.clip(policy_actions, self.action_low, self.action_high)
    ```
  - 但在 `evaluate_actions` 中，却漏掉了此截断逻辑！当动作空间的上下界不对称（如 `low=[-2.0, -1.0], high=[2.0, 3.0]`）时，由于网络最后一层直接使用标量最大值进行缩放，输出的动作将直接越界。

---

## 四、时序差分更新对 Truncation 的处理不当

- **位置**：`evorl/algorithms/mo_td3.py:266`
- **代码**：
  ```python
  done = jnp.maximum(env_extras.termination, env_extras.truncation)
  q_target = vector_bellman_target(
      sample_batch.rewards, done, next_q, self.discount
  )
  ```
- **问题分析**：
  - 强化学习标准更新方程要求：
$$
y = r + \gamma (1 - \text{termination}) Q(s', a')
$$
  - 超时截断（`truncation`）只是由于人为设定的最大步数限制（例如 500 或 1000 步）导致 episode 中断，环境物理状态并未死亡，**必须继续 Bootstrap 下一状态的价值**；
  - 将 `truncation` 视作 `done` 强行将未来折现价值归零，会导致智能体在靠近最大步数时面临虚假的“死亡惩罚”，造成严重的长期价值塌陷（Value Drop-off）；
  - 尽管作者在文档中辩称是为了对齐旧版 Gym 将两者合一的源码行为，但在具备清晰 `termination` 与 `truncation` 划分的现代环境体系下，保留这种缺陷并不合理。

---

## 五、测试套件存在硬编码环境阻碍

- **位置**：`tests/test_mo_td3.py:153-155`
- **代码**：
  ```python
  def test_real_walker_replay_and_delayed_updates_on_gpu():
      assert jax.default_backend() == "gpu"
      assert jax.devices()[0].platform == "gpu"
      assert "CudaDevice" in repr(jax.devices()[0])
  ```
- **问题分析**：
  测试套件直接断言当前运行环境必须是 GPU 设备且字符串包含 `CudaDevice`，未加 `@pytest.mark.skipif` 保护。这导致任何在 CPU 机器、普通轻量环境或自动化持续集成（CI）服务器上运行 `pytest` 时，测试直接无故挂起报错。

---

## 六、对通用框架代码（`td3.py`）的过度侵入与指标不一致

- **位置**：`evorl/algorithms/td3.py:434-512`
- **问题分析**：
  为了支持多目标并行的更新频率掩码，Trinjq 直接修改了底层的通用标量算法基类 `TD3Workflow`，硬性塞入了 `parallel_actor_mask` 分支：
  1. 破坏了基础算法库与扩展算法之间的解耦边界；
  2. 统计口径出现偏差：标量 TD3 对所有 update 步的损失采用 `scan_and_mean` 求均值，而 Trinjq 在并行分支中直接取了最后一次更新的值：
     ```python
     critic_loss = critic_losses[-1]
     critic_loss_dict = jtu.tree_map(lambda x: x[-1], critic_loss_dicts)
     ```
     完全丢弃了前 $K-1$ 次更新的指标，导致向监控日志上报的 Loss 产生异常剧烈的方差波动。

---

## 七、总结与整改清单

| 序号 | 缺陷类别 | 涉及文件 | 严重程度 | 具体整改要求 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **核心机制缺失** | `mo_td3.py` | **严重** | 将 `interpolate` 和 `directional_angle` 正式接入 Critic Loss 和 Actor Loss，补全 PD-MORL 核心损失。 |
| 2 | **逻辑错误/隐患** | `pd_morl_interpolator.py` | **高** | 移除 `key_preferences` 中的 `np.unique`，消除多目标偏好与最优解错位风险。 |
| 3 | **动作边界漏洞** | `mo_td3.py` | **高** | 在 `evaluate_actions` 中补充对 `action_low` 与 `action_high` 的 `jnp.clip`。 |
| 4 | **环境强依赖阻碍** | `test_mo_td3.py` | **中** | 为 GPU 断言测试添加 `@pytest.mark.skipif`，保证 CPU 环境测试可正常通过。 |
| 5 | **指标口径失真** | `td3.py` | **中** | 将并行更新的指标统计由只取切片 `[-1]` 修正为对 $K$ 步更新取均值。 |
| 6 | **理论偏差** | `mo_td3.py` | **中** | 区分 `termination` 与 `truncation`，停止在截断步错误清零未来价值。 |
