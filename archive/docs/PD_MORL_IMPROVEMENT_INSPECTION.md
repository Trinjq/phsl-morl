> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PD-MORL 实现最新改善与进展复查报告

## 审查说明

本报告针对用户请求：“再检查一遍看看是否有改善”，在上一轮全景对照的基础上，对作者 **Trinjq** 截至 2026-09-27 19:12 的最新提交记录（第 14 次提交 `5d550a7`）、工作区最新改动（Step 9.0/9.1 复现协议与步数预算合同）以及此前指出的关键问题进行逐一核查。

---

## 一、本次复查发现的重要改善与新增进展

### 1. 新增第 14 次提交：`5d550a7` (2026-09-27 18:25:48)
作者提交了 `PD_MORL`：
- **修复 Worker 完成 Episode 统计维度问题**：
  在 `evorl/algorithms/mo_td3.py` 的 `completed_episodes_by_worker` 中，增加了对一维 `dones` 数组的防御性判断：
  ```python
  dones = jnp.asarray(dones, dtype=jnp.uint32)
  if dones.ndim == 1:
      worker_ids = jnp.arange(dones.shape[0]) % process_count
      return (
          jnp.zeros((process_count,), dtype=jnp.uint32)
          .at[worker_ids]
          .add(dones)
      )
  ```
  避免了单步或扁平化 Done 向量调用 `.sum(axis=0)` 时造成的索引或维度不符问题，确保了在线 Key 偏好更新触发条件 `(episode_count > eval_cnt_ep).all()` 的统计精确性。

---

### 2. 重大改善：确立并固化官方步数预算合同（Step 9.0 / Step 9.1）
在新增的协议文档 `docs/PD_MORL_REPRODUCTION_PROTOCOL.md`、配置文件 `configs/experiment/pd_morl_walker_reproduction.yaml` 以及 `scripts/train.py` 中，作者进行了极具深度的核查与纠偏，彻底厘清了官方训练步数的语义：

#### (1) 厘清官方训练总步数定义
- **此前歧义**：早期配置文件 `mo-td3.yaml` 中设置 `total_timesteps: 1000000`（100 万步），但这实际上只是单 Worker 的步数；
- **官方源码对照**：官方 `train_Walker2d_MO_TD3_HER.py` 中主循环为：
  ```python
  for ts in range(0, Cp * N, Cp): # Cp=10, N=1,000,000
  ```
  每个子进程跑 $N = 1,000,000$ 步，因此整个环境的真实物理交互步数必须是：
$$
\text{Total Valid Transitions} = N \times C_p = 1,000,000 \times 10 = 10,000,000 \text{ 步}
$$
  Trinjq 在正式复现配置中明确修正为 $10,000,000$ 步（10M），杜绝了因步数少了一个数量级而导致实验不充分的问题。

#### (2) Prefill 步数的重算纠正
- 官方的 $N = 1,000,000$ 已经包含了每个 Worker 前 153 步的初始随机预填充（Prefill 共 1,530 步）；
- 纠正了在此基础上“额外再加 1,530 步”的双重计数草案，使得：
$$
\text{Post-prefill Rounds} = 1,000,000 - 153 = 999,847 \text{ 轮}
$$
$$
\text{Critic Updates} = 999,847 \times 10 = 9,998,470 \text{ 次}
$$
$$
\text{Actor Updates} = 9,998,470 \div 10 = 999,847 \text{ 次}
$$
- 在 `scripts/train.py` 中增加了 `_source_budget` 与 `_assert_source_budget`，强制在训练结束后断言步数严丝合缝匹配。并通过 `scripts/test_pd_morl_counters.py` 验证了计数器算术逻辑的完全正确。

---

### 3. 重大改善：评估快照持久化机制（`_write_evaluation_snapshot`）
在 `evorl/algorithms/mo_td3.py` 的工作流中：
- 增加了 `_write_evaluation_snapshot` 方法；
- 每次触发全量评估（`training_full`）以及训练结束后的终期评估（`training_final`）时，系统会将当前迭代轮次、各 Repeat 的收益矩阵、平均收益、$\text{HV}$、$\text{Sparsity}$ 以及 Pareto 前沿点集完整序列化为 JSON 行追加写入 `pd_morl_evaluations.jsonl` 文件；
- 避免了评测数据仅存于控制台日志或内存中，确保了复现实验过程数据的完整可追溯性。

---

### 4. 优化：`total_it` 全局更新计数器显式追踪
在 `AgentState.extra_state` 中增加了：
```python
total_it = jnp.uint32(0)
```
并在每轮工作流步进中显式累加 $10$（即 `total_it += process_count`），严格对齐了官方 PyTorch 源码中 `self.total_it` 驱动 Actor 和 Target 延迟更新（`total_it % 10 == 0`）的时序语义。

---

## 二、仍未改善或需要继续留意的细节

经过深入检查代码，上一轮指出的以下两处细节在当前代码中**尚未修改**：

### 1. `sparsity` 计算依然缺失 Pareto 非支配过滤（主要统计口径偏差）
- **代码位置**：`evorl/evaluators/pd_morl.py:163`
- **现状**：
  在 `pd_morl.py` 中，`sparsity` 函数依然是直接对输入的收益矩阵按目标排序后计算差分平方和除以 $N - 1$：
$$
\text{Sparsity} = \frac{1}{N - 1} \sum_{j=1}^{L} \sum_{i=1}^{N - 1} \left( \tilde{r}_{i+1, j} - \tilde{r}_{i, j} \right)^2
$$
  而在调用处（第 189 行），直接传入了单次 Repeat 的全部 201 个评估点（未过滤被支配解）。
- **官方源码对比** (`lib/utilities/MORL_utils.py:130`):
  ```python
  def compute_sparsity(obj_batch):
      non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)
      objs = obj_batch[non_dom]
      ...
  ```
  官方是在通过非支配排序提纯后的 Pareto 前沿点集上计算稀疏度。如果输入全部 201 个点（包含大量劣解），分母使用 $200$ 会使得稀疏度数值被大幅人为稀释偏小。
- **改善建议**：在 `sparsity(returns)` 前先调用 `indices = non_dominated_indices(returns)`，仅对 `returns[indices]` 计算稀疏度。

### 2. `evaluate_actions` 仍缺少动作空间边界裁剪
- **代码位置**：`evorl/algorithms/mo_td3.py:288-299`
- **现状**：
  `compute_actions` 中做了 `jnp.clip(policy_actions, self.action_low, self.action_high)`，但在 `evaluate_actions` 中直接返回了 `actor_network.apply(...)`。
- **评估**：由于 Walker2d 环境动作空间上下界对称且经过 Tanh 刚好落在 $[-1, 1]$，在当前 Walker 任务中不会发生实际越界；但若作为通用 MORL Agent 推广至动作空间上下界不对称的环境，仍缺乏保护。

---

## 三、复查结论

相比于上一轮检查，代码库取得了非常显著且实质性的改善：
1. **预算与步数逻辑完全成熟**：厘清并锁定了官方 10M 真实步数的语义，消除了步数少一个数量级和预填充重复计数的风险；
2. **训练过程监控与数据持久化建立**：通过 JSONL 快照完整记录了所有阶段的 HV、Sparsity 和 Pareto 收益前沿；
3. **多卡到单卡独立 Seed 的回归**：彻底解决了多卡并行与 PD-MORL 集中式 Replay 的机制冲突；
4. **遗留建议**：后续仅需在 `pd_morl.py` 中为 `sparsity` 补充一行非支配过滤，即可实现与官方源码 100% 统计口径的严密一致。
