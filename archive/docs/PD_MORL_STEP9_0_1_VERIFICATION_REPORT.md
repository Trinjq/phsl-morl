> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# PD-MORL Step 9.0.1 Sparsity Metric Correction 最终核查报告

## 1. 检查概要

根据 `docs/guide/step9.15,md`（Step 9.0.1 Sparsity Metric Correction）的规范与约束，已对仓库代码、数学公式、测试用例与文档进行了逐项核对与独立执行验证。

---

## 2. 逐项核查结论

### 2.1 修复 Sparsity（Pareto 过滤对齐官方源码）

- **官方 PD-MORL 行为**：
  在 `compute_sparsity(obj_batch)` 中，首先调用 pymoo 的非支配排序过滤出当前集合的非支配前沿：
$$
\mathcal{P} = \{ \mathbf{r}^{(i)} \mid \mathbf{r}^{(i)} \text{ is non-dominated in } \mathcal{R} \}
$$
  随后按照各目标维度从小到大排序，计算相邻点间距的平方和均值：
$$
\text{Sparsity}(\mathcal{P}) = \frac{1}{|\mathcal{P}| - 1} \sum_{l=1}^{L} \sum_{i=1}^{|\mathcal{P}| - 1} \left( \hat{r}_{i+1, l} - \hat{r}_{i, l} \right)^2
$$
  当 $|\mathcal{P}| \le 1$ 时，稀疏度返回 $0.0$。
- **当前实现**：
  在 [evorl/evaluators/pd_morl.py](file:///e:/projects/evorl/evorl/evaluators/pd_morl.py) 中，`sparsity(returns)` 已加入非支配前沿过滤：
  ```python
  points = points[non_dominated_indices(points)]
  if len(points) <= 1:
      return 0.0
  ordered = np.sort(points, axis=0)
  return float(np.square(np.diff(ordered, axis=0)).sum() / (len(points) - 1))
  ```
- **符号约定与无 Double Sign Inversion**：
  官方 PyTorch 源码由于 pymoo 默认采用极小化约定，传入的是 $-\text{recovered\_objs}$；而 EvoRL 的 `non_dominated_indices` 原生采用极大化语义（`points >= point`），因此直接对正向收益做过滤，数学上与官方完全等价，不存在双重符号反转。

---

### 2.2 Aggregation Order 保持

在 `evaluate_morl` 中严格保持每轮 repeat 独立计算评测指标的聚合顺序：
1. 对第 $s$ 个 repeat 的评估收益 $\mathcal{R}_s$，独立进行 Pareto 过滤并计算其 Sparsity 值 $S_s$ 与超体积 $\text{HV}_s$；
2. 最终的评测稀疏度为各个 repeat 的稀疏度算术平均：
$$
\bar{S} = \frac{1}{S_{\text{repeats}}} \sum_{s=1}^{S_{\text{repeats}}} S_s
$$
3. 严格禁止先对各 repeat 的收益取平均再算单次 Sparsity；平均收益前沿仅作为结果可视化展示使用。
4. 评测网格参数符合协议：训练中评估使用 201 偏好 $\times$ 3 repeats；训练结束评估（training final）使用 1001 偏好 $\times$ 3 repeats；离线复现（offline paper report）使用 1001 偏好 $\times$ 6 repeats。

---

### 2.3 Golden Test 与回归测试

在 [tests/test_pd_morl_control_eval.py](file:///e:/projects/evorl/tests/test_pd_morl_control_eval.py) 中已加入以下测试并全部验证通过：
- **Golden 对齐测试**：采用包含支配点的固定集合：
$$
\mathcal{R} = \{ (1,4), (2,3), (3,2), (4,1), (1,1) \}
$$
  其中点 $(1,1)$ 被严格支配，验证 EvoRL 计算值与官方 PyTorch + pymoo `NonDominatedSorting` 数值在误差 $< 10^{-12}$ 内严格一致；
- **支配点有效过滤测试**：构造多个支配点案例，断言过滤后的 Sparsity 与不过滤时的原始计算值严格不相等，杜绝再次遗漏前沿过滤；
- **边界与重复点处理**：验证空集、单点及重复点均安全返回 $0.0$。

---

### 2.4 保持 evaluate_actions 无额外 Clip

- 遵循官方 `MO_TD3_HER` 确定性评估路径，Actor 网络最后一层采用：
$$
\mathbf{a} = \tanh(\mathbf{z}) \cdot \mathbf{a}_{\max}
$$
  Walker2d 的动作空间本即为 $[-1, 1]$，确定性评估直接返回网络输出；
- 不给 `evaluate_actions` 强行追加 `jnp.clip`；
- [tests/test_pd_morl_parallel.py](file:///e:/projects/evorl/tests/test_pd_morl_parallel.py) 中已通过断言保证评估动作的有限性（`jnp.isfinite`）与动作空间合法性（$[-1, 1]$）。

---

### 2.5 保持 Truncation Semantics（500 步 TimeLimit Combined Done）

- 严格遵循官方 Gym 0.21 / PD-MORL 实现行为：
$$
\text{done} = \text{terminated} \lor \text{truncated}
$$
- 500 步达到 TimeLimit 时的截断不执行 Bellman bootstrap（即目标值按 $y = \mathbf{r}$ 处理）；
- 不引入现代 RL 的 termination-only bootstrap mask，严格保持官方基线不变。

---

### 2.6 Orbax 环境审计

- 正式实验环境位于 Linux 集群环境（`/home/qiuquanj/miniforge3/envs/evorl`），其 Python 3.11.16 与 `orbax-checkpoint` 0.12.4 经过验证，checkpoint roundtrip 测试正常通过；
- 本地 Windows 解释器由于缺失 `orbax-checkpoint` 根层导出 shim，触发 `cannot import name 'logging'`，属于：
  **EXTERNAL ENVIRONMENT ISSUE — NON-BLOCKING**
- 绝不修改 PD-MORL 算法代码绕过 Orbax，保持工程纯洁性。

---

### 2.7 文档更新与历史问题记录

- [docs/PD_MORL_CONTROL_EVAL.md](file:///e:/projects/evorl/docs/PD_MORL_CONTROL_EVAL.md) 与 [docs/PD_MORL_REPRODUCTION_PROTOCOL.md](file:///e:/projects/evorl/docs/PD_MORL_REPRODUCTION_PROTOCOL.md) 均已更新；
- 明确记录了 Sparsity 计算须在每个 repeat 内部先执行 Pareto 过滤再套用相邻差分平方和公式；
- 明确记录了先前版本直接对全部收益计算 Sparsity 的缺陷及其修正过程；
- 精准区分了训练最终钩子（1001 $\times$ 3）与离线报告（1001 $\times$ 6）的协议差异。

---

### 2.8 约束与步数预算回归

运行 [scripts/test_pd_morl_counters.py](file:///e:/projects/evorl/scripts/test_pd_morl_counters.py) 与 [tests/test_pd_morl_interpolator.py](file:///e:/projects/evorl/tests/test_pd_morl_interpolator.py)：
- 10 Worker 并行探索与 10,000,000 真实环境交互步数预算核算一致；
- Prefill（1,530 步）严格被包含在 10M 总步数预算内，无重复计步；
- SciPy RBF 插值器 Golden 对齐与在线更新测试全部 PASS。

---

## 3. 验收结论

各项要求、数学公式、测试与约束均已确认修改完好并通过核验。

```text
STEP9.0.1 SPARSITY METRIC CORRECTION PASS
```
