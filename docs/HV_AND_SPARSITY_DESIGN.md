# PD-MORL 当前 Hypervolume (HV) 与 Sparsity (稀疏度) 设计与实现详解

## 一、指标数学定义与物理意义

在多目标强化学习（MORL）中，评价算法产出的策略集合在目标空间中的优劣，核心依赖两个互补的度量指标：**超体积（Hypervolume, HV）** 衡量前沿的**收敛性与广度**，**稀疏度（Sparsity）** 衡量解在目标前沿分布的**均匀度**。

### 1. 超体积（Hypervolume, HV）
对于给定的多目标收益集合 $P = \{r^{(1)}, r^{(2)}, \dots, r^{(N)}\} \subset \mathbb{R}^L$，以及参考点 $r_{\text{ref}} \in \mathbb{R}^L$，超体积定义为被点集 $P$ 所支配、且支配参考点 $r_{\text{ref}}$ 的目标空间超几何体积的测度：
$$
\text{HV}(P, r_{\text{ref}}) = \Lambda\left( \bigcup_{r \in P} [r_{\text{ref}}, r] \right)
$$
在 PD-MORL 的官方约定中：
- 目标为最大化收益，参考点固定为原点 $r_{\text{ref}} = \mathbf{0} \in \mathbb{R}^L$；
- 采用 `pymoo` 库计算时，由于 `pymoo` 遵循多目标优化的极小化惯例，需要将所有收益向量取负：
$$
-r \le \mathbf{0}
$$
然后以零向量为参考点计算极小化意义下的超体积。

### 2. 稀疏度（Sparsity）
稀疏度用于量化解在前沿分布的均匀程度。数值越小，代表解在目标空间上的间隔越均匀，没有出现聚团或大面积空白。

设参与计算的点集大小为 $N$（维度为 $L$）。对于每个目标维度 $j \in \{1, \dots, L\}$，将所有点在该维度的取值从小到大排序，记为：
$$
\tilde{r}_{1, j} \le \tilde{r}_{2, j} \le \dots \le \tilde{r}_{N, j}
$$
则稀疏度定义为所有目标维度上相邻有序点差值的平方和除以 $N - 1$：
$$
\text{Sparsity} = \frac{1}{N - 1} \sum_{j=1}^{L} \sum_{i=1}^{N - 1} \left( \tilde{r}_{i, j} - \tilde{r}_{i+1, j} \right)^2
$$
当点集大小 $N \le 1$ 时，定义：
$$
\text{Sparsity} = 0
$$

---

## 二、当前仓库的代码实现（`evorl/evaluators/pd_morl.py`）

当前实现位于 `evorl/evaluators/pd_morl.py`，整体设计分为基础度量函数和评测聚合调度两部分：

### 1. 基础度量函数

#### (1) Hypervolume 计算
```python
def hypervolume(returns: np.ndarray) -> float:
    """Official pymoo minimization adapter with zero reference."""
    points = np.asarray(returns, dtype=np.float64)
    return float(HV(ref_point=np.zeros(points.shape[1]))(-points))
```
- 使用 `from pymoo.indicators.hv import HV`；
- 输入形状为 $[N, L]$ 的收益数组；
- 以全零向量为参考点，输入 `-points` 进行计算，返回标量浮点数。

#### (2) Sparsity 计算
```python
def sparsity(returns: np.ndarray) -> float:
    points = np.asarray(returns, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("returns must have shape [N, L]")
    if len(points) <= 1:
        return 0.0
    ordered = np.sort(points, axis=0)
    return float(np.square(np.diff(ordered, axis=0)).sum() / (len(points) - 1))
```
- `np.sort(points, axis=0)`：沿着第 0 轴对每个目标维度独立升序排序；
- `np.diff(ordered, axis=0)`：计算相邻点差分；
- `np.square(...).sum()`：平方并在所有维度上求和；
- 除以 $N - 1$ 返回标量。

#### (3) 非支配排序（Pareto 筛选）
```python
def non_dominated_indices(returns: np.ndarray) -> np.ndarray:
    """Maximization non-dominated front, retaining duplicate entries."""
    points = np.asarray(returns)
    if points.ndim != 2:
        raise ValueError("returns must have shape [N, L]")
    dominated = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        dominated[i] = np.any(
            np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        )
    return np.flatnonzero(~dominated)
```
- 严格遵循最大化非支配准则：若存在点 $p'$ 满足各维度 $p' \ge p$ 且至少一维 $p' > p$，则点 $p$ 被支配；
- 允许多个完全相等的非支配重复点同时保留。

---

### 2. 评测协议与多 Repeat 聚合架构（`evaluate_morl`）

在 `evaluate_morl` 函数中，评测流程设计如下：

1. **测试网格划分**：
   - 训练期全量评估网格（Step 0.005，共 201 个偏好点）：
     $$
     w \in \{ [0.0, 1.0], [0.005, 0.995], \dots, [1.0, 0.0] \}
     $$
   - 离线终期评估网格（Step 0.001，共 1001 个偏好点）。
2. **随机种子设计**：
   采用官方指定的 $11 \times \text{repeat}$ 种子序列：
   $$
   \text{seeds} = [0, 11, 22] \quad (\text{训练期 3-repeat})
   $$
   $$
   \text{seeds} = [0, 11, 22, 33, 44, 55] \quad (\text{离线 6-repeat})
   $$
3. **评估数据流**：
   - 在确定性动作模式（`deterministic=True`）下，依次遍历偏好网格采集未折现的累计收益向量；
   - 得到形状为 $[S, K, L]$ 的收益张量，其中 $S$ 为 repeat 次数，$K$ 为偏好数量，$L$ 为目标数量（Walker2d 为 2）。
4. **聚合计算顺序**：
   - **Per-repeat 计算**：对每个 repeat 生成的 $K$ 个点分别计算超体积 $\text{HV}_s$ 与稀疏度 $\text{Sparsity}_s$；
   - **求均值**：
     $$
     \overline{\text{HV}} = \frac{1}{S} \sum_{s=1}^{S} \text{HV}_s
     $$
     $$
     \overline{\text{Sparsity}} = \frac{1}{S} \sum_{s=1}^{S} \text{Sparsity}_s
     $$
   - **前沿收益提取**：对跨 repeat 的收益求平均值 $\overline{r}_k = \frac{1}{S} \sum_s r_{s, k}$，最后在平均收益上做非支配排序，得到 Pareto 前沿。

---

## 三、与官方原版 PD-MORL（`MORL_utils.py`）的比对与关键差异

对比 `E:\PD-MORL\PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm\lib\utilities\MORL_utils.py` 中的原生实现：

### 1. 一致的设计
1. **HV 参考点与极小化适配**：
   官方采用 `perf_ind = get_performance_indicator("hv", ref_point = np.zeros(2))`，输入 `-recovered_objs`；当前实现使用 `pymoo` 计算 `HV(ref_point=np.zeros(2))(-points)`，二者数学与数值完全一致。
2. **Sparsity 计算公式形式**：
   分维度独立排序后计算相邻差分平方和除以 $N - 1$，数学公式形式完全一致。
3. **评测种子与 Repeat 协议**：
   种子步长 $11 \times \text{repeat}$、训练期 3-repeat、离线 6-repeat 的设计与官方完全吻合。

### 2. 当前实现的关键遗漏与潜在缺陷（重点关注）

在官方源码 `MORL_utils.py` 中，`compute_sparsity` 的前两行为：
```python
def compute_sparsity(obj_batch):
    non_dom = NonDominatedSorting().do(obj_batch, only_non_dominated_front=True)
    objs = obj_batch[non_dom]
    ...
```
**官方源码在计算 Sparsity 之前，必须显式调用非支配排序，只在真正处于 Pareto 非支配前沿的点集上计算稀疏度！**

- **官方逻辑**：
  若 201 个偏好采样出来的点中只有 30 个点是非支配前沿点，官方仅对这 30 个前沿点排序并除以 $30 - 1 = 29$；被支配的劣解不参与 Sparsity 的统计。
- **当前仓库现状**：
  在 `pd_morl.py` 中：
  ```python
  sp = np.asarray([sparsity(row) for row in returns])
  ```
  直接将包含全部 201 个（或 1001 个）采样点的数组 `row` 传入了 `sparsity()` 函数，内部**没有进行非支配过滤**。
- **潜在影响**：
  由于包含了大量被支配解，点集被人工密集化，分母使用了全量样本数 $N - 1 = 200$，导致算出的稀疏度数值会被大幅稀释（数值偏小），且与官方论文中汇报的 Sparsity 基准数值存在统计口径差异。
