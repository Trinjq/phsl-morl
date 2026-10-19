> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入 Step 5.4：

PD-MORL Multi-Dimensional Preference Interpolator I(w)

这是高风险模块。

本阶段目标是：

1. 先独立复现官方 interpolator；
2. 建立可供 EvoRL/JAX 最终调用的纯 JAX forward；
3. 做 SciPy ↔ JAX Golden Test；
4. 暂时不要接入 MO-TD3 actor/critic loss；
5. 暂时不要接入完整 training loop。

==================================================
一、SOURCE AUDIT
================

先阅读并核查：

1. PD-MORL 论文 Appendix B.1.3；
2. PD-MORL 官方：
   - train_Walker2d_MO_TD3_HER.py
   - train_Walker2d_MO_TD3_HER_Key.py
   - lib/common_ptan/agent.py
3. 当前 EvoRL / JAX 相关结构。

重点回答：

1. key preference set 如何构造；
2. Walker 二目标下 key preferences 的实际顺序；
3. key solutions 从哪里来；
4. key-training 是否使用 HER；
5. key-training 是固定 preference 还是动态 preference；
6. 初始 key solutions 如何 normalization；
7. 在线更新后的 key solutions 如何 normalization；
8. 官方是否使用：
   scipy.interpolate.RBFInterpolator
9. 官方传给 RBFInterpolator 的精确参数；
10. kernel 是否为 linear；
11. smoothing / degree / neighbors / epsilon 是否显式设置；
12. 如果没有显式设置，SciPy 使用什么默认行为；
13. interpolator 在线什么时候被更新；
14. key solution 替换的判断条件是什么；
15. update 后是否重新 fit 一个新的 interpolator；
16. MO-TD3 中 I(w) 的实际输出如何被使用。

先输出：

SOURCE AUDIT

不要先修改代码。

==================================================
二、Key Preferences
===================

不要自己设计 key preference。

按照论文和官方源码。

对于 L 个 objectives：

key preference set 包括：

- 每一个 one-hot preference；
- 一个均匀 preference：

  [1/L, ..., 1/L]

对于当前 Walker L=2：

论文语义为：

[1,0]
[0.5,0.5]
[0,1]

但最终代码顺序必须以官方 Walker source / key-training
实际生成顺序为准。

如果源码实际顺序为：

[0,1]
[0.5,0.5]
[1,0]

则必须复现源码顺序。

不要为了与论文书写顺序一致而重排。

==================================================
三、Key Solutions 来源
======================

非常重要：

本阶段不要凭空生成“官方 key solutions”。

连续控制任务中的 key solutions
来自独立的 fixed-preference MO-TD3 training：

- 一个 key preference 对应一个固定 preference agent；
- 不使用 HER；
- 训练后 evaluation；
- 保存该 preference 对应的 objective return。

本 Step5.4：

不要重新实现整套 key-training pipeline。

允许：

A. 从已有官方 key-training artifact 加载；
B. 如果当前仓库还没有 artifact，
   使用明确标记的 synthetic / test fixture
   来测试 interpolator 数学行为。

必须在文档中明确：

synthetic key solutions
不是官方 Walker key solutions。

不得把测试 fixture 当作论文实验数据。

==================================================
四、Reference Implementation
============================

实现一个离线 reference path。

允许：

NumPy
SciPy

使用官方等价：

scipy.interpolate.RBFInterpolator(
    key_preferences,
    normalized_key_solutions,
    kernel="linear",
    ...
)

其中 `...` 必须严格按官方源码。

如果官方只写：

RBFInterpolator(..., kernel="linear")

那么 reference implementation
必须首先确认当前 SciPy 版本中其默认：

- smoothing
- degree
- epsilon
- neighbors

等行为。

不要自己假设内部公式。

Reference implementation：

不要求 JIT；
不进入训练 step；
只用于 Golden Test 和 host-side refit。

==================================================
五、Normalization
=================

必须区分两条源码路径。

1. Initial interpolator

官方源码初始化时：

x_unit =
x / ||x||_2

即：

L2 normalization

2. Online update interpolator

官方源码在线更新后：

x_unit =
x / ||x||_1

即：

L1 normalization

这是源码真实差异。

不要“修复”为统一 L2。

不要“修复”为统一 L1。

分类：

SOURCE-FAITHFUL:
initial L2
online-update L1

如果统一 normalization：
DEVIATION

需要增加数值安全保护时，
单独标记为：

FRAMEWORK-ADAPTATION

且不能改变正常非零 solution 的结果。

==================================================
六、JAX Implementation 的边界
=============================

把 JAX 部分拆成：

A. fit/refit control plane
B. forward data plane

---

A. fit / refit
--------------

本阶段允许两种实现：

方案 1：
host-side SciPy fit/refit
+
把 fixed-shape parameters 转成 JAX arrays

方案 2：
实现纯 JAX 等价 RBF fit

但只有在方案 2
通过逐值 Golden Test 后，
才能称为等价。

如果无法严格复现 SciPy fit：

不要硬写一个“差不多的 RBF”。

优先使用：

host-side SciPy fit
+
pure-JAX forward

作为第一版。

---

B. JAX forward
--------------

training-facing JAX forward 必须：

- pure JAX；
- fixed shape；
- 无 SciPy；
- 无 NumPy runtime；
- 无 Python callback；
- 可 jax.jit；
- 可 batch；
- 输出：

  wp = I(w)

shape：

single:
w  [L]
wp [L]

batch:
w  [B,L]
wp [B,L]

==================================================
七、不要在 training JIT 中做 SciPy
==================================

training-step JIT 内禁止：

- scipy.interpolate.RBFInterpolator；
- Python interpolator object；
- host_callback；
- 动态增删 knots；
- 动态 shape；
- 每个 gradient step 重新 fit。

training JIT 内只允许：

fixed parameters
+
pure-JAX interpolation forward

==================================================
八、Interpolator State
======================

设计一个固定 shape 的 interpolator state。

至少应能表达：

- key_preferences；
- normalized_key_solutions；
- JAX forward 所需的 RBF coefficients；
- SciPy linear-RBF 所需的其他固定参数。

不要把：

scipy RBFInterpolator Python object

直接放进：

workflow state
AgentState
JAX PyTree training path。

所有训练可见状态必须是：

JAX arrays / static metadata。

==================================================
九、Golden Test
===============

这是本阶段最重要的 PASS 条件。

构造同一组：

key_preferences [K,L]
key_solutions   [K,L]

分别经过：

Reference SciPy path
JAX path

比较：

wp_ref
wp_jax

测试 query 至少包括：

1. key knots 自身；
2. preference 空间内部点；
3. midpoint；
4. 接近边界但在 simplex 内的点；
5. batch inputs。

Walker L=2 示例 query 可以包括：

[0.0,1.0]
[0.1,0.9]
[0.25,0.75]
[0.5,0.5]
[0.8,0.2]
[1.0,0.0]

但 Golden Test 的 key solutions
使用固定 deterministic fixture。

==================================================
十、Golden Test 必须覆盖两种 normalization
==========================================

分别做两套测试。

A. Initial-fit path

key solutions
→ L2 normalize
→ SciPy fit
→ JAX representation
→ compare outputs

B. Online-update path

updated key solutions
→ L1 normalize
→ SciPy refit
→ JAX representation
→ compare outputs

不要只测试 L2。

否则无法验证官方在线更新路径。

==================================================
十一、误差报告
==============

不要只写：

np.allclose PASS

必须报告：

max absolute error
mean absolute error
max relative error

并记录：

reference dtype
JAX dtype

如果：

SciPy reference = float64
JAX training = float32

则明确记录由 dtype 带来的误差。

测试 tolerance
必须根据实际误差决定并记录，
不要随意设置很松的 tolerance。

==================================================
十二、Update Semantics
======================

本阶段只实现：

纯函数 / host-side control logic

来表达官方 update 语义。

不要真正接入在线训练。

需要实现并测试类似：

update_key_solution(
    old_solution,
    candidate_solution,
    key_preference,
)

但其判断条件必须先从官方源码确认。

必须核查是否实际为：

candidate_score =
w_key^T candidate_return

old_score =
w_key^T old_return

只有 candidate_score 改善时
替换该 key solution。

如果官方源码还有：

- averaging；
- repeat evaluation；
- comparison tolerance；
- non-dominated check；

都必须以源码为准。

不要根据论文文字自行补充。

更新 key solution 后：

按官方在线路径：
L1 normalization
→ refit interpolator

==================================================
十三、Update Timing
===================

论文说 training 中会更新 interpolator，
并提到对 key preferences 做 evaluation。

但最终 update timing
必须以官方 source 为准。

明确记录：

- 是每 episode；
- 每 outer-loop；
- 每 evaluation interval；
- 还是满足其他条件。

本阶段：

只实现 host-boundary update/refit API。

不要接到：

environment step
learner step
actor update
critic update

中。

==================================================
十四、wp 的语义
===============

Interpolator 输出：

wp = I(w)

不要在 I(w) 输出之后额外：

- L1 normalize；
- L2 normalize；
- simplex projection；
- clipping；
- softmax。

官方 MO-TD3 后续直接使用 interpolator output
进入 cosine / directional-angle 相关计算。

Step5.4 只负责产生 wp。

暂时不要实现：

angle loss
actor angle
critic angle

这些属于后续步骤。

==================================================
十五、重要区分
==============

必须明确区分：

w
=

原始 preference

key preference
==============

建立 interpolator 的 anchor input

key solution
============

对应 key preference 的 vector return/objective solution

normalized key solution
=======================

interpolator target

wp
==

I(w) 的 projected preference

不要把：

wp

误写成概率分布。

它不要求：

sum(wp) == 1

也不要自动把它投影回 simplex。

==================================================
十六、JIT / VMAP Tests
======================

JAX forward 必须测试：

- eager；
- jax.jit；
- jax.vmap 或原生 batch；
- float32；
- single [L]；
- batch [B,L]。

fit/refit 本身：

如果仍使用 SciPy host-side，
不要求 JIT。

不要为了“全 JAX”
把 SciPy fit 强行塞进 JIT。

==================================================
十七、暂时禁止接入训练
======================

Step5.4 完成后：

不要修改：

- MO-TD3 critic loss；
- MO-TD3 actor loss；
- TD target；
- Step5.1 directional angle；
- Step5.2 HER；
- Step5.3 parallel exploration；
- replay；
- policy_freq；
- evaluation loop。

尤其不要让：

wp

进入 actor/critic training。

本阶段只做：

interpolator subsystem + independent tests。

==================================================
十八、文档
==========

创建：

docs/PD_MORL_INTERPOLATOR.md

必须记录：

1. Paper semantics
2. Official source behavior
3. Key preference construction
4. Key solution source
5. Initial L2 normalization
6. Online-update L1 normalization
7. SciPy RBFInterpolator exact configuration
8. Reference implementation
9. JAX representation
10. JAX forward
11. Golden Test
12. Numerical error
13. dtype difference
14. JIT boundary
15. Refitting boundary
16. Online update rule
17. Paper/source differences
18. SOURCE-FAITHFUL / FRAMEWORK-ADAPTATION / DEVIATION classification

==================================================
十九、PASS 条件
===============

只有以下全部满足才 PASS：

- key preference source 已核实；
- key solution source 已核实；
- 没有伪造官方 key solutions；
- initial L2 path 正确；
- online L1 path 正确；
- official SciPy RBF configuration 已核实；
- reference implementation 可运行；
- JAX forward 可运行；
- SciPy/JAX Golden Test 通过；
- knot/interior/batch 查询通过；
- numerical error 有明确报告；
- JAX forward 可 jit；
- JAX batch/vmap 正常；
- update/refit API 独立可测；
- SciPy 不进入 training JIT；
- wp 不做额外 normalization；
- 尚未接入 actor/critic training。

完成后输出：

STEP5.4 INTERPOLATOR RESULT

然后停止。
