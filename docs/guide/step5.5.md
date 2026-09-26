
现在进入 Step 5.5：

PD-MORL Preference-Q Alignment Loss Integration

当前已经完成并冻结：

- vector reward；
- preference lifecycle；
- preference subspace / parallel exploration；
- source-faithful add-time HER；
- MO-TD3 vector Actor/Critic；
- whole-vector pessimistic TD target；
- scalarization w^T Q；
- multi-dimensional interpolator I(w)；
- JAX MORL math primitives。

本阶段目标：

正式把

wp = I(w)

和

directional angle g(wp,Q)

接入 PD-MORL 的 critic loss 与 actor loss。

严格以：

1. PD-MORL 官方 PyTorch source；
2. PD-MORL ICLR 2023 paper；
3. official settings/config；

为依据。

源码行为优先于论文中存在歧义的公式。

==================================================
一、先做 SOURCE AUDIT
=====================

先审查：

- lib/common_ptan/agent.py
- lib/utilities/settings.py
- train_Walker2d_MO_TD3_HER.py
- PD-MORL paper Appendix B.1.5 / Algorithm 3
- 当前 evorl/utils/morl_math.py
- 当前 interpolator implementation
- 当前 evorl/algorithms/mo_td3.py

先明确回答：

1. critic angle 使用的是 w 还是 wp；
2. actor angle 使用的是 w 还是 wp；
3. cosine_similarity 的实际输入；
4. cosine clamp 区间；
5. acos 输出最终是 radians 还是 degrees；
6. critic angle term 是否有 coefficient；
7. actor angle coefficient 来自哪里；
8. Walker 的 actor_loss_coeff 精确值；
9. critic regression 使用 MSE 还是 Smooth-L1；
10. Smooth-L1 reduction 是什么；
11. actor scalar objective 使用 w 还是 wp；
12. TD target critic selection 使用 w 还是 wp；
13. angle 是否参与 TD target；
14. actor 使用 Q1 还是 twin critics；
15. actor/target update timing 是否保持已有 policy_freq。

先输出 SOURCE AUDIT。

不要立即修改代码。

==================================================
二、检查 MORL math primitive
============================

不要假设当前 directional_angle 已经正确。

实际检查当前代码。

官方源码等价行为必须是：

cos =
cosine_similarity(wp, Q)

cos =
clip(cos, 0.0, 0.9999)

angle_rad =
acos(cos)

angle =
rad2deg(angle_rad)

即：

g(wp,Q)
=======

degrees(
    acos(
        clip(
            cosine_similarity(wp,Q),
            0,
            0.9999
        )
    )
)

特别注意：

不是：

clip [-1,1]

也不是：

radians。

如果当前 morl_math.py
仍然是旧版 radians / [-1,1]：

先修正 primitive，
再进行 loss integration。

safe norm 必须继续使用
已经验证过的 zero-gradient-safe JAX 实现。

==================================================
三、wp 的来源
=============

对于 replay batch 中原始 preference：

w [B,L]

调用 Step5.4 已验证的 pure-JAX forward：

wp = interpolate(interpolator_state, w)

得到：

wp [B,L]

不要：

- 再 L1 normalize；
- 再 L2 normalize；
- clip wp；
- softmax wp；
- projection to simplex。

直接使用 interpolator output。

==================================================
四、w 与 wp 的职责必须严格区分
==============================

原始 w：

用于：

1. Actor / Critic 网络条件输入；
2. target critic scalarization；
3. twin target critic selection；
4. actor scalarized objective w^T Q。

wp：

只用于：

directional-angle term。

严格禁止：

- 用 wp 替换 target scalarization 中的 w；
- 用 wp 替换 actor 主目标中的 w；
- 把 wp 输入 replay；
- 修改 transition preference；
- 使用 wp 改变 TD target。

==================================================
五、Directional Angle
=====================

严格实现：

g(wp,Q)
=======

rad2deg(
    acos(
        clip(
            cosine_similarity(wp,Q),
            0.0,
            0.9999
        )
    )
)

shape：

wp [B,L]
Q  [B,L]

输出：

angle [B]

然后 loss 中使用：

mean(angle)

不要沿 objective dimension
分别计算 angle。

一个 Q vector
对应一个 directional angle。

==================================================
六、必须锁定源码边界值行为
==========================

单元测试必须使用源码期望值。

1. Same direction

例如：

wp = [1,0]
Q  = [2,0]

cos=1
→ clip=0.9999

所以：

angle ≈ 0.810291 degrees

不是 0。

---

2. Orthogonal

wp = [1,0]
Q  = [0,2]

cos=0

angle = 90 degrees

---

3. Opposite

wp = [1,0]
Q  = [-2,0]

cos=-1
→ lower clamp to 0

angle = 90 degrees

不是 180。

---

4. Positive scale invariance

wp=[1,2]
Q1=[2,4]
Q2=[20,40]

angle(Q1)
≈
angle(Q2)

---

5. Different magnitudes

确认 angle 只依赖方向，
不会因为 Q 正比例放大而改变。

==================================================
七、Critic Loss
===============

保持 Step4.4 已有 vector Bellman target：

y [B,L]

不要改变：

- target actor；
- target smoothing noise；
- whole-vector pessimistic critic selection；
- Bellman target；
- original w scalarization。

得到在线 critics：

Q1 [B,L]
Q2 [B,L]

计算：

angle1 =
g(wp,Q1)

angle2 =
g(wp,Q2)

官方源码语义：

critic_loss =
mean(angle1)
+
smooth_l1_loss(Q1,y)
+
mean(angle2)
+
smooth_l1_loss(Q2,y)

必须核实并复现
PyTorch F.smooth_l1_loss 默认 reduction。

不要：

- 给 critic angle 乘 actor_loss_coeff；
- 新增 critic_angle_coeff；
- 平均 Q1/Q2 再算 angle；
- 对 objective 分别算 angle。

Critic angle coefficient：

隐式为 1。

==================================================
八、Actor Loss
==============

actor 使用：

a =
actor(s,w)

然后：

Q =
Q1(s,w,a)

只使用 critic 1。

主目标：

scalar_q =
w^T Q

angle：

actor_angle =
g(wp,Q)

JAX optimizer 是 minimization，
因此必须实现：

actor_loss =
-mean(scalar_q)
+
actor_loss_coeff * mean(actor_angle)

不要改正负号。

不要写成：

+mean(w^TQ)

也不要写成：

-angle

Walker：

actor_loss_coeff

必须从 official settings/config 读取。

不要在 loss function 内硬编码 10。

如果 Walker official config 是 10，
测试确认配置读取结果 == 10。

==================================================
九、论文 / 源码差异
===================

文档必须明确记录：

1. 论文 directional-angle 公式存在 w / wp
   记号不一致或印刷歧义；
2. 官方源码实际使用：
   wp 与 Q 计算 cosine；
3. 论文公式写 acos，
   官方源码额外：

   - clamp [0,0.9999]
   - rad2deg；
4. critic paper expression 与源码 regression
   可能存在 MSE / Smooth-L1 表述差异；
5. 官方实际 critic：
   Smooth-L1
   +
   unweighted angle means。

最终 implementation：

以官方源码逐操作行为为准。

==================================================
十、Interpolator integration
============================

Step5.4 当前：

host-side SciPy fit/refit
+
fixed-shape pure-JAX forward

本阶段只把：

PDMORLInterpolatorState

接到 learner 所需位置。

training JIT 内只能调用：

pure-JAX interpolate(state,w)

不得：

- 调 SciPy；
- refit interpolator；
- host callback；
- 创建 Python RBFInterpolator。

在线 interpolator refit
仍然留给后续 evaluation/control boundary integration。

本阶段只接当前固定 interpolator state。

==================================================
十一、Gradient 路径
===================

Critic update：

gradient 必须经过：

Q1/Q2
→ Smooth-L1
+
Q1/Q2
→ angle

并只更新 critic params。

Actor update：

gradient 必须经过：

actor
→ Q1
→ scalarized Q

以及：

actor
→ Q1
→ directional angle

最终只更新 actor params。

critic params 在 actor gradient step 中
作为固定参数输入。

不要对 critic optimizer
应用 actor-loss gradient。

==================================================
十二、Gradient Clipping
=======================

保持已经冻结的官方行为：

critic gradients：
global L2 norm clip = 100

actor gradients：
global L2 norm clip = 100

不要因为加入 angle
新增另一层 clip。

不要：

- clip angle；
- clip Q；
- clip wp；
- 单独 clip angle gradient。

==================================================
十三、Numerical Tests
=====================

必须增加独立数值测试。

至少覆盖：

A. directional angle

- same direction
- orthogonal
- opposite
- positive rescaling
- batch
- float32
- finite

B. critic loss

人工构造：

wp
Q1
Q2
y

手工计算：

smooth_l1_1
smooth_l1_2
angle1
angle2

验证：

loss ==
smooth_l1_1

+ smooth_l1_2
+ mean(angle1)
+ mean(angle2)

并明确验证：

critic angle 没有 × actor_loss_coeff。

C. actor loss

人工构造 Q 和 w。

验证：

loss ==
-mean(w^TQ)

+ alpha * mean(angle)

并验证：

改变 alpha
只影响 actor angle term。

==================================================
十四、Gradient Tests
====================

测试：

critic gradient finite
actor gradient finite

并检查：

- no NaN；
- no Inf；
- gradient PyTree finite。

至少包含：

- normal non-collinear case；
- same-direction upper-clamp case；
- opposite-direction lower-clamp case；
- small-norm case。

注意：

因为 clamp 会产生零梯度区域，
不要要求所有测试情况下 angle gradient 非零。

只要求：

数值符合源码且 gradient finite。

==================================================
十五、JIT / VMAP
================

验证：

directional_angle:

- eager
- jit
- vmap / batch

critic loss:

- jit

actor loss:

- jit

完整一次：

sample replay batch
→ wp=I(w)
→ critic update
→ delayed actor update

可 JIT 执行。

不要为了满足测试
机械给已经 batched 的网络再套 vmap。

==================================================
十六、Regression
================

必须重新运行：

- Step5.1 MORL math；
- Step5.4 interpolator Golden Test；
- Step4.4 MO-TD3 tests；
- Step5.2 HER；
- Step5.3 parallel exploration；
- scalar TD3 regression。

不得破坏：

- vector TD target；
- HER；
- preference lifecycle；
- collection/update ratio；
- delayed policy update；
- scalar TD3。

==================================================
十七、暂时不要做 interpolator online refit integration
======================================================

Step5.5 只完成：

fixed interpolator state
+
wp forward
+
alignment loss

不要同时实现：

- key preference evaluation scheduling；
- online key-solution replacement；
- online SciPy refit；
- Pareto metrics；
- HV；
- sparsity；
- multi-GPU placement。

这些留给后续步骤。

==================================================
十八、文档
==========

创建：

docs/PD_MORL_ALIGNMENT_LOSS.md

必须记录：

1. Paper formula
2. Official PyTorch behavior
3. w vs wp roles
4. directional-angle exact definition
5. clamp [0,0.9999]
6. degree conversion
7. critic loss
8. actor loss
9. Walker actor_loss_coeff source
10. Smooth-L1 reduction
11. gradient paths
12. numerical boundary behavior
13. JIT strategy
14. regression results
15. SOURCE-FAITHFUL / FRAMEWORK-ADAPTATION / DEVIATION

==================================================
十九、PASS 条件
===============

只有以下全部满足才 PASS：

- wp 来自 Step5.4 pure-JAX interpolator；
- wp 没有额外 normalization；
- directional angle 使用 wp；
- cosine clamp == [0,0.9999]；
- angle 单位 == degrees；
- same-direction ≈ 0.810291°；
- orthogonal == 90°；
- opposite == 90°；
- critic angle coefficient == 1；
- critic 使用两个 Smooth-L1 + 两个 angle mean；
- actor 使用 Q1；
- actor loss 正号/负号严格正确；
- actor_loss_coeff 来自 config；
- original w 继续用于 scalarization；
- target 完全未改；
- critic gradient finite；
- actor gradient finite；
- JIT 正常；
- regression 全通过；
- 尚未接入 online interpolator refit。

完成后输出：

STEP5.5 ALIGNMENT LOSS RESULT

然后停止。
