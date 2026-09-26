
现在进入 Step 6：

PD-MORL PyTorch ↔ JAX/EvoRL Golden Tests

当前不要新增任何算法。

本阶段唯一目标：

在完全受控的固定输入、固定参数条件下，
逐层比较 PD-MORL 官方 PyTorch 实现
与当前 JAX/EvoRL 实现的关键数值。

如果核心数值不能对齐：

立即停止。

不要进入 PSL-MORL，
不要通过调大 tolerance 掩盖差异。

==================================================
一、测试边界
============

本阶段测试：

- Actor forward；
- Critic forward；
- vector Q；
- scalarization；
- interpolator output wp；
- cosine similarity；
- directional angle；
- target critic selection；
- vector Bellman target；
- Smooth-L1 critic regression；
- critic total loss；
- actor scalarized objective；
- actor angle objective；
- actor total loss；
- raw gradients。

不测试：

- environment physics；
- MuJoCo ↔ Brax trajectory matching；
- Adam optimizer state；
- replay stochastic sampling；
- full training convergence；
- Pareto metrics；
- online interpolator refit；
- multi-GPU。

使用固定 synthetic batch，
从而只比较算法数值，
不把环境差异带入 Golden Test。

==================================================
二、先审查两端实现
==================

Reference：

PD-MORL official PyTorch source：

- lib/models/networks.py
- lib/common_ptan/agent.py
- train_Walker2d_MO_TD3_HER.py
- lib/utilities/settings.py

JAX：

- current MO-TD3 Actor/Critic；
- morl_math.py；
- interpolator implementation；
- alignment loss；
- TD target helpers。

先建立一张映射表：

PyTorch tensor / parameter
↔
JAX tensor / parameter

特别检查：

- Linear weight orientation；
- bias orientation；
- Actor layer order；
- Q1/Q2 separation；
- concatenate order；
- tanh action scaling；
- objective dimension order。

先输出：

GOLDEN TEST MAPPING AUDIT

然后再写测试。

==================================================
三、固定 Golden Input
=====================

建立 deterministic Golden fixture：

tests/golden/pd_morl_walker_input.npz

至少保存：

s
a
r_vec
s_next
done
w

target_noise

actor weights
actor biases

critic Q1 weights
critic Q1 biases

critic Q2 weights
critic Q2 biases

target actor weights
target critic Q1 weights
target critic Q2 weights

interpolator state / reference inputs

建议：

batch size 使用小值，例如：

B = 4 或 8

但：

obs_dim = 17
action_dim = 6
objective_dim = 2

保持 Walker 实际维度。

所有数据使用固定 seed。

==================================================
四、网络权重必须真正一致
========================

不要：

PyTorch 自己 random init
+
JAX 自己 random init

然后比较输出。

必须采用：

一套 canonical weights

然后分别加载到：

PyTorch
JAX

注意：

PyTorch Linear weight shape：

[out_dim, in_dim]

JAX Dense 常见 kernel：

[in_dim, out_dim]

因此必须显式 transpose。

建立：

torch_to_jax_params(...)
jax_to_torch_params(...)

至少其中一个明确可验证。

转换后增加参数级测试：

max abs error == 0

或在 dtype cast 后符合精确 tolerance。

==================================================
五、Golden Reference Export
===========================

编写：

tests/golden/export_pd_morl_pytorch_reference.py

使用官方 PyTorch 逻辑。

输入：

同一份 canonical fixture。

输出：

tests/golden/pd_morl_pytorch_reference.npz

至少保存：

actor_output

target_actor_output

critic_q1
critic_q2

target_q1
target_q2

scalar_q1
scalar_q2

wp

cosine_q1
cosine_q2

angle_q1
angle_q2

selected_target_critic_index
selected_target_q_vector

td_target_y

smooth_l1_q1
smooth_l1_q2

critic_angle_q1
critic_angle_q2

critic_total_loss

actor_scalarized_q
actor_angle
actor_total_loss

如果可行：

actor_raw_gradients
critic_q1_raw_gradients
critic_q2_raw_gradients

==================================================
六、wp / interpolator
=====================

Golden Test 中 interpolator 必须固定。

建议使用 Step5.4 已通过 Golden Test 的
deterministic interpolator fixture。

PyTorch reference：

使用 SciPy interpolator 输出 wp。

JAX：

使用 Step5.4 fixed-shape JAX forward。

先独立确认：

wp_ref ≈ wp_jax

再继续 angle 和 loss comparison。

不要因为真实 Walker key artifact 尚缺失
而阻塞数学 Golden Test。

这个 fixture 必须明确标记：

TEST FIXTURE

不是官方 Walker Pareto solution。

==================================================
七、Scalarization
=================

必须测试：

w^T Q1
w^T Q2

使用：

原始 w

绝不能使用 wp。

增加：

test_scalarization

分别比较：

PyTorch
JAX
hand-calculated reference。

==================================================
八、Directional Angle
=====================

严格比较官方源码：

cos =
cosine_similarity(wp,Q)

cos =
clip(cos,0,0.9999)

angle =
rad2deg(acos(cos))

增加：

test_cosine_similarity
test_directional_angle

覆盖：

normal case
same direction
orthogonal
opposite
different magnitude

确认：

same direction ≈ 0.810291°
orthogonal = 90°
opposite = 90°

==================================================
九、Vector-Q Network Forward
============================

增加：

test_actor_forward
test_vector_q

给完全相同：

s
w
a
weights

比较：

Actor output

Q1
Q2

shape：

actor [B,6]

Q1 [B,2]
Q2 [B,2]

必须逐元素比较。

==================================================
十、Target Critic Selection
===========================

使用：

target Q1 [B,L]
target Q2 [B,L]

计算：

score1 = w^T target_Q1
score2 = w^T target_Q2

按较小 scalar score
选择完整 Q vector。

增加：

test_target_critic_selection

保存：

selected_index

selected_q_vector

明确验证结果不是：

elementwise min(Q1,Q2)。

==================================================
十一、TD Target
===============

严格比较：

y =
r_vec
+
gamma * (1-done) * selected_target_q

增加：

test_td_target

PyTorch 与 JAX
逐元素比较：

[B,L]

不要在这里加入 angle。

==================================================
十二、Critic Loss
=================

Reference 必须逐项导出：

smooth_l1_q1
smooth_l1_q2

mean_angle_q1
mean_angle_q2

以及：

critic_total_loss

严格比较：

critic_total =
smooth_l1_q1
+
smooth_l1_q2
+
mean_angle_q1
+
mean_angle_q2

critic angle coefficient：

1

不要乘：

actor_loss_coeff。

增加：

test_critic_loss

==================================================
十三、Actor Objective
=====================

Actor 使用：

Q1(
    s,
    w,
    actor(s,w)
)

比较：

scalarized =
mean(w^T Q1)

angle =
mean(g(wp,Q1))

actor loss：

-scalarized
+
actor_loss_coeff * angle

Walker：

actor_loss_coeff = 10

增加：

test_actor_objective

分别比较：

scalarized term
angle term
total actor loss。

==================================================
十四、Raw Gradient Golden Test
==============================

这是 Step6 中非常重要的测试。

不要先比较 Adam update。

比较：

d critic_loss / d critic_params

d actor_loss / d actor_params

要求：

PyTorch：

optimizer.zero_grad()
loss.backward()

直接导出：

parameter.grad

不要调用 optimizer.step()。

JAX：

jax.grad

得到相同参数树 raw gradient。

注意：

PyTorch / JAX Linear kernel orientation 不同。

比较 gradient 时：

也必须执行相同 transpose mapping。

至少比较：

每一层：

weight gradient
bias gradient

报告：

max absolute error
mean absolute error
relative error

如果某个参数 gradient 全零，
relative error 不作为主要指标。

==================================================
十五、dtype 策略
================

优先建立两组 Golden Test：

A. float64 diagnostic

用于尽量减少 framework floating-point 差异，
方便定位数学不一致。

B. float32 production

对应实际 JAX GPU 训练。

不要把 float64 tolerance
和 float32 tolerance 混用。

==================================================
十六、Tolerance
===============

不要一开始统一写：

rtol=1e-3

每个测试应根据操作性质设置。

建议先测实际误差，
再冻结 tolerance。

例如：

纯 scalarization：
应非常严格。

forward MLP：
允许少量浮点累积误差。

acos / angle：
因为非线性，
可稍宽于 dot product。

gradient：
可比 forward 稍宽。

文档中必须记录：

metric
dtype
atol
rtol
measured max abs error
measured max rel error

禁止为了 PASS
不断扩大 tolerance。

==================================================
十七、测试结果表
================

生成详细结果。

格式至少：

| Metric | PyTorch | JAX | Max Abs Error | Max Rel Error | atol | rtol | Status |
| ------ | ------: | --: | ------------: | ------------: | ---: | ---: | ------ |

如果是向量或 tensor：

PyTorch/JAX 栏
可以写：

shape / summary

不要打印整个大 tensor。

但：

max abs
mean abs
max rel

必须记录。

==================================================
十八、必须包含的 tests
======================

至少：

test_actor_forward

test_vector_q

test_scalarization

test_interpolator_wp

test_cosine_similarity

test_directional_angle

test_target_critic_selection

test_td_target

test_critic_loss

test_actor_objective

test_critic_raw_gradient

test_actor_raw_gradient

test_jit_compile

test_vmap_batch

==================================================
十九、JIT / VMAP 的定位
=======================

PyTorch reference
不需要比较 JIT。

JAX 额外验证：

eager output
============

jit output

single-sample helper
====================

vmap output

但是：

不要对已经 batched 的 Actor/Critic
机械再套 vmap。

vmap 只用于本来定义为单样本的 helper
或者明确需要验证 batch equivalence 的函数。

==================================================
二十、不要比较 stochastic training behavior
===========================================

Golden Test 中：

所有随机数都提前固定并保存。

例如：

target smoothing noise
exploration noise

不要让：

PyTorch torch.randn
和
JAX random.normal

各自随机采样后再比较。

它们 PRNG 算法不同。

正确方式：

生成一次固定 noise array，
两边共同加载。

==================================================
二十一、Golden 数据版本
=======================

在 .npz metadata 或旁边的 json 中记录：

PD-MORL commit:
35aa1bc...

JAX/EvoRL commit

Python version

PyTorch version

SciPy version

JAX version

dtype

Walker dimensions

actor_loss_coeff

gamma

policy_freq

这样未来代码改动后
Golden fixture 仍然可追溯。

==================================================
二十二、不要在 Step6 修改算法
=============================

如果发现不一致：

先定位。

分类为：

A. PyTorch/JAX tensor-layout mapping error

B. numerical precision difference

C. framework adaptation

D. actual algorithm-semantic mismatch

如果属于 D：

STOP。

不要直接修改算法后继续跑全部测试。

先输出：

GOLDEN MISMATCH REPORT

说明：

- metric；
- PyTorch value；
- JAX value；
- source behavior；
- suspected cause。

等待确认。

==================================================
二十三、文档
============

创建：

docs/PD_MORL_GOLDEN_TESTS.md

记录：

1. Purpose
2. Reference commit
3. Canonical fixture
4. Parameter mapping
5. Network forward comparison
6. Interpolator comparison
7. Scalarization
8. Angle
9. TD target
10. Critic loss
11. Actor loss
12. Raw gradients
13. dtype
14. tolerances
15. measured numerical errors
16. JIT/vmap checks
17. mismatches
18. PASS/FAIL conclusion

==================================================
二十四、PASS Gate
=================

只有以下核心项全部 PASS：

- Actor forward
- Critic Q1/Q2
- scalarization
- wp
- cosine
- directional angle
- target critic selection
- vector TD target
- critic loss
- actor loss
- critic raw gradients
- actor raw gradients

才能宣布：

STEP6 GOLDEN TEST PASS

JIT/vmap 也必须通过，
但它们属于 JAX framework verification，
不是 PyTorch 数值 reference comparison。

如果任何核心项 FAIL：

停止。

不要：

- 进入 PSL-MORL；
- 继续加 Hypernetwork；
- 调大 tolerance 掩盖；
- 比较最终 reward curve 代替数值定位。

完成后停止。
