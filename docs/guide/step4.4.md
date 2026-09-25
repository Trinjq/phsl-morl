
现在开始实现最小版 preference-conditioned MO-TD3。

这一步仍然不是完整 PD-MORL。

目标是：
在 EvoRL 原生 TD3 骨架上完成 preference-conditioned Actor、
vector twin Critic、vector TD target 和 scalarized Actor objective。

不要从零重新实现 TD3。

==================================================
一、沿用已有数据层
==================

直接复用 Step4.1–4.3 已完成的数据：

obs        [...,17]
action     [...,6]
reward     [...,2]
preference [...,2]
done       [...]

preference 继续从：

SampleBatch.extras.policy_extras.preference

读取。

不要修改：

- vector reward；
- official preference grid；
- process_count；
- preference subspace；
- episode preference lifecycle；
- rollout；
- SampleBatch；
- replay buffer。

==================================================
二、Actor
=========

Actor 从：

a = π(s)

扩展为：

a = π(s,w)

输入：

concat(s,w)

Walker：

17 + 2 = 19

严格使用 Step3 已冻结的官方网络：

Linear(19,400)
ReLU
Linear(400,400)
ReLU
Linear(400,6)
tanh

* max_action

target actor 使用完全相同结构：

a' = π_target(s',w)

==================================================
三、Critic
==========

Critic 从：

Q(s,a) -> scalar

扩展为：

Q_i(s,w,a) -> R^2

输入顺序严格为：

concat(s,w,a)

Walker：

17 + 2 + 6 = 25

两个 critics 完全独立：

Q1:
25 -> 400 -> 400 -> 2

Q2:
25 -> 400 -> 400 -> 2

统一输出语义：

Q.shape == [B,2 critics,2 objectives]

或 EvoRL 中数学等价的 PyTree 表达。

不要压平 critic axis 和 objective axis。

target critics 同样输出 vector Q。

==================================================
四、网络初始化
==============

按 Step3 冻结的官方行为：

hidden activation:
ReLU

weights:
Xavier-normal

bias:
0

不要直接沿用 EvoRL TD3 默认 256x256 / LeCun 初始化。

只为新的 MO-TD3 配置使用官方参数，
不要修改原 scalar TD3 默认行为。

==================================================
五、Scalarization
=================

实现独立纯函数：

scalarize(Q,w)

数学定义：

Q_scalar = sum(w * Q_vec, axis=-1)

例如：

Q_vec: [B,2]
w:     [B,2]

输出：

[B]

如果 twin critics：

Q: [B,2 critics,2 objectives]

输出：

[B,2 critics]

必须使用原始 preference w。

==================================================
六、Target action
=================

使用：

a' = π_target(s',w)

随后保留 TD3 原有：

target policy smoothing noise
+
noise clip
+
action clip

不要修改 Step3 已冻结的 noise 参数和顺序。

==================================================
七、Twin vector target selection
================================

target critics 得到：

Q'_1(s',w,a') ∈ R^2
Q'_2(s',w,a') ∈ R^2

分别计算：

z1 = w^T Q'_1
z2 = w^T Q'_2

然后：

idx = argmin([z1,z2])

根据 idx 选择对应 critic 的完整 vector：

Q'_selected ∈ R^2

禁止：

elementwise minimum

禁止：

minimum(Q1,Q2)

禁止生成：

[min(Q1_0,Q2_0), min(Q1_1,Q2_1)]

必须完整选择 Q1 或 Q2 中的一条 vector。

==================================================
八、Vector Bellman target
=========================

使用已有 combined done：

y_vec =
r_vec
+
gamma * (1 - done)[...,None] * Q'_selected

其中：

r_vec.shape == [B,2]
y_vec.shape == [B,2]

必须复用 Step3 冻结的 combined done 语义：

physical termination 和 time-limit truncation
都不 bootstrap。

==================================================
九、Critic loss
===============

本阶段暂时不加入 directional angle。

两个 critics 都回归同一个 y_vec。

使用官方 PD-MORL 的 Smooth-L1 regression：

L_Q =
SmoothL1(Q1,y_vec)
+
SmoothL1(Q2,y_vec)

保持默认 mean reduction。

本阶段不要加入：

g(w_p,Q1)
g(w_p,Q2)

因为 interpolator / w_p / angle 尚未实现。

保留官方 critic gradient clipping：

global norm = 100

==================================================
十、Actor loss
==============

Actor 只使用 critic 1：

a = π(s,w)

Q1_vec = Q1(s,w,a)

Q1_scalar = w^T Q1_vec

当前最小 Actor loss：

L_actor =
-mean(Q1_scalar)

暂时不要加入 directional angle。

保留：

delayed actor update

以及官方 actor gradient clipping：

global norm = 100

==================================================
十一、保留 TD3 骨架
===================

必须继续复用 EvoRL TD3 已有：

- twin critics；
- target actor；
- target critics；
- exploration noise；
- target smoothing；
- delayed actor update；
- soft target update；
- optimizer/workflow 骨架；
- JAX/JIT execution。

不要从零创建另一套 TD3 training loop。

==================================================
十二、暂时不要实现
==================

不要实现：

- HER；
- sample-time HER；
- add-time HER；
- interpolator；
- projected preference w_p；
- cosine similarity；
- directional angle；
- Pareto/HV/sparsity；
- PSL-MORL；
- Hypernetwork。

已有 preference sampler / subspace 继续使用，
但不要修改。

==================================================
十三、测试
==========

至少包含：

1. Actor:
   input s:[B,17], w:[B,2]
   output a:[B,6]
2. Critic:
   input s,w,a
   output twin vector Q:[B,2,2]
3. scalarize:
   Q:[B,2] + w:[B,2] -> [B]
4. twin scalarization:
   Q:[B,2,2] + w:[B,2] -> [B,2]
5. whole-vector target selection：
   构造：

   Q1=[1,10]
   Q2=[5,2]

   使用能明确选中某一 critic 的 w。

   输出必须严格等于完整 Q1 或完整 Q2。

   禁止等于：
   [1,2]
6. vector Bellman target:
   output shape == [B,2]
7. combined done:
   done row 不 bootstrap。
8. twin critic:
   Q1/Q2 参数独立。
9. target smoothing:
   shape/action bounds 正确。
10. delayed actor update:
    critic 每次更新；
    actor/targets 按 policy_freq 更新。
11. JIT:
    actor、critic、target、loss、update 可正常编译。
12. batched execution:
    优先使用 EvoRL/Brax 原生 batch；
    不强制为了测试增加 vmap。
13. GPU:
    backend == gpu
    并记录 CudaDevice。
14. real Walker smoke test：
    不只使用 toy environment。

    使用现有 Brax Walker2d：
    vector reward

    + official preference
    + replay sample
    + 一次 critic update
    + 一次允许触发的 actor update

    确认 loss finite、参数发生预期变化。

==================================================
十四、兼容要求
==============

原始 scalar TD3 必须继续可运行。

不要直接把 EvoRL 原 TD3 critic 全局改成 vector output。

优先新增 MO-TD3 专用 agent/network/config，
同时复用原生 TD3 workflow/helper。

==================================================
十五、输出文档
==============

创建：

docs/MO_TD3_BASELINE.md

记录：

- Actor/Critic shape；
- network architecture；
- vector Q；
- scalarization；
- twin target selection；
- vector Bellman target；
- done mask；
- critic loss；
- actor loss；
- delayed update；
- 修改文件；
- 与原 TD3 的复用关系；
- 测试结果；
- GPU/JIT 验证；
- 当前与完整 PD-MORL 的差异。

==================================================
十六、完成条件
==============

只有以下全部满足才算 PASS：

- Actor(s,w) 正确；
- twin vector Critic 正确；
- scalarization 正确；
- whole-vector pessimistic target selection 正确；
- vector Bellman target 正确；
- combined done 正确；
- critic Smooth-L1 正确；
- actor 使用 critic1 的 w^TQ；
- target smoothing 保留；
- delayed update 保留；
- soft target update 保留；
- gradient clipping 正确；
- JIT/GPU 正常；
- 原 scalar TD3 未被破坏；
- 没有加入 HER/interpolator/angle。

完成后停止。
