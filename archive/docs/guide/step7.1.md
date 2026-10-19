> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在不要开始 Multi-GPU。

先增加一个很小的：

Step 7.1 — Production-Shape Single-GPU Walker Smoke

目标：

在不修改任何 PD-MORL 算法语义的前提下，
使用 production-size Actor/Critic 和 production batch，
确认当前完整 EvoRL-PD-MORL workflow
能够在单张 GPU 上稳定完成：

setup
→ rollout
→ replay/HER
→ learner update
→ control-plane
→ evaluation
→ checkpoint/restore。

这是 Multi-GPU 之前的最后一个 single-GPU production-shape gate。

==================================================
一、先审查当前 Step7 smoke
==========================

先查看：

docs/PD_MORL_EVORL_BASELINE.md

以及：

tests/test_pd_morl_integration.py

当前 real Walker smoke。

确认当前 smoke 中哪些 production 参数被 TEST OVERRIDE。

特别检查：

- network hidden size；
- batch_size；
- K / process_count；
- num_objectives；
- actor_loss_coeff；
- gamma；
- tau；
- policy_freq；
- HER；
- replay；
- matmul_precision。

先输出：

PRODUCTION-SHAPE SMOKE AUDIT

列出：

production value
current smoke value
是否允许继续 override。

==================================================
二、这次必须使用 production shape
=================================

Walker production-shape smoke 必须使用：

num_objectives = 2

process_count = 10
num_preference_subspaces = 10

B = K = 10

Actor：

input = 17 + 2
hidden = [400,400]
activation = ReLU
output = 6
tanh output

Critic Q1：

input = 17 + 2 + 6
hidden = [400,400]
output = 2

Critic Q2：

同样：

hidden = [400,400]
output = 2

Q1/Q2 独立参数。

---

严格禁止：
----------

不要为了 smoke 把：

400 → 64
400 → 32
400 → 128

或者任何其他较小网络。

这次 smoke 的主要目的
就是验证真实 production network shape。

==================================================
三、production learner shape
============================

必须使用：

batch_size = 256

不要缩成：

8
16
32
64

learner 中的：

s
a
w
r
Q1
Q2
wp
target

都必须经过真实 batch=256。

验证实际 shape。

至少记录：

state batch:
[256,17]

preference:
[256,2]

action:
[256,6]

critic input:
[256,25]

Q1/Q2:
[256,2]

wp:
[256,2]

TD target:
[256,2]

==================================================
四、必须保持 production algorithm config
========================================

不要修改：

gamma = 0.995

tau = 0.005

policy_freq = 10

actor_loss_coeff = 10

exploration_noise = 0.1

target_policy_noise = 0.2

noise_clip = 0.5

gradient_clip_norm = 100

num_relabel_preferences = 3

K = 10

HER semantics

preference grid/subspaces

whole-vector pessimistic target selection

Smooth-L1

directional angle

online L1 refit

initial L2 interpolator fit

evaluation semantics。

==================================================
五、GPU numerical policy
========================

必须：

jax.default_backend() == "gpu"

并明确：

matmul_precision = highest

不要依赖 shell 临时设置。

确认项目 production config
实际生效。

输出：

backend
device
dtype
matmul precision policy。

==================================================
六、允许缩短的内容
==================

为了让 smoke 测试时间合理，

允许使用：

TEST OVERRIDE

缩短：

- episode length；
- total training iterations；
- random-action warmup；
- learner-start threshold；
- key-update trigger threshold；
- full-evaluation trigger threshold；
- evaluation preference grid；
- evaluation repeats；
- checkpoint interval。

但是这些 override：

只能存在于 test/smoke config。

不能修改：

production Walker config。

==================================================
七、不允许缩短的内容
====================

以下不允许 override：

- Actor 400×400；
- Critic 400×400；
- batch_size=256；
- K=10；
- B=10；
- objective_dim=2；
- action_dim=6；
- obs_dim=17；
- actor_loss_coeff=10；
- policy_freq=10；
- gamma；
- tau；
- target noise；
- exploration noise；
- gradient clipping；
- HER Nw=3；
- scalarization；
- TD target；
- angle loss；
- matmul_precision=highest。

==================================================
八、测试运行长度
================

不需要跑正式 1M steps。

只需要足够长，
让以下行为至少实际发生一次：

1. replay 有足够样本；
2. critic update；
3. actor delayed update；
4. target update；
5. HER；
6. key-update control hook；
7. online interpolator refit；
8. full evaluation hook；
9. checkpoint save；
10. checkpoint restore。

建议：

不要人为规定固定 100/1000 steps。

根据 TEST OVERRIDE 的 trigger
选择最小但足够覆盖所有路径的运行长度。

==================================================
九、必须记录 learner diagnostics
================================

至少记录：

critic_total_loss

critic_smooth_l1_q1
critic_smooth_l1_q2

critic_angle_q1
critic_angle_q2

actor_scalarized_term
actor_angle_term
actor_total_loss

critic_grad_norm
actor_grad_norm

Q1 min/mean/max
Q2 min/mean/max

wp min/mean/max

TD target min/mean/max

replay size

HER active

episode_count min/max

key update count

interpolator refit count

HV

sparsity

Pareto count。

所有应该 finite 的量：

不得 NaN
不得 Inf。

==================================================
十、确认 optimizer update
=========================

必须确认：

critic parameters
在 critic update 后变化。

actor parameters：

只有：

total_it % policy_freq == 0

时变化。

非 delayed step：

actor params 不得变化。

target params：

按照当前 source-faithful
delayed target update semantics
正确变化。

==================================================
十一、确认真实 400×400 参数树
==============================

在测试里增加明确 assertion。

例如检查：

Actor：

W1 shape
W2 shape
W3 shape

Critic Q1：

W1 shape
W2 shape
W3 shape

Critic Q2：

W1 shape
W2 shape
W3 shape。

必须证明测试没有意外使用
smoke-small network。

输出：

PRODUCTION NETWORK SHAPE TABLE

例如：

actor layer1 ...
actor layer2 ...
actor out ...

q1 layer1 ...
q1 layer2 ...
q1 out ...

q2 layer1 ...
q2 layer2 ...
q2 out ...

==================================================
十二、确认真实 batch=256
========================

增加 runtime assertion：

learner sampled batch size == 256

不要只检查 config。

必须检查真正进入 loss 的 batch。

==================================================
十三、显存和 JIT
================

记录：

- device；
- first JIT compile 是否成功；
- steady-state update 是否成功；
- 是否 OOM；
- 是否发生 unexpected recompilation。

如果可以方便获得：

记录 GPU memory peak。

但：

显存采集不是 PASS 的硬阻塞项。

OOM 则必须 FAIL。

==================================================
十四、Preference Sensitivity
============================

训练后使用同一个 checkpoint：

w = [0,1]
w = [0.5,0.5]
w = [1,0]

在受控 observation / reset 下
做 deterministic action comparison。

确认：

至少有 measurable action difference。

这只验证 conditioning 未丢失。

不要要求论文级 objective ordering。

==================================================
十五、Checkpoint / Restore
==========================

在 online refit 发生后：

save checkpoint

然后 restore。

必须比较：

actor params

critic params

target params

optimizer states

total_it

episode_count

eval_cnt

eval_cnt_ep

raw key solutions

interpolator state

replay size

PRNG state

logical worker state。

特别验证：

restore 后：

I(w)

与 save 前一致。

不要重新加载 initial artifact
覆盖 online-refitted interpolator。

==================================================
十六、不要重新实现算法
======================

这次测试中禁止修改：

PDMORLAgent loss

TD target

HER

preference sampler

interpolator math

evaluation

Pareto

HV

sparsity

control triggers。

如果 production-shape smoke FAIL：

先定位原因。

不要修改算法以让 smoke 通过。

==================================================
十七、Regression
================

production-shape smoke 结束后：

重新运行当前 frozen suite。

至少包括：

Step5.4
Step5.5
Step6 Golden Tests
Step6.1
Step7 integration
scalar TD3 regression。

frozen tolerance：

不得修改。

==================================================
十八、创建独立测试
==================

建议创建：

tests/test_pd_morl_production_shape_smoke.py

或在现有 integration tests 中
建立明确独立测试：

test_pd_morl_walker_production_shape_single_gpu

必须标记：

GPU / slow / integration

避免普通 CPU unit test
每次都运行这个 400×400 smoke。

==================================================
十九、文档
==========

更新：

docs/PD_MORL_EVORL_BASELINE.md

增加：

Production-Shape Single-GPU Smoke

记录：

- GPU；
- network shapes；
- batch size；
- K/B；
- production config；
- TEST OVERRIDE items；
- runtime；
- JIT；
- finite diagnostics；
- checkpoint restore；
- preference sensitivity；
- regression result。

==================================================
二十、PASS Gate
===============

只有以下全部满足才 PASS：

1. real Brax Walker；
2. single GPU；
3. backend == gpu；
4. Actor = 400×400；
5. both Critics = 400×400；
6. batch_size = 256；
7. K = B = 10；
8. production PD-MORL math 未修改；
9. critic update 实际发生；
10. delayed actor update 实际发生；
11. target update 实际发生；
12. HER 实际发生；
13. key update 实际发生；
14. interpolator refit 实际发生；
15. evaluation 实际发生；
16. loss / Q / wp / gradients 全部 finite；
17. preference sensitivity measurable；
18. checkpoint/restore PASS；
19. no OOM；
20. frozen regression PASS；
21. Step6 tolerance unchanged；
22. production config 未被 TEST OVERRIDE 修改。

全部通过后输出：

STEP7.1 PRODUCTION-SHAPE SINGLE-GPU SMOKE PASS

如果失败：

输出：

STEP7.1 PRODUCTION-SHAPE SMOKE MISMATCH REPORT

说明：

- failure stage；
- actual shape；
- expected shape；
- error；
- suspected cause；
- 是否属于 algorithm / framework / resource issue。

然后停止。

不要开始 Multi-GPU。
