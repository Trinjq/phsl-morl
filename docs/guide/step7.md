
现在进入 Step 7：

End-to-End EvoRL PD-MORL Baseline Integration

当前已经完成并冻结：

- MO-TD3 vector Actor/Critic；
- vector reward / vector Q；
- preference lifecycle；
- preference subspace；
- parallel logical workers；
- source-faithful HER；
- whole-vector pessimistic TD target；
- scalarization w^T Q；
- interpolator I(w)；
- preference-Q directional alignment；
- actor / critic PD-MORL losses；
- Step6 PyTorch ↔ JAX Golden Tests；
- Step6.1 Control / Evaluation Plane：
  - initial key artifact；
  - key evaluation；
  - key replacement；
  - online interpolator refit；
  - MORL evaluator；
  - Pareto；
  - HV；
  - sparsity。

本阶段：

不要新增 PD-MORL 算法。

唯一目标：

把已经单独验证并冻结的模块
整合为一个清晰、可运行、可训练、可评估的
single-GPU EvoRL-PD-MORL baseline。

==================================================
一、先做 Integration Audit
==========================

不要立即重构代码。

先检查当前 EvoRL repository。

输出：

STEP7 INTEGRATION AUDIT

逐项列出：

1. 当前 MO-TD3 Agent；
2. 当前 MO-TD3 Workflow；
3. preference wrapper；
4. preference subspace；
5. HER；
6. interpolator state；
7. PD-MORL alignment loss；
8. replay；
9. episode_count；
10. KeyInterpolatorUpdateController；
11. PDMORLEvaluator；
12. Pareto/HV/sparsity；
13. config；
14. logging；
15. checkpoint / restore。

对每项标记：

ALREADY CONNECTED
IMPLEMENTED BUT NOT CONNECTED
NEEDS INTEGRATION
DO NOT CHANGE

如果现有结构已经正确：

不要重写。

==================================================
二、架构原则
============

最大程度复用 EvoRL 原生 TD3 / OffPolicyWorkflow。

优先复用：

- rollout；
- replay；
- optimizer；
- target update；
- WorkflowState；
- checkpoint；
- metrics；
- logging；
- evaluator infrastructure；
- JAX/JIT；
- Brax vectorized env。

只在真正不同的地方扩展 MORL。

禁止：

- 复制整个 TD3 实现再维护一套；
- 复制 replay；
- 复制 optimizer；
- 复制 target-update logic；
- 重写已经通过 Golden Test 的 loss；
- 重新实现 Step6.1 evaluator。

==================================================
三、建议最终结构
================

建立清晰的：

PDMORLAgent

负责：

- Actor/Critic forward；
- vector TD target；
- wp = I(w)；
- critic loss；
- actor loss；
- learner update。

PDMORLWorkflow

负责：

- environment interaction；
- preference lifecycle；
- logical workers；
- replay；
- HER；
- learner scheduling；
- episode counters；
- control hooks；
- evaluation hooks；
- checkpoint state。

Control/Evaluation component：

复用 Step6.1 已验证的：

- key update controller；
- PDMORLEvaluator；
- Pareto/HV/sparsity。

不要把：

Pareto
HV
SciPy refit

塞进 PDMORLAgent。

==================================================
四、完整 Training Dataflow
==========================

端到端必须形成：

logical worker lanes
        ↓
episode preference w
        ↓
actor(s,w)
        ↓
exploration action
        ↓
Brax environment
        ↓
vector reward r
        ↓
transition:
(s,a,r_vec,s_next,w,done)
        ↓
source-faithful HER
        ↓
logical global replay
        ↓
sample minibatch
        ↓
target actor(s_next,w)
        ↓
target Q1,Q2 vectors
        ↓
original w scalarization
        ↓
select whole pessimistic Q vector
        ↓
vector Bellman target y
        ↓
wp = I(w)
        ↓
critic:
Smooth-L1 + directional angle
        ↓
critic update
        ↓
policy_freq gate
        ↓
actor:

- w^T Q1

+ actor_loss_coeff * angle
  ↓
  actor update
  ↓
  target updates

所有数学公式：

直接调用已经冻结的实现。

不得重新写第二套公式。

==================================================
五、Control Plane Dataflow
==========================

训练过程中同时维护：

episode_count[K]

eval_cnt_ep

eval_cnt

raw key_solutions

fixed-shape interpolator state

当 Step6.1 key trigger 成立：

episode counters
        ↓
3 key preference evaluation
        ↓
repeat mean candidate returns
        ↓
strict scalarized replacement
        ↓
online L1 normalization
        ↓
host SciPy linear RBF refit
        ↓
replace fixed-shape interpolator arrays
        ↓
继续 training JIT

严格复用 Step6.1。

不要重新设计 trigger。

==================================================
六、Evaluation Dataflow
=======================

当 Step6.1 full evaluation trigger 成立：

当前 actor checkpoint/state
        ↓
PDMORLEvaluator
        ↓
201 preference train grid
        ↓
3 repeats
        ↓
vector returns
        ↓
Pareto
        ↓
HV
        ↓
sparsity
        ↓
workflow metrics / logger

最终/offline：

1001 preferences
+
对应 repeat 语义。

不要重新实现 evaluator。

只完成 workflow integration。

==================================================
七、Config
==========

配置必须清楚区分：

A. PD-MORL algorithm/config

num_objectives

num_preference_subspaces
或 process_count

num_relabel_preferences

actor_loss_coeff

preference_grid_step

random_action_warmup

learner_start_threshold

replay_capacity

policy_freq

gamma

tau

exploration_noise

target_policy_noise

noise_clip

gradient_clip_norm

B. Interpolator/control

interpolator_artifact

key_update_enabled

eval_freq

eval_episodes

training_eval_preference_step

offline_eval_preference_step

C. Framework numerical policy

matmul_precision: highest

---

不要普通暴露：
--------------

interpolator_kernel

initial_normalization

online_normalization

wp_post_normalization

critic_angle_coeff

这些对于 source-faithful baseline
不是自由超参数。

固定为：

kernel = linear

initial normalization = L2

online normalization = L1

wp post normalization = none

critic angle coefficient = 1

如果未来做 ablation：

单独建立 deviation/ablation config。

==================================================
八、Actor Loss Coefficient
==========================

配置项名称明确使用：

actor_loss_coeff

Walker baseline：

10

不要使用模糊的：

angle_coefficient

因为：

critic angle coefficient
源码固定为 1；

actor angle coefficient
才是配置中的 10。

==================================================
九、Workflow State
==================

最终 WorkflowState / AgentState
必须能够持有或引用：

actor params
critic params
target params

optimizer states

replay state

PRNG state

logical worker preference state

worker step counts

episode_count[K]

eval_cnt_ep

eval_cnt

raw key_solutions [3,2]

PDMORLInterpolatorState

training counters

必要 evaluation/control state

所有进入 JAX PyTree 的部分：

fixed shape。

禁止：

SciPy object
Python RBF object
dynamic Pareto array

进入 learner state。

==================================================
十、Checkpoint / Restore
========================

这是 Step7 必须验证的 integration 功能。

checkpoint 至少必须正确恢复：

- Actor；
- Critics；
- targets；
- optimizer states；
- replay state（如果 EvoRL 当前 workflow 保存）；
- total_it；
- preference worker state；
- episode_count；
- eval_cnt_ep；
- eval_cnt；
- raw key solutions；
- current interpolator state。

特别检查：

训练过程中 interpolator
已经 online refit 后：

save
→ restore

恢复出的：

I(w)

必须与 save 前一致。

不要 restore 后
偷偷重新使用 initial artifact 覆盖
在线更新过的 interpolator。

==================================================
十一、Toy Integration Smoke
===========================

可以先使用：

2-objective continuous toy environment

用途仅限：

工程 integration smoke。

验证：

- workflow setup；
- reset；
- rollout；
- vector reward；
- preference；
- replay；
- HER；
- learner；
- angle；
- interpolator；
- control hook；
- evaluator；
- metrics；
- checkpoint。

标记：

TEST / INTEGRATION INFRASTRUCTURE

不要根据 toy 表现：

调整 PD-MORL 算法。

==================================================
十二、Walker End-to-End Smoke
=============================

Toy PASS 后：

使用已经冻结的：

Brax Walker mapping

进行 single-GPU
short end-to-end run。

严格 baseline：

K = 10 logical workers

B = K

GPU = 1

matmul_precision = highest

不要在 Step7
加入 multi-GPU。

==================================================
十三、Walker Smoke 必查项目
===========================

至少检查：

backend == gpu

num logical workers == 10

preference subspace coverage correct

vector reward finite

replay size increases correctly

HER activation timing correct

learner-start timing correct

random-action warmup correct

critic update occurs

actor delayed update occurs

target update occurs

total_it correct

wp finite

directional angle finite

critic loss finite

actor loss finite

gradient norm finite

episode_count grows

key trigger fires when expected

key candidate evaluation runs

interpolator state updates

full evaluation trigger works
（可以使用测试缩短配置验证 plumbing，
但 production defaults 不得改变）

Pareto finite

HV finite

sparsity finite

checkpoint restore works

==================================================
十四、短测试可以使用 Test Override
==================================

因为真实：

key trigger
full evaluation trigger
random warmup

可能需要较长运行才能触发。

允许在：

tests / smoke-only config

缩短：

warmup
evaluation threshold
training length

用于验证 integration plumbing。

但必须明确标记：

TEST OVERRIDE

不能修改：

production Walker baseline defaults。

测试结束后：

production config
仍保持 source-faithful values。

==================================================
十五、Preference Sensitivity
============================

同一个训练 checkpoint
至少测试：

w1 = [0,1]

w2 = [0.5,0.5]

w3 = [1,0]

在相同或受控 initial-state protocol 下：

分别运行 deterministic actor。

记录：

actions

trajectory summary

objective returns

要求：

至少确认 preference input
实际影响 actor computation。

不要仅检查：

网络输入 tensor 不同。

应检查：

输出 action
或 trajectory / return
存在 measurable difference。

==================================================
十六、不要把 Preference Sensitivity
作为论文性能判定
================

本阶段只要求：

preference conditioning
没有在 integration 中丢失。

不要求：

w=[1,0]
一定比另一个 preference
得到某个特定 objective 数值。

不要求：

形成论文质量的完整 Pareto front。

如果短训练下 returns
还没有明显排序：

记录事实。

不要为了让结果“看起来合理”
调整算法。

==================================================
十七、Training Stability Diagnostics
====================================

短程 run 至少记录：

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

Q1 statistics

Q2 statistics

wp statistics

replay size

HER active/inactive

episode counts

preference coverage

key-update count

interpolator refit count

evaluation HV

evaluation sparsity

要求：

所有应该有限的量：

no NaN
no Inf

不要要求：

RL loss 单调下降。

==================================================
十八、Evaluation Integration
============================

不要“新增加 evaluation”。

直接复用 Step6.1：

PDMORLEvaluator

KeyInterpolatorUpdateController

Pareto/HV/sparsity。

Step7 只验证：

它们被正确调用，
使用当前训练中的 actor/interpolator/state，
结果被正确写入 metrics/logger。

==================================================
十九、Logging
=============

尽量复用 EvoRL recorder/logger。

MORL-specific metrics
集中命名。

例如：

morl/train/critic_loss

morl/train/actor_loss

morl/control/key_update_count

morl/control/interpolator_refit_count

morl/eval/hv

morl/eval/sparsity

morl/eval/num_pareto_points

避免：

把 MORL logging
散落到所有底层模块。

==================================================
二十、不要复制无关 TD3
======================

在最终代码审查中：

明确列出：

哪些 EvoRL TD3 functions/classes
被直接复用。

哪些因为 MORL 必须 override。

哪些是新增 PD-MORL helper。

如果发现：

大段 TD3 update/replay/workflow
被复制粘贴：

优先重构为复用。

但不要为了追求零重复
做破坏性的全框架重构。

==================================================
二十一、Regression
==================

完成 integration 后重新运行：

Step4 MO-TD3

Step5.1 math

Step5.2 HER

Step5.3 parallel exploration

Step5.4 interpolator

Step5.5 alignment

Step6 Golden Tests

Step6.1 control/evaluation

scalar TD3 regression

要求：

全部继续 PASS。

Step6：

frozen tolerances 不变。

GPU：

matmul_precision = highest。

==================================================
二十二、Step7 不做 Multi-GPU
============================

虽然实验室有多张 GPU，

Step7 先只冻结：

single-GPU
complete PD-MORL workflow。

原因：

algorithm integration bug

和：

distributed/sharding bug

必须分开。

Multi-GPU
放到单独后续 Step。

==================================================
二十三、Step7 不做正式论文复现
==============================

不要求：

1M environment steps

论文最终 HV

论文最终 sparsity

six-run mean/std reproduction

Figure 5/6/7 reproduction

这些属于后续：

paper reproduction stage。

Step7 的任务是：

证明完整系统
工程与算法链闭合。

==================================================
二十四、分类
============

文档中继续使用：

SOURCE-FAITHFUL

FRAMEWORK-ADAPTATION

TEST FIXTURE

TEST OVERRIDE

DEVIATION

典型：

SOURCE-FAITHFUL：
所有 PD-MORL algorithm/control semantics。

FRAMEWORK-ADAPTATION：
PyTorch/processes → JAX/Brax/workflow hooks。

TEST FIXTURE：
toy environment / synthetic data。

TEST OVERRIDE：
为了 smoke 缩短的触发 threshold。

任何算法行为变化：

DEVIATION。

==================================================
二十五、文档
============

创建：

docs/PD_MORL_EVORL_BASELINE.md

至少记录：

1. Scope
2. Integration audit
3. Final architecture
4. Reused EvoRL components
5. PD-MORL-specific components
6. Agent responsibilities
7. Workflow responsibilities
8. Full training dataflow
9. Control-plane dataflow
10. Evaluation-plane dataflow
11. Config
12. Workflow state
13. Checkpoint / restore
14. Toy smoke
15. Walker smoke
16. Preference sensitivity
17. Stability diagnostics
18. Evaluation integration
19. Regression
20. Source-faithful / adaptation table
21. Remaining known deviations
22. Remaining work before paper reproduction

==================================================
二十六、PASS Gate
=================

只有以下全部满足：

1. 一个清晰的 PDMORL Agent / Workflow 已形成；
2. 最大程度复用 EvoRL TD3，
   没有无意义复制整套 TD3；
3. learner plane 使用
   已经冻结的 Step4-Step6 实现；
4. control/evaluation plane 使用
   已经冻结的 Step6.1 实现；
5. toy end-to-end smoke PASS；
6. real Walker single-GPU smoke PASS；
7. vector reward / preference / replay / HER
   完整连通；
8. critic updates 正常；
9. delayed actor update 正常；
10. target updates 正常；
11. wp / angle 正常；
12. episode counters 正常；
13. key-update trigger 正常；
14. online interpolator state
    能够真实更新；
15. evaluation 能从当前 actor
    得到 vector returns；
16. Pareto / HV / sparsity 正常；
17. preference 对 actor behavior
    产生 measurable effect；
18. no NaN / Inf；
19. checkpoint / restore
    保持完整 PD-MORL state；
20. Step4-Step6.1
    所有 frozen regression PASS；
21. Step6 Golden Test tolerance
    完全未修改；
22. production baseline config
    没有被 smoke-test overrides 污染；

才能输出：

STEP7 END-TO-END EVORL PD-MORL PASS

否则：

输出：

STEP7 INTEGRATION MISMATCH REPORT

并停止。

不要继续：

multi-GPU
paper reproduction
PSL-MORL。

完成后停止。
