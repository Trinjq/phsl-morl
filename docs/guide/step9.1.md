
修正 Step9.0 reproduction protocol。

当前不要启动正式 Walker run。

发现以下问题：

1. source budget 被 prefill 重复计数；
2. official network architecture classification 错误；
3. gradient clipping classification 错误；
4. final/offline evaluation repeat semantics 需要恢复明确边界。

==================================================

1. 修正 source-equivalent training budget
   ==================================================

官方 executable source 已确认：

N = 1,000,000 rounds per worker
K = 10

因此：

TOTAL VALID ENV TRANSITIONS
===========================

N * K
=====

10,000,000

这是包含 warm-up / early transitions 在内的完整 source budget。

EvoRL prefill：

1530 global transitions
=======================

153 transitions per logical worker

已经属于这 1,000,000 worker steps，
不能再额外加到官方 N 外。

因此正式：

config.total_timesteps

如果该字段包含 prefill，
必须等于：

10,000,000

而不是：

10,001,530。

prefill 后：

remaining worker rounds
=======================

1,000,000 - 153
===============

999,847

remaining valid env transitions
===============================

999,847 * 10
============

9,998,470。

验证最终：

worker_steps[i] == 1,000,000

而不是：

1,000,153。

==================================================
2. 修正 learner-update accounting
=================================

重新从实际 learner-start gate 推导：

正式 run 中：

critic_optimizer_steps
actor_optimizer_steps
target_updates

不要自动假设：

critic_optimizer_steps = 10,000,000。

区分：

environment transition
learner call
actual optimizer application。

当前 source/EvoRL learner start
在 replay threshold 后才真正更新。

给出最终精确公式和数字。

==================================================
3. 修正 dry-run counter test
============================

当前所谓：

source_worker_steps=100

却得到：

30 prefill + 1000 workflow transitions

等于：

103 steps/lane。

这是错误的 source-budget test。

修改为：

总 source worker steps = 100

如果 prefill = 3 steps/lane，

则 post-prefill workflow rounds：

100 - 3 = 97。

最终断言：

worker_steps == 100
valid_env_transitions == 1000

不能是：

worker_steps == 103
valid_env_transitions == 1030。

==================================================
4. 修正 network architecture classification
===========================================

不要只根据论文 Table 5 的：

"1 hidden layer, 400"

判断 executable architecture。

官方 source：

lib/models/networks.py

Actor：

19 -> 400 -> 400 -> 6

Critic Q1/Q2：

25 -> 400 -> 400 -> 2

因此当前 EvoRL：

[400,400]

应分类为：

SOURCE-FAITHFUL / MATCH

同时记录：

PAPER/SOURCE DISCREPANCY：

论文表格写法与 executable source architecture 不一致。

按照本项目既定规则：

official executable source wins。

==================================================
5. 修正 gradient clipping classification
========================================

官方 source 明确：

critic:
clip_grad_norm_(critic.parameters(), max_norm=100)

actor:
clip_grad_norm_(actor.parameters(), max_norm=100)

默认 total L2 norm。

因此：

gradient clipping threshold/order/math

分类：

SOURCE-FAITHFUL / MATCH。

PyTorch clip_grad_norm_
映射到 JAX/Optax global-norm clip
只是：

FRAMEWORK IMPLEMENTATION MAPPING，

不是算法参数 deviation。

==================================================
6. 恢复 evaluation 边界
=======================

冻结 baseline：

training eval:
201 preferences × 3 repeats

training final hook:
1001 preferences × 3 repeats

explicit offline / paper-report evaluation:
1001 preferences × 6 repeats

不要把 final hook 从 3
静默改成 6
同时声称 evaluation unchanged。

正式论文汇报值：

在训练结束后
显式调用 offline six-repeat evaluator。

==================================================
7. 更新 formal reproduction config
==================================

重新生成：

configs/experiment/pd_morl_walker_reproduction.yaml

确保：

expected source worker steps = 1,000,000
expected valid env transitions = 10,000,000
total_timesteps includes prefill exactly once

启动日志打印：

SOURCE_N
K
PREFILL_GLOBAL_TRANSITIONS
PREFILL_STEPS_PER_WORKER
POST_PREFILL_WORKFLOW_ROUNDS
EXPECTED_FINAL_WORKER_STEPS
EXPECTED_VALID_ENV_TRANSITIONS
EXPECTED_CRITIC_OPTIMIZER_STEPS
EXPECTED_ACTOR_OPTIMIZER_STEPS
EXPECTED_TARGET_UPDATES

==================================================
8. 加 regression guard
======================

增加：

assert final_worker_steps == source_N

assert sampled_timesteps == source_N * K

assert prefill + post_prefill transitions
       == source_N * K

禁止：

prefill + source_N*K

这种 double counting。

==================================================
9. 更新文档
===========

修订：

docs/PD_MORL_REPRODUCTION_PROTOCOL.md

明确记录：

- previous Step9.0 budget double-count issue；
- corrected accounting；
- paper/source network discrepancy；
- clipping corrected classification；
- final vs offline evaluation distinction。

==================================================
10. PASS gate
=============

只有：

worker final = 1,000,000 exactly
global environment transitions = 10,000,000 exactly
prefill counted exactly once
dry-run exact arithmetic PASS
network classification corrected
gradient clipping classification corrected
evaluation protocol restored

才输出：

STEP9.0 PD-MORL REPRODUCTION PROTOCOL PASS

然后停止。

不要启动正式 full seed。
