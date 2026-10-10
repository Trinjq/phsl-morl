> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入：

Step 9.0 — PD-MORL Formal Reproduction Protocol Audit

不要立即启动 1M-step 正式训练。

当前已经冻结并通过：

- Step7 single-GPU end-to-end；
- Step7.1 production-shape single-GPU；
- Step7.2 reduced-network removal；
- Step8 independent-seed multi-GPU runner；
- 3 GPU / 3 independent single-GPU processes
  已实际进入 learner update；
- production Walker:
  Actor [400,400]
  Q1/Q2 [400,400]
  K=10
  batch=256。

本阶段唯一目标：

把“PD-MORL 官方实验预算”
精确映射到
“当前 EvoRL-PD-MORL workflow counters”，

并生成一个可用于下一步
单 seed 正式 reproduction 的配置。

==================================================
一、事实来源优先级
==================

按以下顺序审计：

1. PD-MORL 官方 executable source；
2. PD-MORL 论文 Appendix / Algorithm；
3. 当前 EvoRL-PD-MORL implementation；
4. 已冻结 Step3-Step8 文档。

源码与论文若不一致：

记录：

SOURCE EXECUTABLE BEHAVIOR

并以源码作为 implementation baseline。

不要凭经验解释：

“1M steps”。

==================================================
二、先审计官方 N 的真实语义
===========================

官方论文写：

每个 child process runs for N time steps，

因此：

total collected transitions
===========================

N × Cp

Walker：

N = 1,000,000
Cp = 10。

但不要仅凭论文文字直接实现。

检查官方：

train_Walker2d_MO_TD3_HER.py

以及：

process worker loop
main-process loop
settings.py

确认：

1. N / max_timesteps 对谁计数；
2. 是每 child 1,000,000，
   还是全局 1,000,000；
3. 每轮 main process
   实际从多少 worker 接收 transition；
4. 每轮实际执行多少 learner updates；
5. worker step counter
   如何增长；
6. global main counter
   如何增长；
7. termination condition
   用哪个 counter。

输出：

OFFICIAL TRAINING BUDGET TABLE

至少包括：

counter name
initial value
increment location
increment amount
termination expression
physical meaning。

==================================================
三、重点确认环境 transition 总量
================================

如果源码确认：

每个 K=10 child
各跑 N=1,000,000 environment steps，

则记录：

source_worker_steps
===================

1,000,000

source_valid_env_transitions
============================

10,000,000

但只有源码确认后
才能写这个结论。

不要把：

HER relabeled entries

计入：

environment transitions。

HER 是 replay expansion，
不是额外 environment interaction。

==================================================
四、审计当前 EvoRL 所有时间计数器
=================================

找到并解释：

total_timesteps
sampled_timesteps
env_steps
worker_steps
total_it
episode_count
learner_updates
critic_updates
actor_updates
target_updates
replay size
HER inserted entries

如果还有其他相关 counter：

全部加入。

对每个 counter 写：

- shape；
- 初值；
- 每次 workflow round 增量；
- 是否乘 K；
- 是否包含 HER；
- 是否用于 stop；
- 是否用于 logging；
- 是否用于 evaluation trigger。

==================================================
五、特别解释当前 Step8 Smoke 的更新数字
=======================================

当前 3-GPU independent learner smoke：

每个 run 报告：

critic updates = 133120
actor updates = 13312
target updates = 13312
replay size = 238640

必须解释：

这些数字的“单位”到底是什么。

例如确认：

critic_updates

究竟表示：

A. optimizer.step 次数

B. source-equivalent worker learner updates

C. total_it 增量

D. minibatch sample count

E. vectorized K-expanded counter

不要只说：

“比例等于 policy_freq=10”。

必须追代码得到精确定义。

==================================================
六、建立统一 Reproduction Counters
==================================

为了以后不再混淆，

建议在 metrics/documentation 中明确区分：

source_worker_steps

valid_env_transitions

workflow_rounds

replay_base_inserts

replay_her_inserts

replay_total_entries_written

critic_optimizer_steps

actor_optimizer_steps

target_update_steps

total_it

不要用一个：

total_timesteps

同时指多个东西。

这一步主要是：

logging / accounting improvement。

禁止改变算法调度。

==================================================
七、推导 Source ↔ EvoRL 映射公式
=================================

最终必须给出明确公式。

例如形式：

K = 10

每个 EvoRL rollout round：

logical env transitions
=======================

K

如果：

R = workflow rounds

则：

valid_env_transitions
=====================

R × K

每个 logical worker steps
=========================

R

但不要预设公式一定如此。

必须根据实际代码确认。

同理推导：

critic optimizer steps
======================

?

actor optimizer steps
=====================

?

replay writes
=============

?

HER writes
==========

?

==================================================
八、审计 source learner update ratio
====================================

重点检查官方源码：

每收到 K 个 worker transitions 后，

究竟执行：

1 learner update

还是：

K learner updates。

当前 Step5.3 曾按 source mapping
实现 parallel update schedule。

重新核对 executable source，

并证明当前 EvoRL：

| environment transitions |
| :---------------------: |
|     critic updates     |

与官方一致。

给出具体 ratio。

==================================================
九、审计 Warm-up 语义
=====================

重新确认：

random-action warmup

learner-start threshold

HER activation threshold

在官方源码中分别基于：

- worker steps？
- global transitions？
- replay size？
- total_it？

然后映射到 EvoRL。

禁止：

因为正式训练改写
Step5.2/Step5.3 已冻结语义。

这里只做审计。

==================================================
十、审计 Evaluation Timing
==========================

不要把正式 evaluation
简单改成固定 timestep interval。

Step6.1 已冻结：

key-update trigger

和：

full-evaluation trigger

基于 cumulative episode_count。

确认长训练下：

这些 trigger
继续使用现有 source-faithful semantics。

同时记录：

一次 1M reproduction
预计会执行多少次：

key evaluation
full training evaluation

如果次数只能运行时确定：

说明原因。

==================================================
十一、审计 Main Training 与 Key Pretraining
===========================================

区分：

A. key-solution pretraining

B. PD-MORL main MO-TD3-HER training。

当前 production baseline
从：

interp_objs_walker2d.txt

加载已追溯 key artifact。

确认论文表中的：

1×10^6 main training steps

是否包括：

key-pretraining compute。

不要猜。

如果官方指标
只对应 main run：

记录：

MAIN TRAINING BUDGET

和：

KEY PRETRAINING BUDGET

为两个独立实验成本。

==================================================
十二、正式 Walker Hyperparameter Audit
======================================

逐项比较：

Official PD-MORL Walker
vs
Current EvoRL production config。

至少：

N
batch size
gamma
tau
replay capacity
K
HER relabel count
critic LR
actor LR
hidden architecture
policy delay
exploration noise
target noise
noise clip
actor loss coefficient
gradient clipping
episode limit
reward definition
done semantics
key solutions
interpolator
evaluation grids。

输出：

MATCH
FRAMEWORK-ADAPTATION
KNOWN ENVIRONMENT DEVIATION
MISMATCH

任何真正 MISMATCH：

停止 Step9.0。

==================================================
十三、环境差异必须单列
======================

正式结果环境是：

current EvoRL/Brax Walker mapping。

原论文是：

MO-Walker2d-v2 / original MuJoCo path。

因此建立：

ENVIRONMENT COMPARABILITY TABLE

至少列：

obs dim
action dim
reward objective 1
reward objective 2
termination
episode limit
reset
physics backend。

不要把：

Brax结果

称为：

exact paper reproduction

除非环境真正完全相同。

建议正式命名：

EvoRL/Brax PD-MORL reproduction baseline

或：

source-faithful algorithm reproduction
with Brax environment adaptation。

==================================================
十四、Training Seed Audit
=========================

论文明确使用六次 runs，

但不要自动认为：

0,1,2,3,4,5

是官方 training seeds。

搜索：

official source
scripts
README
supplement

确认是否明确给出
六个 training seed 数值。

如果找不到：

输出：

OFFICIAL TRAINING SEEDS NOT SPECIFIED

后续我们自己的：

0..5

或其他六个 seed

只能标记：

EXPERIMENT SEEDS

不能标记：

official seeds。

==================================================
十五、Evaluation Protocol
=========================

保持 Step6.1 冻结：

training evaluation：
201 preferences
3 repeats

final/offline：
1001 preferences
offline 6 repeats

evaluation seed rule：
不由 training_seed 改变。

HV：

reference point = [0,0]

Pareto / sparsity
继续当前冻结实现。

不要为了接近论文数字
修改 evaluation。

==================================================
十六、论文 Walker Reference 仅作 Reference
==========================================

记录论文 reported values：

MO-Walker2d-v2

HV:
5.41 ± 0.004 × 10^6

Sparsity:
0.03 ± 0.005 × 10^4

runs:
6

HV reference:
(0,0)

但明确：

由于当前使用 Brax backend，

这些数字：

REFERENCE ONLY

不是 Step9 PASS threshold。

禁止写：

HV 必须达到 5.41e6
否则代码失败。

==================================================
十七、检查正式输出数据是否足够
==============================

在启动正式 run 前，

确认每个 seed 能持久保存：

resolved config

training_seed

git commit

environment/backend

training counters

environment transitions

learner updates

critic loss

actor loss

HV over training

sparsity over training

Pareto front snapshots

final 1001-point returns

final Pareto front

final HV

final sparsity

runtime

checkpoint。

如果缺少：

HV curve

或：

Pareto progression

现在补 logging。

但不要改算法。

==================================================
十八、Pareto Progression Snapshots
==================================

为了后续和论文 Figure 6/7
进行趋势比较，

建立几个固定的
source-equivalent training progress points。

例如：

0%
25%
50%
75%
100%

或者：

按官方实际 eval trigger
能自然对应的 checkpoints。

优先保持 source evaluation timing。

不要为了漂亮图
引入高频昂贵 evaluation。

记录：

source-equivalent worker steps

而不仅是：

wall-clock time。

==================================================
十九、正式训练前 Dry-Run Counter Test
=====================================

不要直接跑 1M。

建立一个小型：

COUNTER SEMANTICS TEST。

例如运行：

source-equivalent
100 worker steps

或其他小整数。

运行结束后人工可计算：

expected:

workflow rounds
valid env transitions
per-worker steps
critic updates
actor updates
base replay inserts
HER inserts

然后与程序实际值逐项一致。

这是 Step9.0
最重要的 executable test。

必须使用：

K=10
production network
batch256

允许缩短：

warmup/trigger

以验证 counter plumbing。

==================================================
二十、正式预算配置
==================

Audit 完成后：

不要直接运行。

生成一个：

configs/experiment/pd_morl_walker_reproduction.yaml

或等价配置。

这个配置必须表达：

“官方 N=1,000,000 的
source-equivalent training budget”

而不是仅仅机械写：

total_timesteps=1000000。

如果需要新增更明确的配置：

source_worker_steps: 1000000

可以考虑。

然后由代码
明确转换成当前 workflow stop condition。

但：

转换必须经过测试。

==================================================
二十一、建议加入防误用 Guard
============================

如果：

total_timesteps

语义容易与：

source N

混淆，

正式 reproduction config
不要只依赖裸：

total_timesteps。

增加日志：

OFFICIAL_SOURCE_N
EXPECTED_VALID_ENV_TRANSITIONS
EXPECTED_WORKFLOW_ROUNDS
EXPECTED_CRITIC_UPDATES

训练启动时打印。

防止以后：

1M 与 10M

再次混淆。

==================================================
二十二、估算单 Seed 成本
========================

基于 Step7.1 / Step8
已有 timing，

只做粗略：

runtime estimate

disk estimate

checkpoint estimate

evaluation cost estimate。

标记：

ESTIMATE

不要把估算当实际结果。

目的是判断：

一个完整 seed
大概需要多久，

再决定何时启动 6 seeds。

==================================================
二十三、创建文档
================

创建：

docs/PD_MORL_REPRODUCTION_PROTOCOL.md

至少包含：

1. Official source budget semantics
2. Official paper budget
3. EvoRL counter semantics
4. Source↔EvoRL mapping
5. Environment transition definition
6. HER accounting
7. Learner update ratio
8. Warmup accounting
9. Evaluation timing
10. Key-pretraining accounting
11. Hyperparameter parity table
12. Environment comparability
13. Training seed audit
14. Evaluation protocol
15. Paper reference metrics
16. Dry-run counter test
17. Formal reproduction config
18. Expected runtime/resources
19. Remaining deviations

==================================================
二十四、PASS Gate
=================

Step9.0 只有以下全部明确后才 PASS：

1. official N exact meaning known；
2. official child-process stop condition known；
3. official total environment-transition count known；
4. current total_timesteps meaning known；
5. current workflow-round meaning known；
6. current total_it meaning known；
7. critic update counter meaning known；
8. actor update counter meaning known；
9. HER not counted as env interaction；
10. source learner/update ratio matched；
11. warmup semantics matched；
12. evaluation trigger semantics unchanged；
13. key-pretraining budget separated；
14. production hyperparameters audited；
15. environment deviations documented；
16. official seed values searched；
17. 6-run protocol documented；
18. dry-run counter arithmetic exactly passes；
19. formal reproduction config created；
20. no frozen algorithm behavior changed；
21. no Step6 tolerance changed。

全部通过后输出：

STEP9.0 PD-MORL REPRODUCTION PROTOCOL PASS

同时给出下一步建议命令：

ONE FULL WALKER SEED

但：

不要执行该完整 run。

==================================================
二十五、失败条件
================

如果发现：

官方 N 的语义
和当前 EvoRL stop condition
无法一一映射，

输出：

STEP9.0 TRAINING-BUDGET MISMATCH REPORT

必须包含：

official semantics
current semantics
difference factor
affected counters
recommended mapping

然后停止。

不要：

直接启动 1M。
不要：
启动 6 seeds。
不要：
修改算法。
不要：
开始 PSL-MORL。
