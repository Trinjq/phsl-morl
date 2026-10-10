> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入：

STEP9.2 — ONE FULL WALKER REPRODUCTION SEED

目标：

运行一个完整的、source-equivalent budget 的
PD-MORL Walker experiment。

这不是 smoke test。
这不是 reduced-budget test。
这不是 multi-GPU data-parallel test。

只运行：

ONE PROCESS
ONE GPU
ONE COMPLETE TRAINING SEED

==================================================
0. 冻结前置条件
===============

必须基于已经 PASS 的：

STEP9.0 PD-MORL REPRODUCTION PROTOCOL

不得修改：

- PD-MORL learner math
- Actor/Critic architecture
- K=10
- batch=256
- HER=3
- replay capacity=2,000,000
- gamma=.995
- tau=.005
- policy_freq=10
- exploration noise=.1
- target noise=.2
- target noise clip=.5
- actor loss coefficient=10
- gradient clip global norm=100
- interpolator semantics
- control triggers
- evaluation protocol
- Step6 tolerance
- training budget accounting

禁止重新启用：

multi-GPU data-parallel learner。

==================================================

1. 使用的正式实验
   ==================================================

Environment:

Walker / current Brax PD-MORL adapter

Training seed:

1

注意：

seed=1 只是 official executable source 的默认 seed。

不要标记为：

"one of the official six paper seeds"

因为官方没有公开六个具体 training seed 数值。

正式名称使用：

source-faithful PD-MORL reproduction
with Brax environment adaptation

不要称为：

exact MuJoCo paper reproduction。

==================================================
2. GPU 隔离
===========

只使用一张 GPU。

例如：

CUDA_VISIBLE_DEVICES=0

child process 内必须确认：

jax.local_device_count() == 1

并打印：

GPU model
GPU UUID if available
JAX backend
visible GPU count
JAX device
dtype
matmul precision

正式训练路径中不得出现：

psum
pmean
distributed learner
multi-device Mesh
batch sharding
padding
validity masks。

==================================================
3. 正式配置
===========

必须使用：

configs/experiment/pd_morl_walker_reproduction.yaml

训练开始前打印并验证：

SOURCE_N = 1,000,000

K = 10

PREFILL_GLOBAL_TRANSITIONS = 1,530

PREFILL_STEPS_PER_WORKER = 153

POST_PREFILL_WORKFLOW_ROUNDS = 999,847

EXPECTED_FINAL_WORKER_STEPS = 1,000,000

EXPECTED_VALID_ENV_TRANSITIONS = 10,000,000

EXPECTED_CRITIC_OPTIMIZER_STEPS = 9,998,470

EXPECTED_ACTOR_OPTIMIZER_STEPS = 999,847

EXPECTED_TARGET_UPDATES = 999,847

config.total_timesteps = 10,000,000

如果任何一个不一致：

不要开始正式训练。

输出：

STEP9.1 PRE-RUN CONFIG MISMATCH

并停止。

==================================================
4. Production network 必须验证
==============================

训练前输出真实 parameter shapes。

Actor：

19 -> 400 -> 400 -> 6

Critic Q1：

25 -> 400 -> 400 -> 2

Critic Q2：

25 -> 400 -> 400 -> 2

Q1/Q2 参数必须独立。

不要允许：

small-network override。

==================================================
5. Preference / worker 配置
===========================

必须确认：

K = 10 logical preference workers

10 个 preference subspaces
全部存在。

每个完整 training run
都包含 worker IDs：

0..9

GPU 数量不能改变 K。

Episode preference sampling
和 frozen subspace semantics
保持 Step5/Step7 行为。

==================================================
6. Replay / HER
===============

正式 run 中：

只有一个逻辑 global replay。

确认：

capacity = 2,000,000

base transition
和 HER relabeled transition
都写入该 replay。

HER activation threshold
继续使用已经冻结的规则。

HER entries：

不得计入：

environment transitions。

长期训练中记录：

replay size
base inserts
HER inserts
HER active flag。

==================================================
7. 正式 learner
===============

整个 run 中持续检查：

critic loss finite
actor loss finite
Q finite
target Q finite
gradient norm finite
wp finite

至少记录：

critic_total_loss
critic_smooth_l1
critic_angle
actor_total_loss
actor_scalarized_term
actor_angle
critic_raw_grad_norm
actor_raw_grad_norm
Q1 min/mean/max
Q2 min/mean/max
target min/mean/max
wp min/mean/max。

不要因为数值大或曲线震荡
自行停止训练。

只有出现：

NaN
Inf
device failure
checkpoint corruption
counter mismatch

才标记运行错误。

==================================================
8. Counter invariant
====================

训练期间持续或周期性验证：

worker_steps[i]
不能超过 source budget。

结束时必须严格：

worker_steps[i] == 1,000,000
for every i in 0..9

sampled_timesteps == 10,000,000

critic optimizer steps == 9,998,470

actor optimizer steps == 999,847

target updates == 999,847

prefill + post-prefill transitions
==================================

10,000,000

如果最终计数不一致：

训练结果不得作为正式 reproduction result。

==================================================
9. Control-plane behavior
=========================

不要关闭：

key replacement
online interpolator refit
full evaluation

正式 run 要使用真实控制面。

持续记录：

episode_count[10]
eval_cnt_ep
eval_cnt
key evaluation count
key replacement count
interpolator refit count

每次 key trigger：

必须只执行一次。

每次 full evaluation：

必须只执行一次。

==================================================
10. Training evaluation
=======================

保持冻结 protocol：

training evaluation：

201 preferences
×
3 repeats

deterministic policy

undiscounted vector returns

HV reference = [0,0]

记录：

training progress
source_worker_steps
valid_env_transitions
HV
sparsity
Pareto point count。

不要为了节省时间
降低正式 evaluation grid/repeats。

==================================================
11. Training-final evaluation
=============================

训练结束后：

training final hook：

1001 preferences
×
3 repeats

保存：

raw returns
per-repeat metrics
mean HV
mean sparsity
Pareto front
Pareto point count。

这个结果是：

training-final result。

不要和 paper-report offline result 混为一谈。

==================================================
12. Paper-report offline evaluation
===================================

完整训练结束并成功保存 checkpoint 后，

单独执行：

evaluate_offline()

使用：

1001 preferences
×
6 repeats

evaluation seeds：

[0,11,22,33,44,55]

得到最终 paper-report：

HV per repeat
HV mean
HV std

sparsity per repeat
sparsity mean
sparsity std

mean objective returns
final non-dominated front。

这才作为后续 6-training-seed
实验汇总的单-run final metric。

==================================================
13. Pareto progression
======================

正式训练必须保留：

HV progression
sparsity progression
Pareto progression。

至少确保可以得到：

early
middle
late
final

几个阶段的 Pareto front。

优先使用已有 source-faithful
evaluation trigger。

不要新增高频 evaluation
改变训练时间行为。

==================================================
14. Checkpoint
==============

正式 run 必须启用 checkpoint。

至少保证：

最新 checkpoint
最终 checkpoint

可以恢复：

actor
critics
target actor
target critics
optimizer
replay
PRNG
worker_steps
episode_count
total_it
eval counters
key solutions
interpolator。

训练结束后：

执行一次 checkpoint restore verification。

恢复后至少验证：

final deterministic actor outputs
关键 counters
interpolator arrays

与保存前一致。

==================================================
15. 输出目录
============

这个 full seed 必须有独立永久目录。

推荐：

outputs/pd_morl/walker/reproduction/seed_1/

保存：

resolved_config.yaml
run_metadata.json
training log
pd_morl_evaluations.jsonl
checkpoints/
final_training_eval/
offline_eval/
pareto_fronts/
summary.json

不要写进 smoke-test 临时目录。

==================================================
16. Run metadata
================

保存：

algorithm = PD-MORL
training_seed = 1
environment = Walker Brax adapter
backend = Brax
git commit
date
GPU model
GPU UUID
JAX version
Brax version
Python version
CUDA version if available

以及：

source N
K
batch
network architecture
all frozen hyperparameters。

==================================================
17. 运行资源监控
================

记录：

wall-clock start
wall-clock end
total runtime

peak GPU memory if practical

checkpoint disk usage
final output disk usage。

可以记录：

GPU utilization

但性能不是 PASS gate。

==================================================
18. 不要用论文 HV 当在线终止条件
================================

论文参考：

HV:
5.41 ± 0.004 × 10^6

Sparsity:
0.03 ± 0.005 × 10^4

只作为：

REFERENCE ONLY。

当前运行是：

Brax environment adaptation

不是原 MuJoCo backend。

因此禁止：

HV 没达到 5.41e6
→ 自动判算法失败

也禁止：

HV 达到 5.41e6
→ 自动宣称 exact reproduction。

==================================================
19. 训练结束后的 sanity analysis
================================

完整 seed 完成后检查：

A. Preference sensitivity

至少检查：

w=[0,1]
w=[0.5,0.5]
w=[1,0]

确定性 actions
不能全部完全相同。

B. Pareto front

必须存在非空
non-dominated set。

C. HV

必须 finite。

D. Sparsity

必须 finite。

E. Loss

整个后期训练
不得持续 NaN/Inf。

F. Objective trade-off

不同 preferences
应出现可观察的
return trade-off。

这不是要求：

目标严格单调。

==================================================
20. 自动生成单 Seed 报告
========================

训练结束后创建：

docs/PD_MORL_WALKER_SEED1_REPRODUCTION.md

内容：

1. run identity
2. exact config
3. hardware/software
4. training budget
5. final counters
6. runtime
7. loss stability
8. replay/HER behavior
9. key/interpolator behavior
10. HV progression
11. sparsity progression
12. Pareto progression
13. training-final 1001x3 result
14. offline 1001x6 result
15. final Pareto front
16. preference sensitivity
17. checkpoint restore
18. comparison with paper reference
19. known Brax/MuJoCo differences
20. classification
21. conclusion

==================================================
21. Classification
==================

SOURCE-FAITHFUL：

PD-MORL algorithm
K
replay/HER
loss
target
interpolator semantics
update schedule
evaluation metrics
budget semantics。

FRAMEWORK-ADAPTATION：

JAX
Brax
single-GPU execution
PRNG lane mapping
Optax equivalent clipping
checkpoint/logger machinery。

ENVIRONMENT DEVIATION：

Brax physics/reset/termination
relative to original MuJoCo benchmark。

EXPERIMENT SEED：

1

不要称为 official paper seed。

==================================================
22. PASS Gate
=============

只有下面全部满足，
才输出：

STEP9.1 ONE FULL WALKER SEED PASS

要求：

1. exactly one GPU；
2. production [400,400] networks；
3. K=10；
4. batch=256；
5. final worker_steps exactly 1,000,000；
6. valid environment transitions exactly 10,000,000；
7. critic updates exactly 9,998,470；
8. actor updates exactly 999,847；
9. target updates exactly 999,847；
10. HER active；
11. replay semantics correct；
12. no NaN/Inf；
13. key/control plane executed correctly；
14. training evaluations completed；
15. final 1001x3 evaluation completed；
16. offline 1001x6 evaluation completed；
17. finite HV；
18. finite sparsity；
19. nonempty Pareto front；
20. preference-sensitive policy；
21. final checkpoint valid；
22. checkpoint restore PASS；
23. persistent report created。

==================================================
23. STOP BOUNDARY
=================

完成一个 seed 后立即停止。

不要：

启动 seed2
启动 6-seed experiment
修改 hyperparameters
根据结果调参
修改 Brax reward
修改 evaluation
开始 PSL-MORL。

先把：

docs/PD_MORL_WALKER_SEED1_REPRODUCTION.md

和最终结果交给人工审查。
