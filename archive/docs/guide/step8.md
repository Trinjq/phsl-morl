> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入 Step 8：

Multi-GPU EvoRL PD-MORL

前提已经冻结并通过：

- Step7 End-to-End single-GPU PASS；
- Step7.1 Production-Shape Single-GPU PASS；
- Step7.2 已删除所有 real-Walker reduced-network override；
- production Walker Actor = [400,400]；
- production Walker Q1/Q2 = [400,400]；
- learner batch_size = 256；
- logical preference workers K = 10；
- B = K = 10；
- single-GPU control/evaluation/checkpoint 全部 PASS。

本阶段的唯一目标：

在“不修改任何 PD-MORL 算法语义”的情况下，
将已经冻结的 single-GPU baseline
映射到多 GPU。

本阶段只允许修改：

- device placement；
- rollout sharding；
- learner batch sharding；
- parameter replication；
- gradient synchronization；
- PRNG/device mapping；
- multi-device checkpoint placement；
- padded physical slots / validity mask；
- distributed metrics reduction。

禁止修改：

- network architecture；
- batch_size=256；
- K=10；
- replay semantics；
- HER；
- preference sampling；
- scalarization；
- vector Bellman target；
- actor/critic loss；
- angle；
- interpolator；
- key-update semantics；
- evaluation semantics；
- production config；
- Step6 tolerances。

==================================================
一、先做 MULTI-GPU AUDIT
========================

不要直接写多 GPU 代码。

先审查 EvoRL 当前已有的多设备支持。

检查：

- jax.devices()
- local_device_count
- NamedSharding
- Mesh
- PartitionSpec
- pmap
- shard_map
- device_put
- existing EvoRL distributed helpers
- replay-buffer sharding support
- optimizer state sharding
- checkpoint restore with sharded arrays
- Brax multi-device batching

优先复用 EvoRL 已有机制。

不要自行建立第二套 distributed framework，
除非当前 EvoRL 确实没有可复用实现。

先输出：

STEP8 MULTI-GPU AUDIT

至少回答：

1. 当前实验机 GPU 数；
2. 每张 GPU 型号；
3. EvoRL 当前 multi-device abstraction；
4. parameter 是否适合 replicated；
5. optimizer state 是否适合 replicated；
6. env state 如何分片；
7. replay 当前物理位置；
8. sampled minibatch 如何分片；
9. gradient 如何 aggregate；
10. checkpoint 如何保存 distributed state；
11. 2 GPU 是否可无 padding；
12. 3 GPU 哪些维度需要 padding；
13. 是否存在任何算法修改风险。

==================================================
二、总体架构原则
================

必须继续保持：

10 logical preference workers

与：

physical GPU count

完全独立。

GPU 数：

D

不是：

process_count。

永远：

K = 10。

Multi-GPU 只能改变：

logical worker
→ physical device placement

不能改变：

preference subspace 数量。

==================================================
三、正确性优先：Replay 保持一个逻辑全局 Replay
==============================================

不要为每张 GPU 建独立 replay。

必须继续保持：

ONE LOGICAL GLOBAL REPLAY

即：

所有 10 logical workers
产生的 transitions

最终进入同一个：

sampling population
size counter
write ordering domain
capacity domain。

禁止：

GPU0 replay
GPU1 replay
GPU2 replay

各自独立训练。

否则会改变 PD-MORL 的采样分布。

---

第一版推荐实现
--------------

 correctness-first：

rollout devices
      ↓
collect valid logical-worker transitions
      ↓
global replay/controller
      ↓
sample ONE global batch=256
      ↓
shard learner batch across GPUs

这属于：

FRAMEWORK-ADAPTATION

但保持：

SOURCE-FAITHFUL replay semantics。

后续如果要优化 distributed replay：

另开性能步骤。

Step8 不做。

==================================================
四、Parameters / Optimizer State
================================

第一版采用：

replicated model state。

所有 learner GPUs
持有相同：

actor params
critic params
target params
optimizer states
interpolator arrays
global learner counters。

不要：

每张 GPU 各自独立更新一套 model。

每次 learner update 后：

所有 replicas
必须保持数值一致。

增加 assertion：

max parameter difference between replicas
≈ 0 within numerical tolerance。

==================================================
五、Step8.1：先做 2-GPU Balanced Mode
=====================================

先只启用：

CUDA_VISIBLE_DEVICES=0,1

确认：

jax.local_device_count() == 2

不要一开始就上 3 GPU。

原因：

K=10
可以均匀分为：

GPU0:
logical workers 0..4

GPU1:
logical workers 5..9

即：

5 logical lanes / GPU。

learner batch：

256 / 2 = 128

所以：

128 valid samples / GPU

不需要：

padding
mask
uneven reduction。

Step8.1 作为 distributed correctness Golden Gate。

==================================================
六、2-GPU Rollout Placement
===========================

logical worker mapping 固定：

GPU0:
worker IDs [0,1,2,3,4]

GPU1:
worker IDs [5,6,7,8,9]

worker ID：

不能因为 GPU 数变化而改变。

preference subspace：

worker i
仍然使用 single-GPU 时的
subspace i。

PRNG：

必须基于：

logical_worker_id

而不是：

physical_device_id。

禁止：

GPU0 全部共享一个 worker RNG；
GPU1 全部共享另一个 RNG。

device placement
不能改变 logical identity。

==================================================
七、2-GPU Global Replay
=======================

每个 rollout round：

GPU0 返回 workers 0..4 transitions

GPU1 返回 workers 5..9 transitions

然后按照 logical worker order：

0,1,2,...,9

汇入 global replay。

保持 single-GPU source ordering semantics。

HER：

仍按照当前 frozen physical insertion behavior
执行。

不要：

每卡单独执行不同 HER threshold。

HER activation threshold
基于同一个：

global logical replay / source counter semantics。

==================================================
八、2-GPU Learner Batch
=======================

global replay：

只采样一次：

N = 256

不要：

GPU0 sample 256
GPU1 sample 256

否则 global effective batch 会变成 512。

正确流程：

global replay
      ↓
sample [256,...]
      ↓
split
      ↓
GPU0 [128,...]
GPU1 [128,...]

最终 algorithmic batch
仍然是：

256。

==================================================
九、Gradient Synchronization
============================

这是 Step8 核心。

每张 GPU：

只计算自己 local batch 的梯度贡献。

然后：

跨设备 aggregate

形成：

GLOBAL BATCH=256

对应的完整梯度。

必须确保数学上等价于：

single-GPU
一次 batch=256 update。

---

特别注意 gradient clipping
--------------------------

SOURCE-FAITHFUL 顺序必须是：

local gradient contributions
        ↓
GLOBAL aggregation
        ↓
global actor/critic gradient
        ↓
global L2 norm clipping
        ↓
optimizer update

禁止：

每 GPU 先 clip local gradient
        ↓
再 pmean

这会改变算法。

Critic：

Q1+Q2 complete critic gradient PyTree
全局同步后
clip_by_global_norm(100)。

Actor：

完整 actor gradient
全局同步后
clip_by_global_norm(100)。

然后：

所有 replicas
执行相同 optimizer update。

==================================================
十、Loss Reduction 必须保持 Global Mean
=======================================

single-GPU frozen loss 中：

Smooth-L1
和 angle/scalar terms
都具有明确的 mean reduction。

Multi-GPU 后：

不能简单把：

local_mean

不加检查地相加。

2 GPU balanced 模式：

因为：

128 / 128

local batch 相同，

可以使用数学等价的：

mean(local means)

但必须写测试验证。

建议实现时优先表达为：

global sum / global valid count

方便 Step8.2 3-GPU
复用同一个逻辑。

==================================================
十一、随机数语义
================

不要以 physical device ID
决定算法随机性。

至少区分：

A. rollout RNG

依赖：

logical worker ID
worker step/episode state

B. replay sample RNG

全局只有一个 replay-sampling RNG
生成 batch=256。

C. target-policy smoothing noise

必须保证每个有效 global sample
只获得一个 noise sample。

不要：

每张卡重新对完整 batch 采噪声。

---

Parity test
-----------

对于 dedicated distributed parity test：

允许显式固定：

sampled replay batch

和：

target smoothing noise

以隔离随机性，

比较：

single-GPU learner update

vs

2-GPU distributed learner update。

==================================================
十二、2-GPU Learner Parity Golden Test
======================================

构造：

同一 initial params

同一 optimizer states

同一 batch=256

同一 preferences

同一 target noise

同一 interpolator state

分别执行：

A.
single-GPU learner update

B.
2-GPU:
128 + 128
distributed update

比较：

critic loss components

actor loss components

global critic gradient

global actor gradient

updated critic params

updated actor params

updated target params

optimizer states。

必须证明：

distributed update
只存在合理的 floating-point reduction-order error。

不得修改：

Step6 frozen tolerances。

如果误差超出当前可接受数值范围：

输出：

STEP8 DISTRIBUTED PARITY MISMATCH

不要直接放宽 tolerance。

==================================================
十三、2-GPU Control Plane
=========================

Control plane：

不要每张 GPU 各运行一遍。

只能存在一个：

logical controller。

episode_count：

仍然是：

[10]

而不是：

[2,5]
或：

[2,10] 的算法语义。

可以物理 gather 后形成：

episode_count[10]

然后执行：

strict key trigger。

Key evaluation：

只触发一次。

Interpolator refit：

host-side SciPy
只执行一次。

完成 refit 后：

新的 fixed-shape JAX interpolator arrays

broadcast / replicate
到所有 learner devices。

增加 assertion：

all device interpolator replicas equal。

==================================================
十四、Full Evaluation
=====================

Step8 第一版：

不要分布式改写 evaluator。

继续使用 Step6.1 已冻结的：

PDMORLEvaluator。

可以指定：

controller / one GPU

执行 evaluation。

原因：

Step8 首要目标是
distributed training correctness，
不是 evaluation acceleration。

不要为了利用所有 GPU
重新设计 Pareto/HV/sparsity。

==================================================
十五、Checkpoint
================

只能生成一个：

logical global checkpoint。

不要：

每 GPU 各保存一份互相独立 checkpoint。

checkpoint 必须保持：

logical global state：

- actor；
- critics；
- targets；
- optimizer；
- replay；
- PRNG；
- episode_count[10]；
- worker states；
- eval_cnt；
- eval_cnt_ep；
- total_it；
- raw key solutions；
- interpolator arrays；
- control counters。

2-GPU save
→
2-GPU restore

必须 PASS。

如果当前 Orbax 能方便支持：

额外测试：

2-GPU save
→
1-GPU restore

这是推荐项，
不是第一版硬 blocker。

==================================================
十六、Step8.1 2-GPU Smoke
=========================

使用：

real Brax Walker

production：

Actor [400,400]
Q1 [400,400]
Q2 [400,400]

K=10
global batch=256
actor_loss_coeff=10
policy_freq=10
gamma=.995
tau=.005
HER=3
matmul_precision=highest

允许：

TEST OVERRIDE

仅缩短：

episode length
warmup
learner-start threshold
key/full trigger
evaluation grid/repeats
training duration。

禁止：

缩网络
缩 batch
改 K
改算法。

至少实际发生：

- multi-device rollout；
- global replay insertion；
- global batch sample；
- critic distributed update；
- delayed actor distributed update；
- target update；
- HER；
- key update；
- interpolator broadcast；
- evaluation；
- checkpoint save/restore。

==================================================
十七、Step8.1 PASS Gate
=======================

必须：

1. exactly 2 GPUs active；
2. 5 logical workers / GPU；
3. no padding；
4. global K=10；
5. global batch=256；
6. local batch=128；
7. one global replay；
8. one global replay sample/update；
9. global gradient synchronization correct；
10. clip occurs AFTER global aggregation；
11. replicas remain synchronized；
12. distributed learner parity PASS；
13. HER semantics unchanged；
14. key update only once；
15. interpolator refit only once；
16. interpolator broadcast to both GPUs；
17. evaluator semantics unchanged；
18. checkpoint/restore PASS；
19. no NaN/Inf；
20. no OOM；
21. frozen single-GPU regressions remain PASS。

只有 Step8.1 PASS 后：

才开始 3 GPU。

==================================================
十八、Step8.2：3-GPU Production Placement
=========================================

启用：

CUDA_VISIBLE_DEVICES=0,1,2

确认：

3 GPUs active。

因为：

K=10
不能被 3 整除，

不要改变 K。

推荐使用：

12 physical rollout slots

即：

4 slots / GPU。

mapping：

GPU0:
logical 0,1,2,3

GPU1:
logical 4,5,6,7

GPU2:
logical 8,9,PAD,PAD

定义：

valid_rollout_mask

例如：

[1,1,1,1,
 1,1,1,1,
 1,1,0,0]

physical slots = 12

effective logical workers = 10。

==================================================
十九、3-GPU Padding 绝不能变成 Logical Workers
==============================================

两个 PAD slots：

不能：

- 获得 preference subspace；
- 增加 episode_count；
- 写 replay；
- 执行 HER insertion；
- 影响 metrics；
- 触发 key update；
- 影响 evaluation；
- 增加 effective transition count。

即使 Brax 为 static shape
实际计算了 dummy env step：

输出也必须完全 mask 掉。

algorithmic K：

永远仍是：

10。

==================================================
二十、3-GPU Learner Batch
=========================

production global batch：

256

不能改成：

255
258
264。

由于 256 不可被 3 整除，

推荐 physical learner batch：

258

即：

86 rows / GPU。

其中：

256 valid
2 padded。

例如：

global validity mask：

first 256 = True
last 2 = False。

两个 PAD learner rows：

只用于 static equal device shape。

不能改变：

effective batch=256。

==================================================
二十一、3-GPU Masked Global Reduction
=====================================

因为三个 GPU 的有效 sample 数
可能是：

86
86
84

禁止：

mean(
    local_mean_gpu0,
    local_mean_gpu1,
    local_mean_gpu2
)

这会错误给三个 GPU
相同权重。

必须：

按有效样本计算 GLOBAL reduction。

例如：

actor scalar term：

global_sum(valid per-sample term)
/
256

angle mean：

global_sum(valid angle)
/
256

Smooth-L1：

global_sum(valid element losses)
/
(256 * num_objectives)

所有 padding rows：

贡献严格为 0。

gradient：

必须对应上述
global mean loss。

==================================================
二十二、3-GPU Padding 数据
==========================

不要让 padding：

从 replay 多采两条 transition。

replay 仍然：

只 sample 256。

然后：

在 sharding boundary
追加 2 dummy rows。

推荐：

复制有效样本作为 dummy
再设：

valid_mask=False

以避免：

zero-state
zero-preference

进入 cosine/angle
产生特殊 NaN 路径。

无论 dummy 值是什么：

masked contribution
必须严格为 0。

==================================================
二十三、3-GPU Gradient Clipping
===============================

流程仍必须：

local masked contributions
      ↓
psum / equivalent global aggregation
      ↓
divide by GLOBAL valid denominator
      ↓
GLOBAL gradient
      ↓
global norm clip 100
      ↓
optimizer update

不是：

local clip
→ global aggregate。

这一点必须单独 unit test。

==================================================
二十四、3-GPU Learner Parity
============================

使用与 Step8.1 相同：

initial params
batch=256
noise
interpolator state。

比较：

single-GPU batch256

vs

3-GPU:
physical 258
valid 256

更新结果。

必须比较：

loss
global gradients
params
targets
optimizer states。

另外验证：

改变两个 padding rows 的 dummy 数值

不能改变：

loss
gradient
updated params。

增加：

test_padding_rows_have_zero_effect

这是 3-GPU 最重要的测试之一。

==================================================
二十五、3-GPU Logical-Lane Test
===============================

验证：

logical IDs exactly:

0..9

一次且仅一次。

PAD：

不属于 logical ID。

验证：

preference subspace coverage
与 single-GPU 完全相同。

验证：

episode_count shape
仍为：

[10]

而不是：

[12]。

==================================================
二十六、3-GPU Replay Test
=========================

一个 rollout round：

physical env slots = 12

但：

replay base insert count
必须只对应：

10 valid logical workers。

HER 后的 write count
严格使用当前 source-faithful规则。

两个 padding lanes：

不能写入任何 replay entry。

==================================================
二十七、Control / Evaluation in 3 GPU
=====================================

仍然：

一个 logical controller。

不要每 GPU：

各自 key-evaluate/refit/evaluate。

control trigger：

只看：

10 real logical workers。

host refit 后：

broadcast interpolator state
到 3 个 replicas。

Pareto/HV/sparsity：

仍然使用冻结 evaluator。

==================================================
二十八、Multi-GPU PRNG Independence
===================================

必须保证：

logical worker 0

无论放在：

GPU0
GPU1
GPU2

它的 logical PRNG identity
都由：

worker ID

而非 device ID
决定。

增加 placement invariance test：

改变：

logical worker → GPU mapping

但保持 worker ID 和 global seed，

确认 preference assignment /
logical RNG stream
不会因为 device ordinal 改变。

至少验证：

worker identity semantics。

==================================================
二十九、Multi-GPU Logging
=========================

不要每个 GPU
重复写 recorder。

只有 controller
写：

morl/train/*
morl/control/*
morl/eval/*

新增 distributed diagnostics：

distributed/num_devices

distributed/logical_workers

distributed/physical_rollout_slots

distributed/valid_rollout_slots

distributed/global_batch

distributed/physical_learner_batch

distributed/valid_learner_samples

distributed/replica_param_max_diff

distributed/replica_interpolator_max_diff

distributed/gradient_sync_error

distributed/padding_fraction

distributed/jit_compile_time

distributed/steady_update_time

可选：

throughput
GPU memory。

==================================================
三十、性能不是 PASS 条件
========================

记录：

1 GPU steady update time
2 GPU steady update time
3 GPU steady update time

以及：

rollout throughput。

但是：

Step8 不要求一定线性加速。

如果 3 GPU
比 1 GPU 慢：

只要 correctness PASS，

不要修改算法
去追求 benchmark。

性能优化：

后续单独处理。

==================================================
三十一、Regression Strategy
===========================

每完成一个阶段：

重新跑 frozen single-GPU tests。

必须证明：

引入 distributed code
没有改变 single-GPU path。

至少：

Step6 Golden Tests
Step6.1
Step7
Step7.1
Step7.2 guards
scalar TD3。

single-device execution
仍使用原 frozen semantics。

==================================================
三十二、文件结构
================

尽量集中 multi-device 代码。

例如：

evorl/distributed/
或当前 EvoRL 已有 distributed utilities

以及：

PDMORLWorkflow 中最小 integration hooks。

不要把：

if num_devices > 1

散落到：

Actor
Critic
HER
interpolator
evaluation
所有 loss 函数中。

Learner math
应尽量保持 device-agnostic。

==================================================
三十三、创建 Tests
==================

至少增加：

test_pd_morl_two_gpu_balanced.py

覆盖：

- exactly 2 GPU
- lane mapping 5/5
- batch 128/128
- learner parity
- gradient synchronization
- post-aggregation clipping
- replica equality
- global replay
- control/refit
- checkpoint

以及：

test_pd_morl_three_gpu_masked.py

覆盖：

- exactly 3 GPU
- 12 physical rollout slots
- 10 valid logical workers
- PAD workers excluded
- physical learner batch 258
- valid batch 256
- masked global reductions
- padding-zero-effect
- distributed learner parity
- global replay
- interpolator broadcast
- checkpoint

标记：

gpu
multi_gpu
slow
integration

普通 single-GPU/CPU pytest：

不应运行它们。

==================================================
三十四、文档
============

创建：

docs/PD_MORL_MULTI_GPU.md

记录：

1. Scope
2. Frozen single-GPU baseline
3. EvoRL distributed audit
4. Multi-GPU architecture
5. Logical vs physical workers
6. Global replay semantics
7. Replicated parameter model
8. 2-GPU placement
9. 2-GPU learner parity
10. Gradient synchronization
11. Gradient clipping order
12. PRNG semantics
13. 3-GPU padded rollout
14. 3-GPU padded learner batch
15. Valid masks
16. Masked loss reduction
17. Control-plane ownership
18. Interpolator broadcast
19. Evaluation ownership
20. Checkpoint/restore
21. Single-GPU regression
22. Performance observations
23. SOURCE-FAITHFUL / FRAMEWORK-ADAPTATION classification
24. Remaining distributed optimization work

==================================================
三十五、Classification
======================

SOURCE-FAITHFUL：

- K=10 logical workers；
- preference subspaces；
- one global replay semantics；
- global effective batch=256；
- HER；
- learner math；
- update schedule；
- gradient clip threshold/semantic；
- control trigger；
- evaluator semantics。

FRAMEWORK-ADAPTATION：

- logical worker → GPU placement；
- replicated params；
- gradient all-reduce；
- 12 physical rollout slots for 3 GPU；
- valid masks；
- 258 physical learner rows / 256 valid；
- host controller；
- interpolator broadcast；
- sharded checkpoint placement。

TEST OVERRIDE：

只允许 runtime / trigger /
evaluation-cost shortening。

DEVIATION：

以下全部禁止：

- changing K to 9/12；
- changing batch to 255/258；
- independent replay per GPU；
- independent models per GPU；
- local-gradient clipping before sync；
- padding samples affecting loss；
- duplicate control/evaluation execution；
- changing network size；
- changing learner math。

==================================================
三十六、Step8 Final PASS Gate
=============================

STEP8 只有以下全部满足才 PASS：

A. 2 GPU

1. 2 devices active；
2. 5+5 logical lane mapping；
3. global K=10；
4. global batch=256；
5. local batch=128；
6. global replay；
7. learner parity PASS；
8. global gradient sync PASS；
9. clipping after global aggregation；
10. replicas synchronized；
11. control/refit single execution；
12. checkpoint PASS。

B. 3 GPU

13. 3 devices active；
14. 12 physical rollout slots；
15. exactly 10 valid logical workers；
16. padding slots never enter replay/counters/HER；
17. global effective batch=256；
18. physical batch=258；
19. exactly 256 valid learner rows；
20. masked reductions mathematically correct；
21. padding-zero-effect PASS；
22. distributed learner parity PASS；
23. clipping after global aggregation；
24. replicas synchronized；
25. interpolator broadcast synchronized；
26. control/evaluation single execution；
27. checkpoint PASS。

C. Regression

28. production [400,400] unchanged；
29. batch256 unchanged；
30. K=10 unchanged；
31. Step6 tolerance unchanged；
32. single-GPU frozen regression PASS；
33. no NaN/Inf；
34. no OOM；
35. no algorithm DEVIATION introduced。

全部通过后输出：

STEP8 MULTI-GPU EVORL PD-MORL PASS

并停止。

不要开始：

- paper 1M-step reproduction；
- multi-node；
- distributed replay optimization；
- PSL-MORL；
- hypernetwork。

如果任一 distributed parity
或 masked reduction 失败：

输出：

STEP8 MULTI-GPU MISMATCH REPORT

包含：

- device topology；
- logical/physical shapes；
- loss error；
- gradient error；
- parameter error；
- replay counts；
- mask statistics；
- suspected cause。

不要通过修改算法
或放宽 Step6 tolerance
强行 PASS。
