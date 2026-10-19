> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入 Step 5.3：

PD-MORL Parallel Exploration Integration

重要：

这一步不是重新实现 preference sampling。

Step4.3 已经完成并冻结：

- official preference grid；
- preference subspaces；
- process_count 参数化；
- worker → subspace 绑定；
- episode preference sampling；
- episode 内 preference 固定；
- done lane 单独重新采 preference；
- preference rollout / replay 数据链。

Step5.2 已经完成并冻结：

- source-faithful add-time HER；
- HER warm-up；
- HER preference sampler；
- physical relabeled replay entries；
- logical global replay semantics。

本阶段只负责：

把这些已经完成的组件，
真正接成 PD-MORL 的 parallel exploration training flow。

不要重复实现 Step4.3。

==================================================
一、先做 SOURCE AUDIT
=====================

先审查官方：

PD-MORL/train_Walker2d_MO_TD3_HER.py

重点查看：

- child_process；
- process_count；
- train_queue_list；
- 主训练循环；
- replay_buffer_main.populate；
- learner update loop。

同时审查当前 EvoRL：

- EpisodePreferenceWrapper；
- Brax env factory；
- rollout；
- MO-TD3 workflow；
- replay add；
- Step5.2 HER integration。

先明确回答：

1. 官方每个 child process 一次向 main process 提交什么；
2. main process 一轮如何遍历 process_count 个 workers；
3. 一轮实际收集多少 base transitions；
4. 收集后如何写入 global replay；
5. HER 在这个过程中何时生效；
6. 一轮之后执行多少 learner updates；
7. update 次数是否依赖 process_count；
8. actor/critic 参数如何被 child workers 使用；
9. worker 是否拥有独立环境状态和随机状态；
10. main process 是否只有一个 replay pool。

先输出：

SOURCE AUDIT

不要立即修改代码。

==================================================
二、本阶段真正要实现的逻辑
==========================

定义：

K = process_count

默认：

K = 10

严格 source-faithful baseline 使用：

B = num_envs = K

因此：

K 个 logical workers
====================

K 个 Brax environment lanes

当前 Walker 默认：

K = 10
B = 10

每个 lane 已经由 Step4.3 正确绑定：

env_lane_i
→ logical_worker_i
→ preference_subspace_i

本阶段不要重新实现这个映射。

==================================================
三、Parallel Collection
=======================

使用 EvoRL / Brax 原生 batched environments
替代官方 Python multiprocessing。

官方：

K child processes
×
K envs

JAX：

一个 batched Brax env
batch dimension = K

要求：

每一个 collection step：

K 个 logical workers
各产生自己的 transition。

transition 必须继续携带：

observation
action
vector reward
next observation
termination / truncation
preference

preference 必须来自 Step4.3 已冻结 lifecycle。

不要在这里重新采 preference。

==================================================
四、参数共享语义
================

官方所有 child workers 使用同一个训练中的
actor / critic 参数体系。

JAX 版本必须保持：

所有 K 个 environment lanes
使用同一套当前 actor parameters。

禁止：

- per-worker actor；
- per-subspace actor；
- per-env independent model；
- 每个 worker 独立训练一套网络。

parallel exploration 只并行采数据，
不是训练 K 套 policy。

==================================================
五、Global Replay Integration
=============================

一轮 parallel collection 得到：

K 个 base transitions

这些 transition 必须进入：

同一个 logical global replay pool。

数据流应为：

K parallel base transitions
        ↓
Step5.2 source-faithful add-time HER
        ↓
original + possible relabeled entries
        ↓
one global replay pool

禁止创建：

- per-worker replay；
- per-subspace replay；
- per-env replay；
- per-GPU isolated replay。

不要修改 Step5.2 HER 实现。

==================================================
六、HER 联调
============

验证 Step5.2 在 parallel collection 下仍然正确。

例如 K=10：

parallel env step
→ 10 base transitions

HER warm-up 前：

每个 base transition
只产生 original entry。

HER active 后：

每个 base transition
按 Step5.2 规则最多产生：

1 original + N_w relabeled entries。

Walker 默认：

N_w = 3

因此一轮 10 个 base transitions
在 HER fully active 时最多产生：

10 × 4 = 40 replay entries。

注意：

这里只验证集成结果。

不要重新实现 HER sampler、
warm-up 或 ring-buffer semantics。

==================================================
七、Collection / Update Scheduling
==================================

这是本阶段最重要的新内容。

严格审查官方 main loop。

确认：

每一轮 parallel collection 后：

- 收集多少 base transitions；
- 做多少 critic learner updates；
- actor delayed update 如何由已有 policy_freq 决定。

JAX workflow 必须保持与官方等价的：

| base environment transitions |
| :--------------------------: |
|       learner updates       |

比例。

不要因为 Brax batching：

一次得到 K 条 transition
却只做 1 次 update，

除非官方源码本来就是如此。

也不要为了 GPU 吞吐：

擅自做更多 update。

Walker 默认行为必须以官方源码为准。

把最终确认的关系写入文档，例如：

collection round:
K base transitions
→ U learner updates

其中 K、U 都必须来自源码审查，
不要凭猜测填写。

==================================================
八、process_count 参数化
========================

继续沿用当前：

process_count

它是动态配置参数。

默认：

10

但测试还应覆盖：

K = 4
K = 8
K = 10

只检查 orchestration 是否随 K 正确变化。

不要重新测试：

- preference grid 数值；
- np.array_split 内容；
- subspace sampling 正确性；

这些属于 Step4.3。

这里只验证：

K 个 workers
→ K 条并行 collection lanes
→ 对应数量的 base transitions
→ 正确的 global replay 写入
→ 正确的 learner scheduling。

==================================================
九、严格 baseline 使用 B = K
============================

Step5.3 默认：

num_envs == process_count

即：

B = K

Walker：

B = 10
K = 10

现有 Step4.3 如果支持：

B > K

例如：

lane_id % K

不要删除该能力。

但：

B > K

暂时只标记为：

FRAMEWORK-ADAPTATION / PERFORMANCE MODE

本阶段不要用它作为 source-faithful baseline。

也不要研究：

K=10, B=100

对吞吐量的优化。

==================================================
十、测试重点
============

本阶段不要重复 Step4.3 已有测试。

不要重新重点测试：

- grid；
- subspace 内容；
- sampler 是否越界；
- episode lifecycle；
- union 是否覆盖 simplex。

这些已经 PASS。

新增测试应聚焦 training integration。

至少增加：

1. parallel collection count

K=10

一次 collection step 应产生
10 个有效 base transitions。

---

2. worker identity preserved

构造可辨识的 worker/lane metadata。

确认同一轮 collection 中：

worker IDs = 0...K-1

每个 worker 正好贡献一个有效 base transition
（如果官方源码审查确认就是这一语义）。

---

3. shared policy

所有 lanes 使用同一 actor parameter PyTree。

确认没有生成 per-worker model state。

---

4. global replay

一轮 parallel collection 后：

所有 workers 的 transitions
进入同一个 replay state。

能够从同一个 replay sample
看到来自多个 worker 的数据。

---

5. HER integration

HER active 时：

K 个 base transitions

产生源码等价的：

K × (1 + N_w)

最大 entry 数。

Walker：

K=10
N_w=3

→ 40 entries

同时确认每组 relabel
仍然只修改 preference。

---

6. warm-up integration

HER threshold 附近执行
parallel collection。

确认 Step5.2 已验证的逐 base-transition
strict `>` 语义没有被 batch orchestration 破坏。

---

7. collection/update ratio

构造小型配置：

K = 小值
U = 源码对应 update 数

明确计数：

environment base transitions
learner updates
critic updates
actor updates

确认与官方调度关系一致。

---

8. JIT

parallel collection
+
global replay add
+
已有 HER integration

能够在当前 JAX workflow 下执行。

不要为了测试额外 vmap
已经 batched 的 Brax env。

---

9. GPU smoke test

使用：

K = 10
B = 10

在单 GPU 上运行若干：

collect
→ replay add / HER
→ learner update

循环。

确认：

- backend == gpu；
- loss finite；
- replay size 正常增长；
- actor/critic parameter update 正常；
- preference data 没有丢失。

==================================================
十一、暂时不要实现 multi-GPU placement
======================================

本阶段只做：

single-GPU logical parallel exploration

即：

一张 GPU
承载 K=10 个 logical workers / env lanes。

不要在这一阶段实现：

- 3-GPU 4/3/3 worker placement；
- padded 12 slots；
- cross-device replay sharding；
- NamedSharding 重构；
- pmap/pjit 新架构。

但文档中明确记录：

process_count
与
device_count

完全独立。

multi-GPU placement 留到后续单独步骤。

==================================================
十二、不要修改已冻结模块
========================

禁止修改 Step4.3 的核心算法语义：

- official grid；
- subspace subdivision；
- worker sampler；
- episode lifecycle。

禁止修改 Step5.2：

- HER sampler；
- HER warm-up；
- add-time relabel；
- ring-buffer behavior。

禁止修改：

- MO-TD3 Actor/Critic；
- vector TD target；
- Smooth-L1 critic loss；
- actor objective；
- MORL math；
- interpolator；
- projected preference；
- directional angle；
- evaluation；
- Pareto metrics；
- hypernetwork。

==================================================
十三、文档
==========

创建：

docs/PD_MORL_PARALLEL_PREFERENCE.md

文档重点不要重复 Step4.3。

分成：

1. Already implemented in Step4.3

简短列出：

- preference grid；
- subdivision；
- worker/subspace binding；
- lifecycle。

并明确：

本阶段没有重新实现。

2. PyTorch execution model

记录：

- child processes；
- main process；
- queues；
- shared policy；
- collection round；
- learner update scheduling。

3. JAX/EvoRL replacement

记录：

PyTorch multiprocessing
→
Brax batched env

child process
→
logical worker / env lane

main-process aggregation
→
batched transition aggregation

single global replay
→
existing EvoRL global replay

4. Collection/update ratio

准确记录官方源码与 JAX 当前实现。

5. HER integration

记录：

parallel collection
→ add-time HER
→ global replay。

6. Classification

明确：

SOURCE-FAITHFUL:

- logical parallel workers；
- shared policy；
- global replay；
- collection/update semantics。

FRAMEWORK-ADAPTATION:

- Python processes → Brax batch；
- queue → array batch；
- multiprocessing RNG → JAX PRNG。

DEVIATION:
如果没有，则写 None。

==================================================
十四、PASS 条件
===============

只有以下全部满足才 PASS：

- 没有重新实现 Step4.3；
- K 个 logical workers 真正参与 parallel collection；
- B=K strict baseline 可运行；
- 所有 workers 共享同一 policy；
- 每轮 base-transition 数与源码一致；
- 所有数据进入同一 global replay；
- Step5.2 HER 在 parallel collection 下正确；
- collection/update ratio 对齐官方源码；
- JIT 正常；
- 单 GPU Walker smoke test 正常；
- Step4.3 regression 全部通过；
- Step5.2 regression 全部通过；
- 没有加入 interpolator / angle / evaluation；
- 没有加入新的 multi-GPU architecture。

完成后停止。
