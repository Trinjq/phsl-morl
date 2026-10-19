> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


请继续处理当前 EvoRL 项目中的 PD-MORL GPU-native v2。

这次任务不要再进行新的大规模架构重构，不要创建“v3”训练框架。当前 v2 的主体结构先冻结。

当前 v2 已经具备：

- 10 个 logical preference groups；
- 160 个 Brax physical environment lanes；
- logical-group warm-up；
- global HER threshold；
- replay batch size 4096；
- critic_updates_per_rollout = K；
- K 次 Critic update 位于 JAX lax.scan 内；
- policy delay 基于 global critic optimizer step；
- batched evaluator；
- horizon=500；
- final checkpoint；
- Pareto JSON/NPZ；
- 1M seed42 已完整运行。

当前任务的目标是：

1. 核对 evaluation 是否真正与官方 PD-MORL benchmark 语义一致；
2. 审计并修正 key evaluation / RBF interpolator 在多 lane 下的控制语义；
3. 补齐 v2 实际运行计数；
4. 完成 K=5/10/20 learner benchmark；
5. 完成 80/160 env 的 time-to-quality benchmark；
6. 建立同 Brax 环境下的 faithful-style/reference baseline；
7. 最终判断 GPU-native 是否真正带来 time-to-quality 加速。

不要切换 MuJoCo、MJX、Gymnasium 或 PyTorch。
继续使用 JAX + Brax + GPU。

---

一、先核对官方 evaluation 聚合语义(论文对应源码位置"E:\PD-MORL\PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm")
------------------------------------------------------------------------------------------------------------------------------------------

这是当前第一优先级。

当前 v2 evaluator 是：

每个 repeat：
    preference returns
    -> non-dominated filtering
    -> HV
    -> sparsity

这个逻辑可以保留作为 diagnostic metric，
但请重新检查官方 PD-MORL：

- lib/utilities/MORL_utils.py
- eval_benchmarks_MO_TD3_HER.py
- train_Walker2d_MO_TD3_HER.py

重点确认官方最终 benchmark 到底是：

A.
每个 repeat 单独生成 Pareto front 后分别算 HV/sparsity

还是：

B.
先对同一个 preference 的多个 repeat 求 mean return
然后：
mean return per preference
-> non-dominated filtering
-> Pareto front
-> HV / sparsity

请不要根据当前实现推断，必须直接根据官方源码确认。

如果官方 benchmark 是 B，则正式/reference-compatible 指标必须改成：

returns:
[R, P, L]

先：

mean_returns = mean over repeat axis

得到：

[P, L]

然后：

mean_returns
-> non-dominated filtering
-> source-compatible Pareto front
-> HV
-> sparsity
-> Pareto point count

同时保留当前 per-repeat Pareto/HV/sparsity，作为 diagnostic metrics。

最终结果请明确区分：

source_hv
source_sparsity
source_pareto_point_count

和：

repeat_hv
repeat_sparsity
repeat_pareto_point_count

不要混在同一个字段中。

确认官方源码中以下行为：

- hypervolume reference point；
- maximization/minimization方向；
- sparsity 排序方式；
- non-dominated filtering；
- duplicate point 是否去重；
- Pareto 点数 <=1 时 sparsity 的行为；
- mean 是先按 episode/repeat 聚合还是先做 Pareto；
- evaluation preference grid 顺序。

新增 golden tests：

1. 手工构造 2D returns，确认 source-compatible 聚合顺序；
2. serial evaluator 和 batched evaluator 的 source metrics 一致；
3. repeat diagnostic metrics 与 source metrics 字段互不混淆；
4. dominated point 不进入最终 Pareto；
5. 单 Pareto 点时 sparsity 行为和官方源码一致。

不要继续使用未经核对的 evaluation 结果作为后续 benchmark 的主指标。

---

二、审计 key evaluation / RBF interpolator 在多 lane 下的控制语义
-----------------------------------------------------------------

这是第二优先级，也是当前最重要的剩余算法语义问题。

原始 PD-MORL：

1 logical worker
================

1 environment timeline
======================

1 episode counter

当前 GPU-native：

1 logical preference group
==========================

16 physical environment lanes
=============================

16 asynchronous episode timelines

请完整追踪当前 v2：

- group episode_count 如何计算；
- key evaluation trigger 如何判断；
- eval_cnt_ep 如何推进；
- 16 个 lane 的 episode completion 如何映射成一个 logical worker episode count；
- key evaluation 是否可能在一个 chunk 内被延迟；
- RBF refit 什么时候真正生效；
- fold_iters=4 是否导致 interpolator 更新滞后；
- 同一个 chunk 中 rollout 2/3/4 是否可能继续使用旧 interpolator。

首先只做审计，不要立即修改。

输出：

docs/PD_MORL_GPU_NATIVE_V2_CONTROL_AUDIT.md

至少包含：

1. 原始 CPU source 的控制流程；
2. 当前 GPU-native 控制流程；
3. group episode counter 的数学定义；
4. key evaluation trigger 的数学定义；
5. 最大可能 trigger delay；
6. 实际 1M seed42 中：
   - key evaluation count
   - RBF refit count
   - key solution replacement count
   - 平均 trigger delay
   - 最大 trigger delay
7. 当前语义属于：
   - source-faithful
   - framework adaptation
   - algorithmic deviation

不要默认“平均 episode count”或“总 episode/lanes”一定合理。

如果当前语义明显有问题，再提出最小修复方案。

目标不是恢复 CPU 事件级顺序，而是定义一个稳定、可解释、不会随 envs_per_group 无意改变的 logical-group control rule。

优先考虑让 key-control semantics 只依赖 logical group，而不是 physical lane 数。

任何修改必须增加：

envs_per_group = 1 / 4 / 16

下的 control trigger tests。

---

三、补齐 1M v2 的实际运行计数
-----------------------------

当前文档只记录了部分结果。

请从：

outputs/gpu_native_v2_1m_seed42

的日志、metadata、checkpoint 和 evaluation artifacts 中提取或重新计算以下实际值：

- total global base transitions
- global random transitions
- global policy-driven transitions
- random_action_fraction
- 每个 logical group:
  - random transitions
  - policy transitions
  - group transition count
  - episode count
- HER actual activation transition
- actual critic optimizer update count
- actual actor update count
- actual target update count
- actor_update_count / critic_update_count
- replay size progression
- key evaluation count
- key solution replacement count
- RBF refit count
- final replay size
- final workflow iteration
- wall-clock total
- JAX compile time
- steady-state training time
- evaluation total time
- learner total time
- rollout total time
- peak GPU memory
- average/representative GPU utilization if available

不要通过配置理论推算代替实际计数。

如果某个值当前没有记录：

1. 在 workflow 中增加计数器；
2. 在下一次 smoke/benchmark 中验证；
3. 不要伪造旧 1M run 的值。

将实际已有值与“无法从旧 run 恢复”的值分开标记。

更新：

docs/PD_MORL_GPU_NATIVE_V2_REPORT.md

增加：

“Actual runtime counters”

章节。

---

四、正式执行 K=5 / 10 / 20 benchmark
------------------------------------

当前 v2 已定义：

replay_batch_size = 4096

测试：

K = 5
K = 10
K = 20

当前配置：

num_envs = 160
rollout_length = 4

因此：

new transitions per rollout = 640

data reuse：

K=5:
5*4096/640 = 32

K=10:
64

K=20:
128

先不要增加 K=40。
只有 K=20 仍然表现出明显 learning-quality 改善，且 learner 仍未明显成为瓶颈时，才把 K=40 作为额外上限实验。

重要：

不要只跑纯吞吐 microbenchmark。

每个 K 必须同时做：

A. throughput benchmark
B. short learning benchmark

A 至少记录：

- transitions/s
- critic updates/s
- actor updates/s
- replay samples/s
- learner wall time
- rollout wall time
- GPU utilization
- peak GPU memory
- compile time

B 使用完全相同：

- Brax environment
- seed
- transition budget
- evaluation schedule
- warm-up
- HER
- preference grid

建议短学习预算：

200k 或 300k global base transitions

不要完整跑 1M × 3，除非运行时间足够低。

对每个 K 保存：

- HV vs environment transitions
- HV vs critic optimizer steps
- HV vs wall-clock
- Pareto point count vs wall-clock
- objective ranges
- sparsity
- best HV
- final HV

最终选择 K 的规则：

time-to-quality 优先。

不能仅根据：

transitions/s

或：

updates/s

选择。

如果不同 K 在短预算下还无法区分，则不要武断宣布 K=10 最优，保留多个候选进入下一阶段。

---

五、重新 benchmark 80 vs 160 environments
-----------------------------------------

当前 160 environments 的优势来自旧的 sampler-heavy 配置。

现在：

batch=4096
K>1

learner 开销已经发生变化。

因此确定一个合理 K 后，再比较：

envs_per_group = 8
num_envs = 80

和：

envs_per_group = 16
num_envs = 160

可选：

envs_per_group = 4
num_envs = 40

但不要跑完整笛卡尔积。

保持：

- 同 K
- 同 batch size
- 同 global transition budget
- 同 random warm-up total semantics
- 同 HER threshold
- 同 evaluation
- 同 seed

比较：

- transitions/s
- learner time
- rollout time
- total training time
- GPU utilization
- peak memory
- HV vs wall-clock
- Pareto count vs wall-clock

最终判断：

160 env 是否真的带来更好的 time-to-quality。

不能继续使用旧的：

149.91 transitions/s

直接作为“160 env 最优”的证据。

---

六、不要误把 lax.scan(K) 描述成并行 optimizer updates
-----------------------------------------------------

保留当前实现：

jax.lax.scan

是合理的。

但是更新文档表述：

K 次 Critic optimizer step 存在参数依赖：

theta_0
-> update1
-> theta_1
-> update2
-> theta_2
...

因此 lax.scan 的作用是：

- 把顺序更新放进一个编译后的 JAX loop；
- 减少 Python dispatch；
- 让单次大 batch gradient computation 在 GPU 上高效运行。

不要写成：

“K 次 optimizer updates 被并行执行”。

不要尝试直接用 vmap 并行 K 次带参数依赖的 optimizer step。

如需要进一步加速 learner：

先 profile，
再考虑：

- batch size；
- network compute；
- replay sampling；
- XLA fusion；
- buffer donation；
- compilation；
- data layout。

不要改变 optimizer dependency semantics。

---

七、建立同 Brax 的 reference baseline
-------------------------------------

这是后续证明 GPU-native 有效的关键。

新增一个：

pd_morl_brax_reference.yaml

目标：

使用完全相同的：

- Brax Walker adapter
- reward definition
- horizon = 500
- network architecture
- HER
- RBF/key solutions
- preference grid
- evaluation code
- transition budget
- seed

但是训练调度尽量接近原 PD-MORL：

- 10 logical preference workers/groups
- envs_per_group = 1
- num_envs = 10
- rollout_length 尽量接近 source-faithful
- replay batch = 256
- learner schedule 接近原源码
- policy_freq = 10
- global replay
- source-compatible warm-up
- source-compatible HER threshold

注意：

这个 reference 仍然使用 JAX + Brax。

它不是论文 MuJoCo 数值复现。

它的作用是：

在同一个环境里，
把：

source-like scheduling

和：

GPU-native scheduling

进行公平比较。

暂时先：

1. 建配置；
2. 建 workflow 入口；
3. 通过 unit tests；
4. 做 smoke test。

不要立刻跑完整 10M。

---

八、统一后续比较协议
--------------------

以后正式比较：

Brax reference
vs
GPU-native

必须统一：

- Brax environment
- reward
- horizon
- seed
- total global base transition budget
- evaluation preference grid
- evaluation repeats
- source-compatible metric aggregation
- checkpoint policy
- random warm-up semantics
- HER semantics

主图至少准备：

1. HV vs environment transitions
2. HV vs wall-clock
3. Pareto point count vs wall-clock
4. sparsity vs wall-clock

辅助图：

5. HV vs critic optimizer steps
6. transitions/s
7. replay samples/s
8. GPU utilization
9. peak GPU memory

最终 GPU-native 是否成功，主要根据：

time-to-quality

判断。

例如：

达到相同 HV：
GPU-native 需要多少分钟
reference 需要多少分钟

而不是：

谁的 transitions/s 更高。

---

九、暂时不要直接与论文 Table 3 的绝对 HV 做成功判断
---------------------------------------------------

继续保留：

Brax != MuJoCo

这一实验边界。

论文 Table 3 可以：

- 作为算法背景；
- 作为 PD-MORL 原始表现参考。

但不要把：

Brax GPU-native HV

直接除以：

MuJoCo paper HV

然后得出“复现程度百分比”。

正式 GPU-native 结论必须来自：

同 Brax reference 对照。

---

十、当前 1M v2 不要重复跑多 seed
--------------------------------

在完成以下事项前：

- source-compatible evaluation 聚合；
- key/RBF control audit；
- K benchmark；
- env-count benchmark；
- runtime counter补齐；

不要马上重复运行 3/6 个 seed。

当前 seed42 已经说明：

- 数据流能运行；
- warm-up 已修；
- learner 能学习；
- Pareto front 不再明显坍缩；
- checkpoint/evaluation artifacts 可以生成。

下一步需要解决的是：

“什么 GPU-native 配置最合理”

而不是：

“同一个未完全验证的配置多跑几遍”。

---

十一、最终交付物
----------------

本任务完成后至少提供：

docs/PD_MORL_GPU_NATIVE_V2_CONTROL_AUDIT.md

更新后的：
docs/PD_MORL_GPU_NATIVE_V2_REPORT.md

evaluation source-compatible implementation + tests

K benchmark JSON/CSV

80/160 env benchmark JSON/CSV

pd_morl_brax_reference.yaml

reference smoke test report

runtime counters artifact

一个简洁总结：

docs/PD_MORL_GPU_NATIVE_V2_NEXT_DECISION.md

该文件只回答：

- evaluation 是否已与 source benchmark 对齐；
- key/RBF control semantics 是否合理；
- K=5/10/20 哪些候选仍值得保留；
- 80/160 env 哪个更合理；
- 当前 GPU-native 的主要瓶颈是 sampler 还是 learner；
- 是否已经具备进行 Brax reference vs GPU-native 正式对照的条件。

---

十二、实施顺序
--------------

严格按以下顺序执行：

A. 核对官方 evaluation 源码
B. 修 source-compatible metrics
C. 跑 evaluator tests
D. 做 key/RBF control audit
E. 增加缺失 runtime counters
F. 跑小 smoke 验证 counters
G. 跑 K=5/10/20 benchmark
H. 选择合理 K 候选
I. 跑 80/160 env benchmark
J. 建立 Brax reference config/workflow
K. reference smoke test
L. 更新所有报告

任何阶段失败：

先修问题，
不要继续后面的长实验。

---

十三、禁止事项
--------------

本阶段不要：

- 切换到 MuJoCo；
- 切换到 MJX；
- 增加多 GPU；
- 重写 Actor/Critic 网络；
- 删除 HER；
- 修改 directional-angle 数学；
- 修改 vector target 选择规则；
- 改 RBF 数学；
- 新增第二套 replay；
- 引入梯度累积框架；
- 用 vmap 并行有参数依赖的 K 次 optimizer update；
- 删除 v1/v2 历史实验；
- 覆盖旧 checkpoint；
- 为通过测试而降低 assertion/tolerance；
- 在没有 source 证据时自行猜测 evaluation 行为。

---

十四、完成标准
--------------

只有满足以下条件才认为本任务完成：

- source-compatible evaluation 聚合已明确；
- evaluator golden tests 通过；
- key/RBF logical-group control semantics 已审计；
- trigger delay 已量化；
- v2 实际 critic/actor/random/policy/HER/key counters 已记录；
- K=5/10/20 benchmark 已完成；
- 80/160 env benchmark 已完成；
- time-to-quality 数据已经产生；
- Brax reference 配置和入口已建立；
- reference smoke test 通过；
- faithful 现有 regression tests 没有被破坏；
- 所有结果已经写入文档和 artifact，而不是只存在终端输出中。

先审计，再修改，再测试，再 benchmark。
不要跳步骤。
