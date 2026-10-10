> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在开始 Step4.1：Vector Reward Data Path。

本阶段是 Step4 的第一项最小 MORL 工程改造。

只解决：

scalar reward
r ∈ R

→

vector reward
r ∈ R^L

当前首个任务已经在 Step3 中冻结为：

PD-MORL:
MO-Walker2d-v2

EvoRL/Brax:
walker2d

目标数固定：

L = 2

==================================================
一、本阶段严格边界
==================

本阶段只实现 vector reward 数据链。

不要：

- 加入 preference w；
- 修改 Actor；
- 修改 Critic；
- 实现 vector Q；
- 实现 scalarization；
- 修改 TD target；
- 实现 HER；
- 实现 interpolator；
- 实现 angle loss；
- 实现 Pareto/HV/sparsity；
- 修改 PD-MORL learner 更新逻辑。

完成 vector reward 数据链并通过测试后立即停止。

==================================================
二、reward 定义禁止自行设计
===========================

严格使用 Step3 已冻结的官方 Walker2d 两目标 reward：

r1 = x_velocity + 1.0

r2 = 5.0 - sum(clip(action,-1,1)^2)

即：

reward = stack(
    [
        x_velocity + 1.0,
        5.0 - sum(clip(action,-1,1)^2)
    ],
    axis=-1
)

最终单环境 reward shape：

[2]

batched environment reward shape：

[B,2]

其中：

objective 0:
官方 forward-speed objective

objective 1:
官方 energy-efficiency objective

禁止：

- 直接复用 Brax 默认 scalar reward；
- 使用 Brax 默认 control-cost coefficient 0.001；
- 修改 +1、5.0 或 action-square reduction；
- 自行增加 reward normalization；
- 自行增加 reward clipping；
- 自行改变 objective 顺序。

reward 必须在 JAX/device 环境路径内计算。

禁止将 state/action 转回 Python/NumPy host 后计算 reward。

==================================================
三、首先做影响面检查
====================

在修改代码前，检查 vector reward 从环境产生后经过的完整数据链：

Brax environment
→ EvoRL environment adapter/wrapper
→ rollout / env_step
→ SampleBatch
→ replay dummy/spec
→ replay add
→ replay sample
→ 当前 logging / recorder

列出：

1. 哪些代码明确假定 reward 是 scalar；
2. 哪些代码其实已经可以透明支持 trailing objective axis；
3. 哪些地方必须修改；
4. 哪些地方不应该修改。

不要因为 reward 变成 vector 就机械修改所有文件。

==================================================
四、兼容原则
============

必须保证 EvoRL 原来的单目标算法不被破坏。

单目标环境仍保持：

reward shape:
[...]

MORL Walker 环境：

reward shape:
[...,2]

优先让现有 generic data structures 同时支持：

scalar reward
和
fixed-tail vector reward

不要修改原 TD3 数学逻辑。

不要把所有单目标环境强制改成：

[...,1]

除非现有框架结构证明这是必须的；如果认为必须这样做，先报告原因，不要直接实施。

==================================================
五、SampleBatch
===============

优先复用现有 SampleBatch。

目标：

single-objective:

rewards: [N]

multi-objective:

rewards: [N,2]

不要默认新增 MORLSampleBatch 或其他平行 batch 类型。

只有在证明现有 SampleBatch 无法安全表达 vector reward 时：

停止，
说明具体类型/shape/PyTree限制，
不要自行建立第二套数据模型。

==================================================
六、Replay Buffer
=================

检查现有 generic replay buffer 是否已经能够存储：

rewards:[N,2]

如果 ring buffer 对固定 PyTree leaf shape 是透明的，则复用现有实现。

只修改：

dummy batch
schema
shape inference

等真正需要修改的位置。

禁止为了 MORL 重写 replay buffer。

本阶段尚不实现 preference，也不实现 HER。

==================================================
七、环境实现
============

为 Brax Walker2d 增加最小的多目标 reward 承载方式。

要求：

observation:
保持原 Brax Walker observation

action:
保持 [B,6]

physics:
保持 Brax Walker2d

termination:
本阶段不要修改 TD bootstrap 逻辑。

reward:
替换/扩展为：

[B,2]

确保 vector reward 在：

jax.jit

和 Brax batched execution

中保持 device array。

不要引入：

Gymnasium
MuJoCo Python training env
PyTorch
host callback
NumPy reward computation

==================================================
八、metrics / logging
=====================

本阶段不要实现正式 MORL evaluation。

只确保 vector reward 不会导致现有 rollout / recorder / logging 崩溃。

如需要记录 reward，可临时记录：

reward/objective_0
reward/objective_1

或 vector reward shape/debug statistics。

不要：

- scalar sum 两个 objectives 后冒充 episode return；
- 提前实现 HV；
- 提前实现 Pareto front；
- 提前实现 sparsity。

如果现有 logger 强制要求 scalar reward，明确记录这一点，并做最小兼容修改。

==================================================
九、测试
========

必须包含以下测试。

Test 1：单 transition

验证：

reward.shape == (2,)

reward 数值严格满足：

r0 = x_velocity + 1

r1 = 5 - sum(action^2)

==================================================

Test 2：batched environment

设 B > 1。

验证：

reward.shape == (B,2)

不同环境 lane 的 reward 独立。

优先测试 Brax 原生 batch 语义。

不要为了满足测试要求机械添加 jax.vmap。

==================================================

Test 3：rollout

执行短 rollout。

验证：

若：

T = rollout length

B = env batch

则：

rewards.shape == (T,B,2)

==================================================

Test 4：Replay add/sample

向现有 replay 写入 vector-reward transitions。

验证：

写入：

[N,2]

采样：

[batch_size,2]

objective axis 不丢失、不 flatten、不交换。

==================================================

Test 5：JIT

对涉及 reward 生成和必要数据路径执行 jax.jit。

验证：

- 可以正常编译；
- reward shape 正确；
- 无 host callback；
- 无 tracer error；
- reward finite。

==================================================

Test 6：单目标回归测试

至少运行现有一个原始 TD3 scalar-reward 测试或最小 smoke test。

确认：

原单目标 reward shape 仍为 scalar trailing shape，
TD3 原路径不受影响。

==================================================

Test 7：GPU execution

运行前记录：

jax.default_backend()
jax.devices()

必须确认：

jax.default_backend() == "gpu"

至少看到目标 CudaDevice。

禁止：

JAX_PLATFORMS=cpu

smoke test 必须实际运行在 GPU backend。

==================================================
十、Smoke Test
==============

可以使用 synthetic/toy batch 做快速 unit test。

但是最终 smoke test 必须使用：

Brax walker2d

验证至少完成：

reset
→ action
→ step
→ vector reward
→ rollout
→ SampleBatch
→ replay add
→ replay sample

完整数据链。

无需训练 Actor/Critic。

==================================================
十一、输出文档
==============

创建：

docs/MORL_VECTOR_REWARD.md

必须记录：

1. 修改前 reward shape；
2. 修改后 reward shape；
3. Walker2d 两项目标的精确公式；
4. 修改文件；
5. 每个修改的必要性；
6. 未修改文件及原因；
7. SampleBatch 兼容方式；
8. replay 兼容方式；
9. 单目标兼容策略；
10. JIT/GPU 验证结果；
11. 所有测试命令；
12. 所有测试结果；
13. 是否存在 DEVIATION / BLOCKED。

==================================================
十二、完成条件
==============

只有以下全部成立才算 Step4.1 PASS：

- Walker vector reward 数值公式正确；
- reward shape 为 [B,2]；
- rollout 保持 objective axis；
- SampleBatch 保持 objective axis；
- replay add/sample 保持 objective axis；
- JIT 正常；
- GPU backend 已确认；
- 原单目标 TD3 数据链没有被破坏；
- 没有加入 preference；
- 没有修改 Actor；
- 没有修改 Critic；
- 没有实现后续 PD-MORL 模块。

完成后输出：

STEP4.1 PASS / FAIL

如果 FAIL：
说明失败点及阻塞原因。

完成后停止。

不要继续 Step4.2。
