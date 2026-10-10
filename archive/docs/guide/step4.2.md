> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在开始 Step4.2：Preference Data Path。

当前 Step4.1 已经完成 vector reward 数据链。

本阶段只加入 preference w 的生命周期和数据传递。

不要修改 Actor、Critic 或 learner 数学逻辑。

==================================================
一、本阶段目标
==============

只建立：

preference w
→ rollout
→ transition
→ SampleBatch
→ replay add/sample

的数据链。

当前首任务为 Walker2d：

num_objectives = L = 2

因此：

single preference:
w.shape == (2,)

batched preference:
w.shape == (B,2)

==================================================
二、preference 基本约束
=======================

每个 preference 必须满足：

w_i >= 0

sum_i(w_i) ≈ 1

当前 Step4.2 可以提供一个临时的纯 JAX smoke-test sampler：

sample_preference(key, batch_shape)

输出：

[...,2]

该 sampler 仅用于验证 preference 数据链。

如果使用连续均匀 simplex sampling，必须明确标记：

DEVIATION / TEMPORARY

它不是 PD-MORL 官方正式 episode sampler。

禁止在文档中把该 sampler 描述成 source-faithful PD-MORL。

正式 PD-MORL sampler 后续必须替换为：

discrete w_batch
+
np.array_split(w_batch,10)
+
logical worker subspace sampling

==================================================
三、episode preference 生命周期
===============================

每个环境 lane 必须维护自己的 episode preference。

规则：

reset / episode start:
为该 lane 分配一个 w

episode 内：
w 保持不变

combined done:
仅对 done lane 重采样新的 w

未结束的 lane：
继续使用原来的 w

禁止：

- 每个 environment step 都重新采样 w；
- 所有 lanes 共用同一个随机 w；
- 因某一个 lane done 而重采样全部 lanes。

resample 必须使用 JAX PRNG 和 mask，
不得使用 Python per-lane control flow。

==================================================
四、observation 与 preference 必须分离
======================================

保持：

obs:[B,17]

preference:[B,2]

不要：

concat 到 Brax observation
修改 observation space
让 environment physics 依赖 w

本阶段 w 只是 agent/workflow 条件 metadata。

Actor 尚未读取 w。

==================================================
五、SampleBatch / transition
============================

优先复用现有 SampleBatch。

将 preference 放在已经冻结的路径：

extras.policy_extras.preference

rollout transition 应满足：

observations: [...]
actions: [...]
rewards: [...,2]
dones: [...]
next_observations: [...]
preference: [...,2]

不要新增 MORLSampleBatch。

如果现有 extras 结构无法安全承载 preference，
停止并报告具体原因。

==================================================
六、Replay Buffer
=================

继续复用现有 generic replay buffer。

验证：

写入前：

preference.shape == (N,2)

写入 replay 后：

storage preference leaf 保持 objective/preference axis

sample 后：

preference.shape == (batch_size,2)

本阶段：

不做 HER relabel。

==================================================
七、本阶段禁止实现
==================

不要实现：

- preference subspace；
- 10 logical worker 正式采样；
- HER；
- interpolator；
- projected preference w_p；
- cosine / angle term；
- Hypernetwork；
- Actor 输入 (s,w)；
- Critic 输入 (s,w,a)；
- vector Q；
- scalarization；
- MO-TD3 target。

==================================================
八、测试
========

Test 1：
单 preference

验证：

shape == (2,)
all(w >= 0)
sum(w) ≈ 1

Test 2：
batched preference

B > 1

验证：

shape == (B,2)
每个 lane 满足 simplex constraint
不同 lanes 使用独立 PRNG

Test 3：
episode lifecycle

验证：

- episode 未结束时 w 不变化；
- 某一 lane done 时，只该 lane 的 w 更新；
- 其他 lane 的 w 完全保持；
- time-limit combined done 同样触发新 episode preference。

Test 4：
rollout

验证：

若 rollout 为 T steps、B envs：

preference trajectory shape == (T,B,2)

同一个未结束 episode 内对应 lane 的 w 保持一致。

Test 5：
SampleBatch / flatten

验证：

flatten 后 preference shape == (T*B,2)

reward 的 objective axis 与 preference axis 均不丢失。

Test 6：
Replay add/sample

验证：

preference 写入 replay 后可完整恢复；

sample 后：

preference.shape == (batch_size,2)

且与原 transition 对应关系保持。

Test 7：
JIT

preference sampler、
done-mask resample、
rollout、
replay data path

均能够在 jax.jit 下运行。

无 tracer error。
无 host RNG。
无 host callback。

Test 8：
GPU

记录：

jax.default_backend()
jax.devices()

要求：

backend == "gpu"

并看到目标 CudaDevice。

Test 9：
Step4.1 回归

确认 vector reward 数据链仍然通过：

reward.shape == (...,2)

并确认原 scalar Walker 路径仍未被破坏。

==================================================
九、模块位置
============

新增 preference utility 时遵循当前 EvoRL 项目结构。

可创建：

evorl/.../morl/preference.py

或更符合当前代码组织方式的等价位置。

不要直接创建与项目结构不一致的 src/... 路径。

==================================================
十、输出文档
============

创建：

docs/MORL_PREFERENCE_PIPELINE.md

记录：

1. preference shape；
2. 临时 sampler 算法；
3. 明确说明该 sampler 是否为 DEVIATION；
4. episode 生命周期；
5. preference 存储位置；
6. 修改文件；
7. 未修改文件；
8. replay shape；
9. JIT/GPU 验证；
10. 所有测试结果；
11. 尚未实现的正式 PD-MORL sampler/subspace。

==================================================
十一、完成条件
==============

只有以下全部满足才算：

STEP4.2 PASS

- w shape 正确；
- w 非负；
- sum(w)≈1；
- episode 内 w 固定；
- combined done 后只更新对应 lane；
- rollout 保留 w；
- SampleBatch 保留 w；
- replay add/sample 保留 w；
- JIT 正常；
- GPU backend 正确；
- Step4.1 vector reward 未被破坏；
- Actor 未修改；
- Critic 未修改；
- HER 未实现；
- preference subspace 未实现。

完成后停止。

不要继续 Step4.3。
