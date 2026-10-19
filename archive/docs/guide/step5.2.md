> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入 Step 5.2。

当前已经完成：

- Step4.1 vector reward；
- Step4.2 preference data path；
- Step4.3 official episode preference sampling；
- Step4.4 minimal preference-conditioned MO-TD3；
- Step5.1 MORL math primitives。

现在实现 PD-MORL 官方源码中的
preference relabeling / HER。

本阶段目标是：

SOURCE-FAITHFUL HER

不是 sample-time HER，
不是 lazy relabel，
不是传统 goal-state HER。

==================================================
一、先审查官方源码，不要立即修改
================================

首先检查 PD-MORL 官方源码：

lib/common_ptan/experience.py

重点检查：

ExperienceReplayBuffer_HER
其 _add / populate / sample 相关逻辑。

同时检查：

lib/utilities/settings.py
train_Walker2d_MO_TD3_HER.py

请明确回答：

1. 原 transition 在哪里写入 replay；
2. HER 在哪里生成；
3. 一个 base transition 最多生成多少 replay entries；
4. N_w / weight_num 的 Walker 默认值；
5. HER 的启动条件；
6. 启动条件使用 > 还是 >=；
7. 条件判断发生在 original entry 写入前还是写入后；
8. relabeled entries 的写入顺序；
9. buffer 满时原源码如何淘汰旧 entry；
10. relabeled preference 的精确生成过程；
11. 是否显式保证 w' != original w；
12. reward/state/action/next_state/done 是否发生任何变化。

先输出 SOURCE AUDIT。

如果源码实际行为与下面要求有冲突，
以源码为准，并先报告差异，不要自行折中。

==================================================
二、论文语义
============

PD-MORL 论文描述：

对于原 transition：

(s, a, r_vec, s', done, w)

除原 transition 外，
额外使用 N_w 个随机 preference：

w'_1 ... w'_Nw

得到：

(s, a, r_vec, s', done, w'_i)

核心原则：

只有 preference 被 relabel。

以下全部保持不变：

- state
- action
- reward vector
- next state
- done / termination information

不要重新计算 reward。

这不是 conventional goal-state HER。

==================================================
三、官方源码 HER preference sampler
===================================

严格复现官方 PyTorch 源码 sampler。

HER preference 使用：

1. independent standard normal samples；
2. absolute value；
3. L1 normalization；
4. round to 3 decimals。

数学上：

z ~ N(0, I)

u = abs(z)

w' = u / sum(u)

然后按源码：

round(w', 3)

使用 JAX PRNG 实现等价过程。

禁止使用：

- Dirichlet sampler；
- episode 1001-point grid；
- worker preference subspace sampler；
- uniform [0,1] sampler；
- rounding 后额外重新 normalize。

如果 zero/near-zero L1 norm 需要 JAX 数值保护，
标为：

FRAMEWORK-ADAPTATION

不要改变正常样本的源码分布。

==================================================
四、配置
========

新增：

num_relabel_preferences

它对应官方：

N_w / weight_num

Walker2d 默认：

num_relabel_preferences = 3

不要硬编码到 sampler 内部。

==================================================
五、严格复现 add-time HER
=========================

本阶段默认实现必须是：

PHYSICAL ADD-TIME RELABELING

禁止默认实现：

- sample-time relabel；
- lazy relabel；
- learner batch expansion。

对于每个 base transition：

原 transition 始终按官方行为写入 replay。

达到官方 HER activation condition 后：

再生成 N_w 个 relabeled preferences，
将它们作为独立 replay entries 物理写入同一个
logical global replay pool。

因此 warm-up 后：

1 base transition

最多对应：

1 + N_w replay entries。

Walker 默认：

1 + 3 = 4 entries。

==================================================
六、HER warm-up
===============

严格复现官方源码：

HER activation threshold =

start_timesteps * process_count

Walker 默认：

10000 * 10 = 100000

当前 process_count 已参数化，
所以不要重新硬编码 10。

使用同一个：

config.process_count

默认仍为 10。

必须检查并复现源码精确判断：

len(buffer) > start_timesteps * process_count

并确认判断发生在 original transition
写入前还是写入后。

不要擅自改成：

> =

也不要改成 environment vector steps。

这里判断的是源码对应的 replay entry count 语义。

==================================================
七、transition 不变量
=====================

对：

original transition

和：

每个 relabeled transition

要求以下逐元素完全相同：

state
action
reward vector
next_state
done
termination
truncation
其他需要保留的 transition metadata

唯一允许变化的是：

extras.policy_extras.preference

即：

w -> w'

禁止：

- 重算 reward；
- scalarize reward 后保存；
- 修改 done；
- 修改 next state；
- 修改 action；
- 修改 worker ID；
- 修改 episode physical transition。

==================================================
八、不要人为强制 w' != w
========================

论文描述 additional preference 与 original preference 不同。

但必须检查官方源码是否真的存在：

if w_prime == w:
    resample

之类逻辑。

如果官方源码没有显式 rejection sampling：

禁止为了符合论文文字自行增加 rejection loop。

测试可以使用固定 PRNG seed，
选择一个确定产生 w' != w 的案例来验证 relabel 生效。

但不要把：

w' != w

实现为额外算法约束。

在文档中记录：

论文语义
vs
源码实际行为。

==================================================
九、global replay 语义
======================

继续复用现有 EvoRL replay buffer。

不要创建第二套 HER replay buffer。

所有 logical preference workers 的：

original entries
+
relabeled entries

必须进入同一个逻辑 global replay pool。

禁止：

- per-worker isolated replay；
- per-preference isolated replay；
- per-GPU isolated replay。

learner 仍然从一个逻辑全局混合 replay distribution
均匀采样。

==================================================
十、JAX / replay 写入方式
=========================

官方源码是 Python/PyTorch sequential add。

JAX 版本允许使用固定 shape batched add，
但必须保证其最终 replay 内容和 ring-buffer
write order / eviction semantics 与源码等价。

不要用 Python 循环作为 jitted training path。

如果一次生成：

[base_batch, 1 + N_w, ...]

允许固定 shape flatten 后批量写入。

但必须验证：

logical insertion order

以及 wrap-around 后最终 buffer 内容
与逐条 source semantics 等价。

这属于：

FRAMEWORK-ADAPTATION

不能改变 replay sampling distribution。

==================================================
十一、满 buffer 行为
====================

必须专门检查并测试官方：

ExperienceReplayBuffer_HER

在 capacity 满时的行为。

使用一个非常小的 toy capacity，
例如能清楚触发 wrap-around / eviction 的容量。

验证：

- original entry；
- N_w relabeled entries；
- insertion order；
- oldest-entry eviction；
- write pointer；
- final buffer contents；

与官方源码逻辑一致。

不要只测试未满 buffer。

==================================================
十二、内存与容量分析
====================

Walker 官方：

replay capacity = 2,000,000 entries

N_w = 3

warm-up 后：

1 environment transition
→ up to 4 replay entries

必须报告两种概念：

A. 固定 replay capacity = 2,000,000

则：

物理预分配容量不因为 HER 自动 ×4，
但 replay 填充速度约增加到 4 倍，
样本年龄和淘汰速度发生变化。

B. 如果为了保持相同 base-transition history
人为把容量扩成：

2,000,000 * (1 + N_w)

则属于额外设计，
会显著增加设备内存。

本阶段不要自动放大 replay capacity。

保持官方 entry capacity 语义。

请计算当前 SampleBatch schema 下：

每个 replay entry 大约占用多少 bytes，
capacity=2,000,000 时理论 payload 大小，
并说明 HER 是否改变预分配内存和有效历史跨度。

==================================================
十三、测试
==========

至少增加：

1. HER sampler

固定 key，
验证：

abs(normal)
L1 normalize
round 3 decimals

行为正确。

---

2. original transition retained

在容量足够且未发生 eviction 的情况下，
写入一个 transition。

确认 original preference 对应 entry 仍存在。

---

3. relabel count

HER active 且：

N_w=3

一个 base transition 应产生源码对应的：

1 original + 3 relabeled entries

如果源码审查发现具体边界行为不同，
以源码为准。

---

4. transition invariants

original 与 relabeled：

state identical
action identical
reward identical
next_state identical
done identical
termination/truncation identical

只有 preference 可不同。

---

5. reward vector

确认：

reward.shape == [2]

且 relabel 不重新计算、不 scalarize reward。

---

6. warm-up boundary

专门测试：

threshold - 1
threshold
threshold + 1

根据源码实际判断位置，
验证 HER 开启时机完全一致。

---

7. process_count parameterization

例如：

process_count = 4
start_timesteps = 10

threshold 应动态成为源码等价的：

40

不能仍然写死 100000 或 ×10。

---

8. N_w parameterization

测试：

N_w = 0
N_w = 1
N_w = 3

确认 entry multiplicity 正确。

N_w=3 是 Walker 官方默认。

---

9. ring-buffer full behavior

使用小容量强制 wrap-around。

验证最终内容与官方逐条 add 语义一致。

---

10. shapes

preference [L]
reward [L]

batch sample 后：

preference [N,L]
reward [N,L]

保持现有 SampleBatch schema。

---

11. JIT

HER preference sampler 可以 jax.jit。

replay add path如果现有 EvoRL replay 本身支持 JIT，
验证完整 add 路径 JIT。

---

12. vmap

只对真正定义为 per-sample 的纯函数
（例如 HER preference sampler）
测试 vmap。

不要为了测试要求机械地对 replay mutation
套 vmap。

==================================================
十四、回归测试
==============

重新运行已有：

- preference pipeline tests；
- tests/test_mo_td3.py；
- tests/test_morl_math.py；
- Walker GPU smoke test 中与本次修改相关的回归。

确认没有破坏：

- vector reward；
- episode preference lifecycle；
- official episode sampler；
- minimal MO-TD3；
- scalarization；
- TD target；
- actor/critic loss；
- Step5.1 math。

本阶段还不要把 directional-angle loss
接入 actor/critic。

==================================================
十五、文档
==========

创建：

docs/PD_MORL_HER.md

必须分别记录：

1. PAPER SEMANTICS

论文如何描述：
每个 transition + N_w additional preferences。

2. SOURCE BEHAVIOR

官方 PyTorch 实际：

- N_w；
- HER sampler；
- rounding；
- warm-up condition；
- add timing；
- physical replay entries；
- full-buffer behavior；
- 是否真正保证 w' != w。

3. JAX MAPPING

哪些完全：

SOURCE-FAITHFUL

哪些属于：

FRAMEWORK-ADAPTATION

4. PAPER / SOURCE DIFFERENCES

特别记录：

- 论文描述每个 transition 都 relabel；
- 源码存在 warm-up；
- 论文描述随机 preference；
- 源码实际使用 absolute-normal + L1 + rounding；
- 是否存在“必须不同”的源码检查。

5. MEMORY / REPLAY ANALYSIS

记录：

capacity
entry bytes
warm-up 前后 entry rate
buffer fill rate
effective history horizon

==================================================
十六、禁止修改
==============

本阶段不要实现或修改：

- interpolator；
- projected preference w_p；
- directional angle loss；
- actor angle term；
- critic angle term；
- evaluation；
- Pareto/HV/sparsity；
- PSL-MORL；
- hypernetwork；
- multi-GPU sharding architecture。

不要修改已有 MO-TD3 数学机制。

==================================================
十七、完成标准
==============

只有以下全部满足才 PASS：

- source audit 完成；
- official HER sampler 对齐；
- N_w=3 默认对齐；
- physical add-time relabel 正确；
- warm-up boundary 对齐；
- original entry 保留；
- relabel multiplicity 正确；
- transition 仅 preference 改变；
- reward vector 完全不变；
- global replay semantics 不变；
- full-buffer behavior 对齐；
- JAX/JIT 可执行部分正常；
- regression tests 全部通过；
- 文档明确区分论文、源码、JAX。

完成后停止。
