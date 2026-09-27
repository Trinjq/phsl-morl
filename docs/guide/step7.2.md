
现在不要开始 Multi-GPU。

先做一次：

Remove Obsolete Reduced-Network Test Overrides

目标：

彻底清理此前 Step7 Walker smoke 中
为了加速测试而使用的缩小版 Actor/Critic 网络参数，

避免后续开发、Multi-GPU 测试或配置继承时
意外回退到小网络。

当前已经确认：

production Walker：

Actor hidden = [400,400]
Critic Q1 hidden = [400,400]
Critic Q2 hidden = [400,400]

Step7.1 production-shape single-GPU smoke
已经在真实 400×400 网络上 PASS。

==================================================
一、先做 Audit
==============

全仓库搜索所有可能影响
PD-MORL Walker 网络大小的配置或测试代码。

重点搜索：

hidden_size
hidden_sizes
actor_hidden
critic_hidden
actor_hidden_layer_sizes
critic_hidden_layer_sizes
network_override
small_network
tiny_network
test_network
64
32
128
256

以及：

tests/test_pd_morl_integration.py
tests/test_pd_morl_production_shape_smoke.py

configs/agent/
configs/env/
configs/
任何 PD-MORL / MO-TD3 / Walker YAML。

先输出：

REDUCED NETWORK OVERRIDE AUDIT

列出：

- file
- field
- current value
- purpose
- 是否影响 PD-MORL Walker
- 是否应删除

不要直接删除前先审计。

==================================================
二、删除 obsolete reduced-network overrides
===========================================

如果发现旧 Step7 smoke 中存在类似：

actor_hidden_sizes = [64,64]
critic_hidden_sizes = [64,64]

或：

hidden_size = 32
hidden_size = 64
hidden_size = 128

等专门为了 PD-MORL Walker smoke
缩小网络的 override：

删除。

包括：

- test fixture override；
- helper function parameter；
- temporary config mutation；
- Hydra override；
- monkeypatch；
- hard-coded small hidden size；
- dedicated small-network Walker config。

不要保留一个
“以后可能会用”的 PD-MORL small-network fallback。

==================================================
三、Production Walker 网络只允许 400×400
=========================================

PD-MORL Walker production baseline：

Actor：

[19,400]
→ [400,400]
→ [400,6]

Q1：

[25,400]
→ [400,400]
→ [400,2]

Q2：

[25,400]
→ [400,400]
→ [400,2]

Q1/Q2 参数独立。

Production config 必须保持：

actor hidden = [400,400]
critic hidden = [400,400]

==================================================
四、保留合法 TEST OVERRIDE
==========================

不要误删其他为了 smoke 加速而合理存在的 override。

允许继续保留：

- shorter episode length；
- shorter prefill；
- shorter learner-start timing；
- shorter random-action warmup；
- shorter HER activation timing；
- shorter key-update trigger；
- shorter full-evaluation trigger；
- smaller evaluation grid；
- fewer evaluation repeats；
- temporary checkpoint directory。

这些属于：

TEST OVERRIDE

但网络尺寸不再属于 TEST OVERRIDE。

==================================================
五、删除 fallback 逻辑
======================

检查是否存在类似逻辑：

if smoke_test:
    hidden_sizes = [64,64]
else:
    hidden_sizes = [400,400]

如果有：

改为：

所有 PD-MORL Walker path
统一从 production config 读取 [400,400]。

smoke_test
只允许覆盖：

runtime duration / trigger / evaluation cost

不得覆盖 network architecture。

==================================================
六、防止以后回退
================

增加一个明确的 regression guard。

例如：

test_pd_morl_walker_network_is_production_shape

必须断言：

actor layer1 == [19,400]
actor layer2 == [400,400]
actor output == [400,6]

Q1 layer1 == [25,400]
Q1 layer2 == [400,400]
Q1 output == [400,2]

Q2 同样。

另外断言：

任何 real-Walker PD-MORL smoke
最终 build 出来的网络
都必须是 production shape。

==================================================
七、检查 production-shape smoke
===============================

确保：

tests/test_pd_morl_production_shape_smoke.py

不再依赖任何旧 small-network helper。

它必须直接或间接从：

production Walker config

获得：

[400,400]

而不是在 test 内重新写一遍
隐藏层尺寸。

最好采用：

production config as single source of truth。

==================================================
八、避免影响普通 TD3
====================

只清理：

PD-MORL / MO-TD3 Walker
此前遗留的 reduced-network smoke override。

不要修改：

普通 scalar TD3
其他环境
其他算法
它们自己的合法测试网络尺寸。

==================================================
九、文档同步
============

更新：

docs/PD_MORL_EVORL_BASELINE.md

把旧 Step7 Walker smoke 中：

“network was shortened”

这件事保留为：

historical Step7 test information

但新增说明：

Obsolete reduced-network override
has been removed after Step7.1.

Current PD-MORL Walker tests
no longer permit network-size override.

不要伪造历史记录，
不要把原来的 Step7 结果改写成
当时也是 400×400。

==================================================
十、Regression
==============

清理后运行：

1. production-shape single-GPU smoke
2. Step7 integration
3. Step6 Golden Tests
4. Step6.1
5. scalar TD3 regression

重点确认：

production-shape smoke 仍为：

Actor = [400,400]
Critics = [400,400]
batch = 256
K = B = 10

且：

frozen tolerances unchanged。

==================================================
十一、最终输出
==============

输出：

REDUCED NETWORK OVERRIDE REMOVAL RESULT

包含：

1. 删除了哪些文件中的哪些 override；
2. 哪些合法 TEST OVERRIDE 被保留；
3. production network single source of truth 在哪里；
4. 新增了什么 guard test；
5. regression 结果；
6. 是否还存在任何 PD-MORL small-network path。

最终要求：

PD-MORL Walker production / smoke / integration
均不得再存在缩小网络入口。

完成后停止。

不要开始 Multi-GPU。
