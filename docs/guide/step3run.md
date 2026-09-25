
你现在执行 Step3：PD-MORL 官方连续控制任务行为冻结与 Brax 映射确认。

重要：本阶段只做源码分析、事实提取、环境对照和设计确认，不修改任何训练代码，不实现 PD-MORL，不重构 EvoRL，不提前编写多目标环境。

当前已有 Step2 文档：
PD_MORL_EvoRL_GAP.md

请严格继承其中已经确定的原则和约束。

==================================================
一、Step3 的核心目标
====================

Step3 要解决两个问题：

1. 从 PD-MORL 官方源码中，完整冻结第一个连续控制任务的实际运行行为；
2. 将该任务依赖的旧 MuJoCo state / observation / reward / termination / action semantics，与 Brax 中对应环境逐项比较，判断是否能够做 source-faithful 映射。

本阶段不是“根据论文设计一个类似环境”。

本阶段必须回答的是：

“PD-MORL 官方源码实际做了什么，以及 Brax 能否表达同样的东西？”

==================================================
二、事实来源优先级
==================

算法和实验行为的事实来源按以下顺序处理：

第一优先级：
PD-MORL 官方源码

重点检查：

1. lib/utilities/settings.py
2. lib/models/networks.py
3. lib/common_ptan/agent.py
4. lib/common_ptan/experience.py
5. PD-MORL/train_*_MO_TD3_HER.py
6. PD-MORL/train_*_MO_TD3_HER_Key.py
7. lib/utilities/morl/MOEnvs/MujocoEnvs/
8. lib/utilities/MORL_utils.py
9. eval_benchmarks_MO_TD3_HER.py

第二优先级：
PD-MORL 论文

论文只用于：

- 解释源码设计；
- 核对算法意图；
- 标记论文与源码不一致。

禁止用论文中的概念性描述覆盖官方源码的实际行为。

第三部分：
EvoRL / Brax 当前源码

只用于寻找等价的 JAX/Brax 承载方式。

==================================================
三、首先确定第一个目标任务
==========================

优先检查 PD-MORL 官方连续控制任务。

如果 Hopper 是最适合作为第一个复现任务的环境，请用源码证据说明原因。

不要因为之前文档提到 Hopper 就默认选择 Hopper。

至少比较：

- Hopper
- Walker
- HalfCheetah
- Ant
- Swimmer

如果官方仓库实际任务名称不同，以仓库为准。

选择标准只考虑：

1. PD-MORL 官方源码完整性；
2. Brax 是否存在对应环境；
3. reward/state/action/termination 是否容易做明确映射；
4. 目标数是否适合第一阶段复现；
5. 是否能避免不必要的环境重写。

输出最终建议的首个任务，但不要因为“实现简单”而修改官方 objective 定义。

==================================================
四、冻结该任务的官方实验配置
============================

选定目标任务后，逐文件提取以下实际参数。

必须给出：

- reward_size / objective count L
- state / observation dimension
- action dimension
- max_action / action bounds
- hidden layers
- hidden size
- activation
- actor output activation
- network initialization（如果源码明确）
- gamma
- tau
- batch_size
- replay buffer size
- start_timesteps
- total training steps
- expl_noise
- policy_noise
- noise_clip
- policy_freq
- actor_loss_coeff
- gradient clipping
- process_count
- weight_num
- HER preference count N_w
- evaluation preference grid
- evaluation repeat / seed behavior
- episode length / time limit
- 其他该任务脚本实际覆盖的参数

每个参数必须注明：

- 数值
- 来源文件
- 尽量给出源码行号
- 是否为全局默认值还是任务特定 override

如果源码没有明确给出，写：

NOT FOUND

禁止根据论文、TD3 常见默认值或经验自行补全。

==================================================
五、冻结官方网络结构
====================

逐层提取 Actor 和 Critic。

Actor 至少记录：

input:
state + preference

每层：
Linear(...)
Activation(...)
...

output:
action

Critic 至少记录：

input:
state + preference + action

Q1:
...

Q2:
...

output:
reward_size-dimensional vector

特别确认：

1. state 与 preference 的拼接顺序；
2. critic 中 action 的拼接位置；
3. Q1/Q2 是两个独立网络还是共享部分结构；
4. hidden size；
5. ReLU 或其他 activation；
6. actor tanh 和 max_action 的位置；
7. 初始化方式。

然后与 EvoRL 当前 TD3 actor/critic 逐项比较。

输出：

OFFICIAL
EVORL
DIFFERENCE
REQUIRED CHANGE

本阶段不要修改代码。

==================================================
六、冻结官方多目标 reward
=========================

这是 Step3 最重要的部分。

请直接阅读：

lib/utilities/morl/MOEnvs/MujocoEnvs/

中目标任务对应的环境源码。

不要只根据变量名概括为：

“forward speed”
“height”
“energy”

而要提取实际计算表达式。

对每个 objective 分别记录：

Objective 1:

- 源码表达式
- 使用的 state / qpos / qvel / xpos / body 信息
- 是否含 dt
- 是否含 alive bonus
- 是否含 control cost
- 是否含 clipping
- 是否与 previous state 比较
- 返回单位/尺度

Objective 2:
同上。

如果有更多 objective 继续列出。

最终写出类似：

r_t = [
    exact_reward_1_expression,
    exact_reward_2_expression
]

但表达式必须来源于官方源码。

==================================================
七、冻结官方 observation / state 语义
=====================================

必须区分：

1. MuJoCo internal state
2. 环境返回给 policy 的 observation
3. reward 计算直接读取的内部物理量

不要把它们混为一谈。

记录：

- observation 构造方式；
- qpos 哪些维度被删除/保留；
- qvel 哪些维度被使用；
- reward 是否读取 observation 中不存在的 internal state；
- actor 实际接收的 state shape；
- reset 后 observation 的结构。

==================================================
八、冻结官方 termination / done / truncation
============================================

逐行跟踪官方环境中：

done
terminated
time limit
episode length

的来源。

必须回答：

1. 什么条件导致物理 termination？
2. time limit 是否通过同一个 done 表达？
3. replay 中保存的 done 到底是什么？
4. MO_TD3 target 中使用的 done 具体来自哪里？
5. 是否存在 TimeLimit wrapper 对 done 的二次处理？

不要预设：

done == termination

也不要预设：

time limit 应该 bootstrap

只记录官方实际行为。

==================================================
九、冻结官方 action semantics
=============================

记录：

- action_space.low
- action_space.high
- max_action
- actor tanh 输出如何缩放
- exploration noise 的缩放
- target smoothing noise 的缩放
- clipping 顺序

特别确认官方环境是否所有动作维都共享相同上下界。

==================================================
十、建立 MuJoCo → Brax 对照表
==============================

然后检查 EvoRL 当前使用的 Brax 版本和对应环境实现。

必须逐项建立如下对照：

| 官方 PD-MORL / MuJoCo | Brax | 是否等价         | 证据 | 结论 |
| --------------------- | ---- | ---------------- | ---- | ---- |
| observation           | ...  | YES/NO/UNCERTAIN | ...  | ...  |
| forward velocity      | ...  | ...              | ...  | ...  |
| height                | ...  | ...              | ...  | ...  |
| qpos                  | ...  | ...              | ...  | ...  |
| qvel                  | ...  | ...              | ...  | ...  |
| action bounds         | ...  | ...              | ...  | ...  |
| termination           | ...  | ...              | ...  | ...  |
| timestep dt           | ...  | ...              | ...  | ...  |
| control cost          | ...  | ...              | ...  | ...  |
| episode time limit    | ...  | ...              | ...  | ...  |

对于每个 reward component，必须明确指出：

官方公式使用：
X

Brax 中候选量：
Y

是否数学/物理等价：
YES / NO / UNCERTAIN

禁止仅因为名字相似就判定等价。

==================================================
十一、映射分类
==============

每一项都必须标记：

SOURCE-FAITHFUL
FRAMEWORK-ADAPTATION
DEVIATION
BLOCKED

定义：

SOURCE-FAITHFUL：
算法/环境语义可以严格复现官方源码。

FRAMEWORK-ADAPTATION：
数值/算法语义一致，只是从 PyTorch/MuJoCo 实现成 JAX/Brax。

DEVIATION：
明确知道行为与官方不同。

BLOCKED：
当前证据不足，无法确认是否等价。

任何 BLOCKED 项都必须单独汇总。

禁止自行“合理处理”。

==================================================
十二、GPU/JAX 原则
==================

本阶段虽然不写训练代码，但所有未来映射必须满足：

- 训练后端继续使用 Brax + JAX；
- 不引入 Gymnasium/MuJoCo Python 环境作为训练后端；
- 不因为官方源码使用 NumPy/PyTorch 而把训练数据拉回 CPU；
- reward 如果未来在 Brax 中实现，应能在 JIT/device 内计算；
- 固定 shape；
- 不使用 Python host callback 处理每 step reward。

但本阶段不要实现这些代码。

==================================================
十三、不要做的事情
==================

本阶段禁止：

1. 修改 evorl/algorithms/td3.py
2. 创建新的 PD-MORL agent
3. 修改 replay buffer
4. 实现 vector critic
5. 实现 preference sampler
6. 实现 HER
7. 实现 interpolator
8. 实现 angle loss
9. 修改 Brax reward
10. 自行编写新的多目标环境
11. 自行选择新的 reward decomposition
12. 自行“修复”官方源码中看起来不合理的行为

Step3 是分析/冻结阶段，不是实现阶段。

==================================================
十四、必须输出的文档
====================

最终生成：

STEP3_PD_MORL_BRAX_MAPPING.md

文档至少包含：

# 1. Step3 结论

说明：

- 选择哪个官方任务；
- 是否具备进入实现阶段的条件；
- 是否存在 BLOCKED 项。

# 2. 官方任务源码索引

文件 → 行号 → 职责

# 3. 官方任务完整配置表

参数 / 数值 / 来源 / 行号

# 4. 官方 Actor / Critic 结构

逐层结构与 shape

# 5. 官方 observation 定义

# 6. 官方 vector reward 定义

对每个 objective 给出精确源码表达式。

# 7. 官方 action 定义

# 8. 官方 done / termination / time-limit 语义

# 9. 官方 training 时序

只记录与所选任务相关的：

- sampling
- replay
- policy_freq
- critic/actor update
- target update
- gradient clipping

# 10. MuJoCo → Brax 对照表

逐项对照 observation / state / reward / action / done / dt。

# 11. Mapping Classification

逐项标：
SOURCE-FAITHFUL
FRAMEWORK-ADAPTATION
DEVIATION
BLOCKED

# 12. BLOCKED / 需要人工确认的问题

只列真正无法通过源码证明的问题。

# 13. Step4 实现前置条件

明确列出哪些条件必须满足后才能开始修改代码。

==================================================
十五、最终判定
==============

最后只能给出以下三种结论之一：

A. READY FOR STEP4
所有核心 reward/state/action/done 映射均已确认，可以进入实现。

B. READY WITH EXPLICIT DEVIATIONS
可以实现，但存在已经明确知道的非等价项，必须列清楚。

C. BLOCKED
存在关键环境语义无法从源码/Brax 中建立可靠对应，不应开始实现。

不得为了推进项目而强行给 READY。

完成后先向我汇报 Step3 文档和发现，不要继续执行 Step4。
