
请继续修改当前 EvoRL 项目中的 PD-MORL GPU-native 实现。

这次任务不是重新设计架构，而是修复当前 GPU-native v1 已经通过实验暴露出来的训练语义和学习强度问题。

重要前提：

1. 继续使用 JAX + Brax + GPU。
2. 不切换到 MuJoCo、MJX、Gymnasium 或 PyTorch。
3. 保留现有 PDMORLWorkflow faithful 路径，不修改其现有行为。
4. 保留当前 PDMORLGPUWorkflow，在此基础上实现 GPU-native v2。
5. 不新增多 GPU、梯度累积、自定义 replay、第二套 HER 或 JAX-native RBF，除非本任务明确要求。
6. 当前目标是构建适合 GPU 的 PD-MORL 替代训练方案，而不是要求 Brax 的绝对 HV 与原论文 MuJoCo Table 3 完全一致。
7. 所有修改必须有测试，不能只改配置然后直接训练。

开始修改前，请先阅读：

- docs/PD_MORL_GPU_NATIVE_REPORT.md
- PD_MORL_GPU_NATIVE_GAP_ANALYSIS.md
- 当前 PDMORLGPUWorkflow
- evorl/algorithms/mo_td3.py
- evorl/evaluators/pd_morl.py
- replay/HER 实现
- Brax MOWalker2d adapter
- 当前 GPU-native yaml
- faithful reproduction yaml

先输出一个简短实施计划，然后直接执行，不需要等待我的二次确认。

---

一、首先修正 evaluation 和日志
------------------------------

这是第一优先级，必须先完成。

当前 sparsity 的计算顺序有问题。

请检查 evorl/evaluators/pd_morl.py。

GPU-native 和 faithful 共用的指标语义必须修改为：

对于每一个 evaluation repeat：

1. 得到全部 preference 对应的 vector returns；
2. 对这些 returns 做 non-dominated filtering；
3. 只在该 repeat 的 non-dominated Pareto front 上计算 sparsity；
4. HV 也明确基于该 Pareto front 计算；
5. 保存该 repeat 的 Pareto point count。

最终 evaluation 必须额外记录：

- final_hv
- final_sparsity
- final_pareto_point_count
- 每个 repeat 的 HV
- 每个 repeat 的 sparsity
- 每个 repeat 的 Pareto point count
- objective 1 的 min/max/mean
- objective 2 的 min/max/mean
- Pareto returns
- preference → return 映射
- evaluation preference grid

如果当前日志系统不适合完整保存 Pareto arrays，可以同时保存独立 npz/json artifact。

训练结束必须保存 final checkpoint。

不要再出现“训练结束后无法重新加载最终 agent”的情况。

为此增加对应测试：

- dominated points 不应进入 sparsity；
- serial evaluator 和 batched evaluator 使用完全相同的 Pareto filtering；
- Pareto point count 与保存的 Pareto returns 长度一致；
- final checkpoint 可以成功 reload 并执行 evaluation。

---

二、重新定义 GPU-native random-action warm-up
---------------------------------------------

这是当前训练失败的主要原因之一。

原 PD-MORL 有：

10 个 logical workers
每个 worker 10,000 random-action steps

因此原算法大致对应：

100,000 global random transitions。

GPU-native 中新增的 envs_per_preference_group 只是 GPU sampling replicas，
不能让每个物理 lane 都重新拥有一个完整的 10,000-step warm-up。

请将 GPU-native warm-up 改为“logical preference group 累计计数”。

保持：

logical_preference_groups = 10

对每个 logical group 维护：

group_random_transition_count

一个 group 内所有物理 lanes 产生的有效 transition 都累加到该 group。

当：

group_random_transition_count < 10,000

时，该 group 内环境使用 random action。

达到：

10,000

之后，该 group 的所有 lanes 改为：

actor(s, w) + exploration noise

因此默认整体约在：

10 × 10,000 = 100,000 global valid transitions

之后基本结束 random-action warm-up。

注意：

envs_per_preference_group 变化时，
100,000 这个 global random-transition scale 不应自动乘以 lane 数。

例如：

16 env/group

意味着一个 group 约在每个 lane 平均贡献 625 条 transition 后结束 random phase，
而不是要求每个 lane 都运行 10,000 step。

请记录：

- 每个 logical group 的 random transition count
- 每个 logical group 的 policy transition count
- global random transition count
- global policy transition count
- random_action_fraction

并增加测试：

envs_per_group = 1 / 4 / 16 时，
logical group 的 warm-up transition budget 都仍然是 10,000，
不能随 physical lane count 改变。

---

三、保持 HER warm-up 的 global base-transition 语义
---------------------------------------------------

继续使用当前：

HER activation threshold = 100,000 global base transitions

不要改成：

100,000 × physical lane count

也不要改成 per-lane threshold。

确认测试：

不同 envs_per_preference_group 下，
HER 都在相同的 global base-transition count 附近启动。

---

四、重新设计 learner 强度，但不要退回 CPU 式 640 次顺序更新
-----------------------------------------------------------

当前配置：

num_envs = 160
rollout_length = 4

因此每个 rollout：

640 new transitions。

当前：

replay_batch_size = 512
critic_updates_per_rollout = 1

导致 learner 明显跟不上 sampler。

GPU-native v2 的目的不是恢复：

640 sequential optimizer updates

因为这会重新引入 CPU 风格串行瓶颈。

请保留：

大 batch
+
较少 optimizer steps
+
GPU 矩阵并行

这个设计。

新增正式实验候选配置，第一轮固定：

replay_batch_size = 4096

然后测试：

critic_updates_per_rollout = 5
critic_updates_per_rollout = 10
critic_updates_per_rollout = 20

暂时不要把 40 作为默认正式配置。

对于每个配置计算并记录：

new_transitions_per_rollout
replay_samples_per_transition

公式：

replay_samples_per_transition
=============================

critic_updates_per_rollout
× replay_batch_size
/
(rollout_length × num_envs)

因此当前 160 env、T=4、batch=4096 时：

K=5  -> 32
K=10 -> 64
K=20 -> 128

将其作为 learner data-reuse intensity。

注意：

不要根据 transitions/s 单独选择最佳 K。

必须同时记录：

- environment transitions/s
- critic optimizer updates/s
- replay samples/s
- actor updates/s
- learner wall time
- rollout wall time
- evaluation wall time
- GPU utilization
- GPU memory
- HV vs environment transitions
- HV vs critic optimizer steps
- HV vs wall-clock time

最终目标是 time-to-quality，而不是最大 transitions/s。

---

五、Actor / target delayed update 保持 policy_freq 语义
-------------------------------------------------------

第一版 GPU-native v2 不再新增新的 actor scheduling。

继续使用：

global_critic_optimizer_step

作为 policy delay 计数器。

当：

global_critic_optimizer_step % policy_freq == 0

时执行：

actor update
target actor update
target critic update

保持当前 Walker 配置中的：

policy_freq = 10

因此无论 critic_updates_per_rollout 是 5、10 还是 20，
actor/target 都按照实际 critic optimizer step 数触发。

增加测试验证：

K 改变时，
actor update count / critic update count 仍约为 1 / policy_freq。

---

六、Walker episode horizon 统一为 500
-------------------------------------

当前 Brax GPU-native 使用 max_episode_steps=1000，
而原 PD-MORL Walker 配置为 500。

我们继续使用 Brax，不要求物理动力学与 MuJoCo 完全相同，
但是没有必要额外保留 episode horizon 差异。

请新增 GPU-native v2 配置：

max_episode_steps = 500

同时用于后续 Brax reference / GPU-native 对比。

不要删除旧配置；
旧 1M 实验结果必须仍可追溯。

---

七、不要把 160 env 直接视为最终最佳值
-------------------------------------

目前 160 env 最优只是在：

critic_updates_per_rollout = 1

条件下得到的 sampler benchmark。

修复 learner 后需要重新 benchmark。

第一轮 learner benchmark 可以继续使用：

160 env

完成 K=5 / 10 / 20 测试。

得到合理 learner 强度后，再比较：

envs_per_group = 8
即 80 env

和：

envs_per_group = 16
即 160 env

如显存允许，可额外比较：

envs_per_group = 4
即 40 env

但是不要做完整笛卡尔积。

先确定合理 K，
再比较 environment parallelism。

最终选择必须基于：

time-to-quality

不能只基于 transitions/s。

---

八、global environment budget 保持明确
--------------------------------------

GPU-native 的正式预算继续按：

global valid base transitions

定义。

增加 parallel lanes 不能自动增加训练预算。

当前下一轮调试实验继续：

1,000,000 global base transitions

作为 GPU-native v2 的短程验证。

以后正式长实验再单独决定是否跑 10M。

HER relabeled transitions：

不计入 environment-transition budget。

evaluation transitions：

不计入 training-transition budget。

---

九、1M GPU-native v2 实验
-------------------------

完成以上代码、测试和 benchmark 后，
先不要跑多个 seed。

先继续使用：

seed = 42

运行：

1,000,000 global valid base transitions

使用经过 benchmark 后选出的 learner 配置。

该实验主要验证：

1. 是否在约 100k global transitions 后真正退出 random-action phase；
2. actor + exploration 是否参与后续数据采集；
3. critic update 数是否达到设计值；
4. actor update 数是否与 policy_freq 一致；
5. Pareto front 是否仍严重坍缩；
6. HV 是否出现实际学习趋势；
7. replay / HER / interpolator 是否正常；
8. 是否出现 NaN / Inf；
9. wall-clock 是否仍具有明显 GPU 加速价值。

不要以论文 MuJoCo HV=5.41e6 作为此次 Brax 实验的硬性成功阈值。

必须与旧：

outputs/gpu_native_1m_seed42

进行对照，至少比较：

old GPU-native v1
vs
GPU-native v2

包括：

- random_action_fraction
- critic update count
- actor update count
- final HV
- best HV
- Pareto point count
- sparsity
- objective return ranges
- wall-clock
- transitions/s
- replay samples/s

---

十、如果 v2 仍然学习失败，不要直接增加 seed
-------------------------------------------

如果 1M v2 满足：

已经退出 random warm-up，
learner update 强度正常，
没有 NaN/Inf，

但 Pareto front 仍严重坍缩，

则先诊断：

- objective 1 return range
- objective 2 return range
- preference → return mapping
- vector Q range
- target Q range
- actor action range
- interpolator output wp
- directional angle
- HER preference distribution

不要立即启动 3/6 个 seed。

只有单 seed 已证明训练机制正常后，
才进入多 seed 实验。

---

十一、Brax reference baseline
-----------------------------

本项目的目标是 GPU-native PD-MORL 替代方案，
因此暂时不要切换到 MuJoCo 或 MJX。

后续需要建立一个：

Brax faithful-style/reference configuration

要求与 GPU-native 使用：

- 相同 Brax Walker adapter
- 相同 vector reward
- 相同 horizon=500
- 相同 evaluation protocol
- 相同 preference grid
- 相同 metric implementation

但尽量保持原 PD-MORL：

- 10 logical workers
- 较低 environment parallelism
- source-faithful learner scheduling

它的作用不是和论文 MuJoCo 数值完全重合，
而是作为同一 Brax 环境里的质量基线。

不要在本阶段立即跑完整 reference 实验，
先把配置和入口准备好。

---

十二、报告和交付
----------------

不要只在终端输出结果。

更新：

docs/PD_MORL_GPU_NATIVE_REPORT.md

并新增或更新 gap analysis。

报告必须包含：

1. GPU-native v1 已知问题；
2. GPU-native v2 修改；
3. warm-up 新语义；
4. learner data reuse 定义；
5. benchmark 表；
6. serial/batched evaluator correctness；
7. 1M seed42 新结果；
8. old v1 vs v2 对比；
9. 当前仍存在的 algorithmic deviation；
10. 下一步建议。

同时保存：

- benchmark JSON
- 1M experiment config
- final checkpoint
- final Pareto returns
- final metrics JSON/NPZ
- 测试结果摘要

---

十三、实施顺序
--------------

严格按以下顺序执行：

A. 修 evaluator / sparsity / final logging
B. 跑相关 unit tests
C. 修 logical-group random warm-up
D. 跑 warm-up tests
E. 确认 HER threshold tests
F. 修改 horizon=500
G. 实现 learner K 配置
H. benchmark K=5/10/20
I. 根据 benchmark 选择一个候选
J. 运行小规模 smoke test
K. 运行 1M seed42 GPU-native v2
L. 分析结果
M. 更新报告

每个阶段测试失败时先修复，
不要跳过测试继续训练。

不要为了测试通过而降低已有 assertion 或 tolerance。

---

十四、最终停止条件
------------------

本次任务完成必须至少满足：

- faithful 现有 regression tests 仍全部通过；
- GPU-native evaluator correctness tests 通过；
- sparsity 已在 non-dominated front 上计算；
- final checkpoint 能成功 reload；
- warm-up 不再按 physical lane 单独累计 10k；
- 约 100k global random transitions 后训练进入 actor-driven collection；
- HER threshold 不随 lane 数变化；
- policy delay 使用 critic optimizer step 计数；
- K=5/10/20 benchmark 已完成；
- 至少一个 GPU-native v2 配置完成 1M seed42；
- 运行无 NaN/Inf；
- v1/v2 对比和所有实验数据已经写入报告。

不要自行切换 MuJoCo/MJX。
不要自行开启多 GPU。
不要在当前阶段重新设计 PD-MORL 网络、HER 或 interpolator 数学。
不要删除旧实验结果或旧配置。
