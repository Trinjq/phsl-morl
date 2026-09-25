# EvoRL TD3 → PD-MORL 算法与工程差异地图

## 1. 目的与边界

本文要解决的问题是：在不破坏 EvoRL 现有 TD3、JAX/JIT 和 Brax 并行执行路径的前提下，明确实现 PD-MORL 所需的最小算法与工程改动，并把论文描述、官方 PyTorch 源码行为和 EvoRL/JAX/Brax 等价映射严格区分。

本文是设计审查，不包含 PD-MORL 实现。后续实现必须继续使用 **Brax + JAX**，不得引入 Gymnasium 或 MuJoCo Python 环境作为训练后端。官方 PyTorch/MuJoCo 代码仅用于确认算法语义。

### 1.1 核对材料

- EvoRL：`E:/projects/evorl/evorl/algorithms/td3.py` 及其网络、rollout、replay、evaluation 依赖。
- PD-MORL 论文：`E:/PD-MORL/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/Basaklar 等 - 2023 - PD-MORL Preference-Driven Multi-Objective Reinforcement Learning Algorithm.pdf`。
- 官方源码：`E:/PD-MORL/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/`。
- 论文页码以下均指 PDF 正文印刷页码；重点是 Sec. 3、Sec. 4.1–4.2、Appendix B.1.1–B.1.5、Algorithm 3 和 Appendix B.2。

### 1.2 状态定义

- **UNCHANGED**：数学语义可保持；可能仍需把新增维度沿现有接口传递。
- **EXTEND**：保留现有组件，并增加 preference 或向量奖励能力。
- **REPLACE**：现有标量 TD3 逻辑不能表达 PD-MORL，需要换成多目标版本。
- **NEW**：EvoRL TD3 当前没有该概念，需要新增。

本文中的 **逻辑 sampling lanes** 首先指官方 `process_count=C_p=10` 个 preference 子空间/采样 workers；**物理 GPU 数量**仅决定这 10 个逻辑 lanes 放在哪个设备上。两者必须彻底解耦。应优先使用 Brax 和现有网络已经提供的原生 batch 语义；只有某个函数本身只定义了单样本语义时，才局部使用 `jax.vmap`。`vmap` 不是架构要求，也不应为已经 batch 化的 Brax 路径再机械套一层。

### 1.3 事实来源与映射分类

PD-MORL **官方源码是算法实现的主要事实来源**。必须先逐文件提取实际执行行为，再映射到 EvoRL；论文只用于解释公式、定位设计意图和交叉核对，论文与源码不一致时不得自行折中。

每项后续改动必须且只能标为以下三类之一：

- **SOURCE-FAITHFUL**：算法语义和实际行为严格来自 PD-MORL 官方源码。
- **FRAMEWORK-ADAPTATION**：算法语义不变，仅为 EvoRL/JAX/Brax 做等价工程改写。
- **DEVIATION**：不能严格复现源码、或经明确批准暂时采用不同运行行为；必须单独记录并验证影响。

执行规则：

1. reward、preference sampling、HER、MO-TD3 target、actor/critic loss、directional angle、interpolator、policy delay 和 evaluation 不得自行设计。
2. 保留 EvoRL 的 JAX/JIT、Brax batched environment、generic replay、delayed-update 框架和 Polyak helper；不引入 Gymnasium/MuJoCo Python 训练后端或 PyTorch multiprocessing。
3. CPU/multiprocessing 并行采样映射为固定 `C_p=10` 的逻辑 Brax sampling lanes，再将这些 lanes 部署到 1 或 3 张 GPU；GPU 数量不得改变 preference 子空间数量、`np.array_split(w_batch,C_p)` 结果、HER warm-up 中的 `process_count` 或每轮采样/更新比。NumPy/SciPy 训练期计算改为等价 JAX 运算，或放在明确的非 JIT boundary。
4. Brax 与旧 MuJoCo 在 state、reward、termination 或 action bounds 上无法证明对应时，停止实现：列出官方源码定义、Brax 可用量及差异，等待人工确认。未经确认的映射均记为 **DEVIATION**，不得伪装为 source-faithful。

#### 官方源码逐文件行为索引

| 官方文件 | 实际代码行为 | 映射时的约束 |
|---|---|---|
| `lib/models/networks.py` | Actor 拼接 `(state,preference)`；Critic 拼接 `(state,preference,action)`；两个 critic 各输出 `reward_size` 向量；ReLU hidden layers，actor 末层 `tanh*max_action` | 网络输入、输出、层数、hidden size、激活和初始化均从源码/settings 提取，不用 EvoRL 默认值替代 |
| `lib/common_ptan/agent.py` | 实现 exploration、target smoothing、按 `w^TQ` 选择整条 target-Q、vector target、Smooth-L1 critic loss、angle terms、actor loss、delayed actor/target update | 这是 MO-TD3 数学与更新时序的主要事实来源；每个 tensor reduction 都要对齐 |
| `lib/common_ptan/experience.py` | episode preference 生命周期；HER absolute-normal/L1/rounding；warm-up 后写入 relabeled entries；满 buffer 后按 `1+N_w` 淘汰/写入 | 第一阶段 sample-time HER 是明确 `DEVIATION`；后续忠实版要覆盖这些细节 |
| `lib/utilities/MORL_utils.py` | evaluation preference 网格、vector returns、non-dominated filtering、HV、sparsity；sparsity 在非支配点数 `<=1` 时返回 0 | evaluation 不自行改 reference、符号、去重、排序或特殊值行为 |
| `lib/utilities/settings.py` | 各环境的 `weight_num`、noise、`gamma`、batch、`process_count=10`、`tau`、`policy_freq`、actor angle coefficient 等 | `process_count` 是逻辑 preference-worker 数，不是设备数；其余参数也必须按所选任务读取 |
| `PD-MORL/train_*_MO_TD3_HER.py` | 构造离散 `w_batch`、`np.array_split(w_batch,10)` 子空间、10 个 CPU child processes、主 learner 更新比、RBF 初始化与在线重拟合 | 固定 10 个逻辑 workers、采样/更新比和 preference 覆盖；只有进程/队列到 GPU placement 的转换是工程适配 |
| `PD-MORL/train_*_MO_TD3_HER_Key.py` | 对 key preferences 分别预训练并产出 interpolator 所需 objective solutions | key solutions 来源固定为官方流程，不用临时在线猜测替代 |
| `PD-MORL/eval_benchmarks_MO_TD3_HER.py` | 加载 actor、遍历测试 preference、多次评估、构造 mean-return Pareto front 并报告 HV/sparsity | 评估聚合顺序和输出含义保持不变 |
| `lib/utilities/morl/MOEnvs/MujocoEnvs/*.py` | 定义旧 MuJoCo state/reward/termination 的任务语义 | 只作为 Brax 映射依据；没有逐项对应时停止并等待确认，不作为训练后端 |

### 1.4 总览

| # | 模块 | 状态 | 核心差异 |
|---:|---|---|---|
| 1 | environment reward | **REPLACE** | 标量奖励改为固定维度向量奖励 |
| 2 | observation | **UNCHANGED** | 物理 observation 不变，`w` 不并入环境状态 |
| 3 | action | **UNCHANGED** | 连续动作及边界语义不变 |
| 4 | preference `w` | **NEW** | 新增 simplex 上的 episode-level 条件变量 |
| 5 | transition | **EXTEND** | transition 增加向量 reward 和生成动作时的 `w` |
| 6 | `SampleBatch` | **EXTEND** | 沿 PyTree extras 携带 preference，reward 尾维变为 `L` |
| 7 | replay buffer | **EXTEND** | 复用 PyTree ring buffer，但保持跨 10 个 workers 的逻辑全局采样池 |
| 8 | actor input | **EXTEND** | 从 `s` 变为 `(s,w)` |
| 9 | actor output | **UNCHANGED** | 仍输出连续动作向量 |
| 10 | critic input | **EXTEND** | 从 `(s,a)` 变为 `(s,w,a)` |
| 11 | critic output | **REPLACE** | 每个 critic 从标量改为 `L` 维向量 |
| 12 | vector Q | **NEW** | 新增逐目标动作价值 |
| 13 | scalarization `w^T Q` | **NEW** | 用原 preference 比较/优化向量 Q |
| 14 | TD target | **REPLACE** | 标量 target 改为整向量 target |
| 15 | critic loss | **REPLACE** | 向量回归损失加方向角损失 |
| 16 | actor loss | **REPLACE** | 标量化 Q 目标加方向角正则 |
| 17 | target actor | **EXTEND** | 保持 Polyak 机制，网络增加 `w` 输入 |
| 18 | target critics | **EXTEND** | 保持 twin/Polyak 机制，输出变为向量 |
| 19 | exploration noise | **UNCHANGED** | 行为动作上的高斯噪声保持 |
| 20 | target smoothing noise | **UNCHANGED** | clipped target-policy smoothing 保持 |
| 21 | delayed policy update | **UNCHANGED** | actor 和 target 仍按较低频率更新 |
| 22 | preference sampling | **NEW** | episode preference 和 HER preference 采样 |
| 23 | preference-space subdivision | **NEW** | 固定划分为 `C_p=10` 个逻辑 preference 子空间 |
| 24 | parallel child processes | **REPLACE** | 10 个逻辑 workers 保持不变，仅映射到 1 或 3 张 GPU |
| 25 | HER / preference relabeling | **NEW** | 同一物理 transition 以多个 `w` 重放 |
| 26 | interpolator `I(w)` | **NEW** | 从 preference 映射到投影后的目标方向 |
| 27 | projected preference `w_p` | **NEW** | `w_p=I(w)`，只参与角度项 |
| 28 | cosine similarity `S_c` | **NEW** | 新增几何相似度原语；MO-TD3 target 不直接用 Eq. (5) |
| 29 | directional angle `g(w_p,Q)` | **NEW** | 新增 critic/actor 的方向角约束 |
| 30 | evaluation | **REPLACE** | 标量 return 评估改为 preference 扫描和向量 return |
| 31 | Pareto front construction | **NEW** | 从多 preference 的 return 中筛非支配集 |
| 32 | hypervolume | **NEW** | 新增 Pareto 集质量指标 |
| 33 | sparsity | **NEW** | 新增前沿分布稀疏度指标 |

## 2. 数据模型与环境

### 2.1 environment reward — **REPLACE**

- **EvoRL 当前实现**：`evorl/envs/brax.py` 的 `BraxAdapter.step` 直接传递 `brax_state.reward`；`evorl/algorithms/offpolicy_utils.py` 用环境样本构造 replay dummy batch。原版 Brax TD3 假定 reward 为每环境一个标量。
- **PD-MORL 官方源码**：各任务在 `lib/utilities/morl/MOEnvs/MujocoEnvs/` 中返回 reward vector；Hopper 的 `hopper.py` 将前进速度和跳跃高度组成二目标奖励。
- **论文位置**：Sec. 3 定义 MOMDP 的 `r=[r_1,...,r_L]^T`；Appendix B.2 给出连续控制任务的目标定义，Hopper 为 forward speed 与 jumping height。
- **JAX/EvoRL 映射**：先逐行提取官方环境的 reward 公式、所读 state 字段、常数和 termination 条件，再逐项列出 Brax state/metrics 中可用的候选量。只有对应关系获确认后，才实现 Brax-native `[B,L]` reward；不得在 Python host 端回算，也不得凭名称自行定义“速度”“高度”或存活奖励。
- **JIT/vmap 风险**：`L` 必须是静态尾维；禁止依据目标数动态创建数组或用 Python dict 作为每步 reward。Brax metrics 若缺少目标量，应在 jitted env step 内计算。
- **论文/源码不一致**：论文给出目标语义，但官方实现绑定旧 MuJoCo 环境的具体 state/坐标索引；不能逐行移植到 Brax并假定状态索引相同。
- **映射分类**：**DEVIATION**（当前尚未证明旧 MuJoCo reward/state/termination 与 Brax 等价，不能开始实现）。
- **ASSUMPTION / 待确认**：首个任务、Brax 中每个 reward 分量的精确定义、存活奖励归属以及 termination/truncation 对齐均等待人工确认；不得默认选择 Hopper 或默认映射。

### 2.2 observation — **UNCHANGED**

- **EvoRL 当前实现**：`evorl/envs/env.py` 的 `EnvState.obs`，经 `evorl/rollout.py` 直接送入 agent；TD3 policy 只读取物理 observation。
- **PD-MORL 官方源码**：`lib/models/networks.py::Actor.forward` 和 `Critic.forward` 在网络内部将 state 与 preference 拼接；环境 observation 本身未改写。
- **论文位置**：Algorithm 3 的 actor 为 `π_φ(s,w)`，critic 为 `Q_θ(s,w,a)`；`s` 与 `w` 是不同变量。
- **JAX/EvoRL 建议**：保持 `obs:[B,O]`；把 `w:[B,L]` 作为 agent 条件输入和 transition metadata，不把它伪装成 Brax observation。
- **JIT/vmap 风险**：无实质风险；只需保证 `obs` 与 `w` 的 batch 前缀一致。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**（源码明确将 state 与 preference 作为独立输入后拼接）。
- **ASSUMPTION / 待确认**：无算法假设。observation normalization 保持 EvoRL 当前行为，不新增源码不存在的 normalization。

### 2.3 action — **UNCHANGED**

- **EvoRL 当前实现**：`evorl/networks/linear.py::make_policy_network` 输出连续动作；`evorl/algorithms/td3.py::compute_actions` 添加 exploration noise 并裁剪。
- **PD-MORL 官方源码**：`lib/models/networks.py::Actor` 用 `tanh` 后乘 `max_action`；`lib/common_ptan/agent.py::MO_TD3_HER.__call__` 加噪并裁剪。
- **论文位置**：Algorithm 3 延续 TD3 的连续 action 与动作边界。
- **JAX/EvoRL 建议**：保持 `action:[B,A]` 及现有 Brax action clipping；只改变 action 的条件输入，不改变 action 的含义。
- **JIT/vmap 风险**：无；边界应是可广播的 JAX array。
- **论文/源码不一致**：官方源码假定对称的 `[-max_action,max_action]`；Brax 常为 `[-1,1]`，但不能对任意自定义环境盲目假定。
- **映射分类**：**DEVIATION**（Brax action bounds 未与官方环境逐项核对前不能声称等价）。
- **ASSUMPTION / 待确认**：需要列出官方 `max_action` 与目标 Brax action bounds；确认前不得默认 `[-1,1]` 或新增 affine scaling。

### 2.4 preference `w` — **NEW**

- **EvoRL 当前实现**：原 TD3 无 preference；`TD3Agent.compute_actions`、loss 和 workflow state 都没有 `w`。
- **PD-MORL 官方源码**：`lib/common_ptan/experience.py::MORLExperienceSource` 为 episode 保存 preference；`lib/common_ptan/agent.py::MO_TD3_HER` 将其传入 actor/critic。
- **论文位置**：Sec. 3 要求 `w_l>=0`、`Σ_l w_l=1`，并定义线性标量化 `f_w(r)=w^T r`。
- **JAX/EvoRL 建议**：workflow 为每个并行环境维护 `w:[B,L]`，仅在该 lane episode 结束时重采样；evaluation 则使用显式给定的 preference 网格。
- **JIT/vmap 风险**：done lanes 的重采样必须使用 mask 和拆分后的 JAX PRNG key，不能用 Python 控制流；`L` 静态。
- **论文/源码不一致**：论文称从 simplex 均匀采样；源码的 episode preference 多来自离散网格，HER 使用绝对高斯归一化并四舍五入，`L>2` 时并非 simplex 上均匀分布。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。第一版固定采用源码行为：episode 从离散 `w_batch` 子集选取；HER 使用独立标准高斯取绝对值、L1 归一化并保留三位小数。论文的连续均匀 simplex 描述不覆盖源码行为。

### 2.5 transition — **EXTEND**

- **EvoRL 当前实现**：`evorl/sample_batch.py::SampleBatch` 表示 observations/actions/rewards/dones/next_observations/extras；`evorl/rollout.py::env_step` 生成 transition。
- **PD-MORL 官方源码**：`lib/common_ptan/experience.py::Experience` 包含 `state, action, reward, next_state, terminal, preference, step_idx, p_id, info`。
- **论文位置**：Algorithm 3 replay 记录 `(s,a,r,s',d,w)`；Appendix B.1.1 的 HER 只替换 `w`。
- **JAX/EvoRL 映射**：transition 的核心 shape 为 `s:[B,O]`、`a:[B,A]`、`r:[B,L]`、`done:[B]`、`s':[B,O]`、`w:[B,L]`。对 `step_idx/p_id/info` 先追踪所有官方调用者；仅确认不影响采样、HER、同步或日志后，才可作为容器差异省略。
- **JIT/vmap 风险**：extras 的 PyTree 结构必须在所有 step 固定；不能某些 transition 有 preference、某些没有。
- **论文/源码不一致**：源码保存了调度/调试字段，论文 transition 不要求这些字段。
- **映射分类**：**DEVIATION**（官方源码使用旧环境 `done`，Brax termination/truncation 对应关系尚未确认）。
- **ASSUMPTION / 待确认**：不得自行改成“仅 termination 屏蔽 bootstrap”。必须先记录官方 `done` 的实际来源、Brax 两类结束信号及 time-limit 行为，等待确认后固定 mask。

### 2.6 `SampleBatch` — **EXTEND**

- **EvoRL 当前实现**：`evorl/sample_batch.py::SampleBatch` 已是 JAX PyTree，reward 类型也允许 array；`extras` 可承载额外字段。
- **PD-MORL 官方源码**：没有同名批类型；`ExperienceReplayBuffer_HER` 将 `Experience` 列表转换为 NumPy/Torch batch，返回 states/actions/rewards/dones/next_states/preferences。
- **论文位置**：Algorithm 3 的 mini-batch 为 `(s,a,r,s',d,w)`。
- **JAX/EvoRL 映射**：不新增平行 batch 类。令 `rewards:[N,L]`，并在固定 extras 路径 `extras.policy_extras.preference` 保存 `w:[N,L]`。
- **JIT/vmap 风险**：extras key 和叶子 shape 必须静态；所有 rollout/eval 路径若共享 batch 构造器，都要提供同构 placeholder。
- **论文/源码不一致**：无算法不一致，仅数据容器不同。
- **映射分类**：**FRAMEWORK-ADAPTATION**（以 EvoRL PyTree extras 等价承载官方 `Experience.preference`）。
- **ASSUMPTION / 待确认**：无；本文固定使用 `extras.policy_extras.preference`，不支持额外别名。

### 2.7 replay buffer — **EXTEND**

- **EvoRL 当前实现**：`evorl/replay_buffers/replay_buffer.py::ReplayBuffer` 对任意固定 PyTree 进行 ring-buffer add/sample，核心无需理解 reward 含义。
- **PD-MORL 官方源码**：10 个 child processes 经 main-process queue 汇入同一个 `ExperienceReplayBuffer_HER`；该 buffer 保存所有 preference workers 的 transition，并生成 preference relabel 版本。Learner 从这一混合的全局池采样，不存在按 child/process 隔离的 replay。
- **论文位置**：Appendix B.1.1 与 Algorithm 3：每个原 transition 加入 replay，并以 `N_w` 个随机 preference relabel。
- **JAX/EvoRL 映射**：复用现有 buffer，并把它定义为一个**逻辑全局 replay pool**：所有 10 个 logical workers 的有效 transitions 都进入同一采样总体。物理实现可以是单设备集中池、各卡持有完全同步的副本，或按 global index sharding 的分布式存储；无论哪种布局，learner 的每个 sample 都必须统计等价于从全局池均匀抽取。第一阶段在全局 base pool sample 后执行 sample-time HER；后续源码忠实版在逻辑全局 add 时物理写入 relabel entries。
- **JIT/vmap 风险**：三卡上的 4/3/3 worker placement 不得转化成三个条件于本地子空间的 replay distributions。若物理 sharding buffer，必须维护全局 size/write-order/index 语义，并按 global indices 采样后路由到 shard；若复制 buffer，则必须先汇集全部有效 transitions 并保持副本一致。不得假定梯度 `pmean` 能修复输入分布偏差。
- **论文/源码不一致**：论文表述为每个 transition 都 relabel；源码仅在 replay 长度超过 `start_timesteps*process_count` 后开始加入 relabel 样本。
- **映射分类**：逻辑全局 replay pool 为 **SOURCE-FAITHFUL**；其集中、复制或 global-index sharding 的物理承载为 **FRAMEWORK-ADAPTATION**；三个相互隔离的 per-GPU replay 为 **DEVIATION**。第一阶段 sample-time relabel 仍是另一个已批准的 **DEVIATION**。
- **ASSUMPTION / 待确认**：无。第一阶段 deviation 已获明确采用；源码忠实版是完整框架完成后的必做消融。

> **硬性实现约束：禁止默认 per-GPU isolated replay。** 即使模型梯度最终执行 `pmean`，若 GPU 0/1/2 分别只从自己承载的 preference subspaces 采样，三个 learner 输入已经是不同的条件分布，无法等价于官方 main-process replay。后续 unit test 必须给 10 个 logical workers 写入可辨识的 worker IDs，证明单卡与三卡 sample 均能覆盖全部 `0..9`，并验证长期采样频率与逻辑全局池中的条目占比一致。

## 3. 网络、目标与损失

### 3.1 actor input — **EXTEND**

- **EvoRL 当前实现**：`evorl/networks/linear.py::make_policy_network` 输入 observation；由 `evorl/algorithms/td3.py::make_td3_agent` 创建。
- **PD-MORL 官方源码**：`lib/models/networks.py::Actor.forward` 执行 `torch.cat([state, preference], dim=1)`。
- **论文位置**：Algorithm 3，`a=π_φ(s,w)`。
- **JAX/EvoRL 建议**：actor 的最后输入维为 `O+L`，在网络调用边界 `jnp.concatenate([obs,w],-1)`；不改变 Brax obs space。
- **JIT/vmap 风险**：低；需避免单样本 `[L]` 与 batch `[B,L]` 的隐式广播歧义。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。严格采用源码的直接拼接，不增加 preference encoder、FiLM 或 hypernetwork。

### 3.2 actor output — **UNCHANGED**

- **EvoRL 当前实现**：policy 输出 `[...,A]` 连续动作。
- **PD-MORL 官方源码**：Actor 输出 `[batch,action_size]`，经 `tanh*max_action`。
- **论文位置**：Algorithm 3 的 actor 对每个 `(s,w)` 产生一个 action。
- **JAX/EvoRL 建议**：保持 `actor(s,w): [B,O]×[B,L] -> [B,A]`。
- **JIT/vmap 风险**：无新增风险。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无；确定性 actor 与网络外探索噪声来自源码。

### 3.3 critic input — **EXTEND**

- **EvoRL 当前实现**：`evorl/networks/linear.py::make_q_network` 输入 `(obs,action)`；TD3 使用两个 critic。
- **PD-MORL 官方源码**：`lib/models/networks.py::Critic.forward` 拼接 state、preference、action。
- **论文位置**：Algorithm 3，`Q_{θ_i}(s,w,a)`。
- **JAX/EvoRL 建议**：critic 输入末维为 `O+L+A`；保持 EvoRL 现有 twin critic 的批量参数组织。只有现有网络实现确实以单 critic 函数表达 twin 轴时，才沿用其局部变换方式。
- **JIT/vmap 风险**：低；确保 critic-index 轴和 batch 轴不混淆。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**（state、preference、action 直接拼接及无独立 preference 分支）；若 MLP 宽度/激活沿用 EvoRL 而不等同源码，该参数差异必须单独标为 **DEVIATION**。
- **ASSUMPTION / 待确认**：需要从目标任务训练脚本提取官方 hidden sizes、activation 和初始化，并与 EvoRL 配置逐项对照；不得默认沿用 EvoRL 网络超参数。

### 3.4 critic output — **REPLACE**

- **EvoRL 当前实现**：TD3 critic 每个样本、每个 critic 输出标量，组合后近似 `[B,2]`。
- **PD-MORL 官方源码**：`lib/models/networks.py::Critic` 的两个 head 各输出 `reward_size=L`。
- **论文位置**：Sec. 4.1 的 vector Q 定义；Algorithm 3 明确 critic 输出 `L` 个 Q 值。
- **JAX/EvoRL 建议**：统一布局为 `Q:[B,2,L]`，避免把目标轴与 twin 轴压平。
- **JIT/vmap 风险**：loss reduction 必须明确：先对 `L` 处理、再对 twin/batch 取均值；旧 `min(axis=critic)` 不能直接用于向量。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无；两个 critic 各输出完整 `L` 维向量，共用源码所示网络组织。

### 3.5 vector Q — **NEW**

- **EvoRL 当前实现**：无；`evorl/algorithms/td3.py::_critic_loss` 处理 scalar Q。
- **PD-MORL 官方源码**：`Critic.forward/Q1` 返回 reward-size 向量；`MO_TD3_HER.learn` 对向量 target 回归。
- **论文位置**：Sec. 4.1 Eq. (3)–(5) 与 Algorithm 3。
- **JAX/EvoRL 建议**：定义 `Q_i(s,w,a)∈R^L`；全程保持目标维，不在 Bellman target 前提前 scalarize。
- **JIT/vmap 风险**：静态 `L` 无风险；易错点是广播 `done:[B]` 到 `[B,L]`。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无；vector-Q 语义与共享 `gamma` 从源码 target 直接提取。

### 3.6 scalarization `w^T Q` — **NEW**

- **EvoRL 当前实现**：原 TD3 的 Q 已是标量，无 scalarization。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 用 batch matrix multiply 计算原始 `w` 与两个 target Q、actor Q 的内积。
- **论文位置**：Sec. 3 的 `f_w(r)=w^T r`；Algorithm 3 的 critic 选择与 actor objective。
- **JAX/EvoRL 建议**：用 `jnp.sum(w*Q,axis=-1)`，保持 batch/twin 前缀；target critic 比较和 actor 主目标均使用**原始 `w`**，不是 `w_p`。
- **JIT/vmap 风险**：低；禁止 `jnp.dot` 在高阶 batch 上造成错误轴收缩。
- **论文/源码不一致**：无实质不一致。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无；只实现源码实际使用的 `w^TQ`，不增加非线性 utility。

### 3.7 TD target — **REPLACE**

- **EvoRL 当前实现**：`evorl/algorithms/td3.py::_critic_loss` 取两个 scalar target critic 的逐样本最小值，再构造 scalar target。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 先计算 `w^TQ'_1`、`w^TQ'_2`，按较小标量选中一个 critic 的**完整 Q 向量**，再构造向量 target。
- **论文位置**：Algorithm 3：target action 加 smoothing noise；`y=r+γ(1-d)arg_Q min_i w^TQ_{θ'_i}(s',w,a')`。
- **JAX/EvoRL 映射**：得到 `Q':[B,2,L]`，以 `argmin` scalarized score 得 `idx:[B]`，用 `take_along_axis` 选择整向量 `[B,L]`，然后使用经确认与官方 `done` 等价的 mask 构造 `y=r+γ(1-done_equiv)Q'_selected`。绝不能对目标维做 elementwise twin-min。
- **JIT/vmap 风险**：索引 shape 容易出错，但完全可 JIT；用固定 gather，不用 Python 循环。
- **论文/源码不一致**：论文的 `arg_Q min` 记法不严谨，源码清楚地选择整条 critic 向量；应以源码和 TD3 pessimistic-selection 意图为准。
- **映射分类**：critic 选择和 vector Bellman target 为 **SOURCE-FAITHFUL**；结束信号映射当前为 **DEVIATION**，直至 Brax/旧环境语义获确认。
- **ASSUMPTION / 待确认**：bootstrap mask 不预设为 termination-only；按 2.5 节等待官方 `done` 与 Brax 信号对照确认。

> **后续 Step 2 的硬性实现约束：禁止 elementwise twin-min。** 对 `Q':[B,2,L]` 必须先分别计算 `w^TQ'_1` 与 `w^TQ'_2`，再按较小标量选择其中一个 critic 的完整 `L` 维向量。禁止使用 `jnp.minimum(Q1,Q2)` 或沿 critic 轴直接 `min`，因为这会拼出一个可能不属于任一 critic 的虚构 Q vector，并改变 PD-MORL 的 TD target。

后续实现必须提供独立 unit test。测试至少构造一个“两个 critic 在不同目标维交叉大小”的样本，例如 `Q1=[1,10]`、`Q2=[5,2]`，再选择一个能明确决定 scalarized winner 的 `w`；断言输出严格等于 `Q1` 或 `Q2` 的完整向量，同时断言输出不等于 elementwise minimum `[1,2]`。测试还应覆盖 batch 中不同样本分别选择不同 critic，并在 `jax.jit` 下得到相同结果。

### 3.8 critic loss — **REPLACE**

- **EvoRL 当前实现**：`evorl/algorithms/td3.py::_critic_loss` 为两个 scalar critic 的 squared TD error。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 的实际代码是 `mean(g(w_p,Q_1)) + smooth_l1(Q_1,y) + mean(g(w_p,Q_2)) + smooth_l1(Q_2,y)`；critic 的 angle terms **没有乘 `actor_loss_coeff`**。
- **论文位置**：Appendix B.1.5 Eq. (11)；正文说明 MSE，也允许 Huber loss。
- **JAX/EvoRL 映射**：严格复现 `L_Q=mean_b g(w_p,Q_1)+SmoothL1(Q_1,y)+mean_b g(w_p,Q_2)+SmoothL1(Q_2,y)`。角度按每个样本/critic 的完整向量求，不对各目标分别求角；不得把 actor 的 `actor_loss_coeff` 加到 critic angle terms。
- **JIT/vmap 风险**：角度在零向量和 `acos(±1)` 附近梯度不稳定；需要 norm epsilon 和与源码一致的 cosine clamp。
- **论文/源码不一致**：公式写 squared error并以统一系数表达方向项，源码使用 Smooth L1，且 critic angle terms 不乘 `actor_loss_coeff`；实现以源码逐操作行为为准。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。必须逐操作复现源码 `F.smooth_l1_loss` 默认 mean reduction 与 angle mean，并用人工小张量 unit test 对齐 PyTorch 数值；不得自行选择 sum/mean或添加 critic angle coefficient。

### 3.9 actor loss — **REPLACE**

- **EvoRL 当前实现**：`evorl/algorithms/td3.py::_actor_loss` 最大化 critic 1 的 scalar Q。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 使用 `-mean(w^TQ_1)+α mean(g(w_p,Q_1))`。
- **论文位置**：Appendix B.1.5 Eq. (12)。
- **JAX/EvoRL 建议**：只通过 critic 1 求 policy gradient：`L_π=-E[w^TQ_1(s,w,π(s,w))]+αE[g(w_p,Q_1)]`；critic 参数在该梯度路径中视为常量参数输入，不更新。
- **JIT/vmap 风险**：角度稳定性同上；必须确保 actor step 不意外把梯度应用到 critic optimizer。
- **论文/源码不一致**：论文角度公式存在 `w`/`w_p` 记号歧义；源码明确用 interpolated `w_p`。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。系数从目标任务官方 settings 读取；Hopper 源码值为 10，不自行调参或硬编码到网络。

### 3.10 target actor — **EXTEND**

- **EvoRL 当前实现**：`TD3NetworkParams.target_actor_params`，由 `soft_target_update` 更新；critic target action 在 `_critic_loss` 中产生。
- **PD-MORL 官方源码**：`actor_target` 是 actor 的深拷贝，输入 `(s',w)`。
- **论文位置**：Algorithm 3 的 `π_{φ'}(s',w)`。
- **JAX/EvoRL 建议**：复用参数树与 Polyak helper，仅将 target actor 调用扩为 `(s',w)`。
- **JIT/vmap 风险**：无新增风险。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**；复用 EvoRL 参数树/Polyak helper 属于等价的 **FRAMEWORK-ADAPTATION**，实现时分开标注。
- **ASSUMPTION / 待确认**：无。

### 3.11 target critics — **EXTEND**

- **EvoRL 当前实现**：`target_critic_params` 保存 twin scalar critics，并由 `soft_target_update` 更新。
- **PD-MORL 官方源码**：`critic_target` 是 vector twin critics 的深拷贝。
- **论文位置**：Algorithm 3 的两个 `Q_{θ'_i}`。
- **JAX/EvoRL 建议**：复用 twin 参数组织和 Polyak 更新；输出布局变为 `[B,2,L]`。
- **JIT/vmap 风险**：无结构性风险；初始化 dummy output shape 必须与在线 critic 一致。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**；以 EvoRL helper 执行相同 Polyak 公式属于 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无；`tau` 从官方目标任务配置提取。

## 4. TD3 时序机制

### 4.1 exploration noise — **UNCHANGED**

- **EvoRL 当前实现**：`TD3Agent.compute_actions` 在 online actor action 上加高斯噪声并裁剪。
- **PD-MORL 官方源码**：`MO_TD3_HER.__call__` 使用 `std=max_action*expl_noise` 的高斯噪声并裁剪。
- **论文位置**：Algorithm 3 数据收集动作 `a=π_φ(s,w)+ε`。
- **JAX/EvoRL 建议**：保留现有 JAX PRNG、Gaussian noise 和 action clipping；噪声不依赖 `w`。
- **JIT/vmap 风险**：每个 env lane 必须获得独立 key；不能复用同一噪声向量。
- **论文/源码不一致**：无。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：不得沿用 EvoRL 默认 scale；必须从目标任务官方 settings 提取 `expl_noise`，并在 action bounds 对照获确认后按源码公式应用。

### 4.2 target smoothing noise — **UNCHANGED**

- **EvoRL 当前实现**：`_critic_loss` 在 target actor action 上加 clipped noise 并裁剪动作。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 同样使用 `policy_noise`、`noise_clip` 和 action clip。
- **论文位置**：Algorithm 3 的 target-policy smoothing。
- **JAX/EvoRL 建议**：原逻辑保持，target actor 的输入增加 `w`。
- **JIT/vmap 风险**：低；噪声 shape 必须等于 `[B,A]`，不能多出 critic/目标轴。
- **论文/源码不一致**：源码为 action clamp 做了 CPU/NumPy 往返；这是框架写法，不是算法要求，JAX 版必须使用 `jnp.clip`。
- **映射分类**：噪声分布与 clipping 顺序为 **SOURCE-FAITHFUL**；以 `jnp.clip` 替代源码 NumPy/CPU 往返为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：目标 Brax action bounds 必须先与官方 `max_action` 对照；未确认前不实现边界映射。

### 4.3 delayed policy update — **UNCHANGED**

- **EvoRL 当前实现**：`evorl/algorithms/td3.py::TD3Workflow.step` 先做 critic 更新，再按 `actor_update_interval` 条件更新 actor，并在同一分支 soft-update targets。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 仅当 `total_it % policy_freq == 0` 时更新 actor 和全部 target networks。
- **论文位置**：Algorithm 3 将 actor 与 target soft update 放在 delayed 分支；Appendix B.1.5 说明 actor/target 更新较慢。
- **JAX/EvoRL 建议**：保持 EvoRL 的 delayed update 结构；只替换分支内 loss。更新频率作为任务配置。
- **JIT/vmap 风险**：现有 `lax.cond`/scan 模式可保持；两个分支 PyTree shape 必须一致。
- **论文/源码不一致**：Appendix 一处文字可读成 target 每 step 更新，但 Algorithm 3 与源码均在 delayed actor 分支更新，应以二者为准。
- **映射分类**：**SOURCE-FAITHFUL**；使用 EvoRL 现有 delayed branch 和 Polyak helper 承载同一时序为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无。不得使用 EvoRL 默认 interval=2；必须从所选官方任务 settings 读取 `policy_freq`（Hopper 源码为 20，其他任务逐文件核对）。

#### 官方 gradient clipping 约束

官方 `lib/common_ptan/agent.py::MO_TD3_HER.learn` 明确使用 gradient clipping，不能删除或自行猜测：

| 更新 | 官方位置 | 被裁剪参数 | 时机 | 阈值/范数 |
|---|---|---|---|---|
| critic | `agent.py:353` | `self.critic.parameters()`，即同一 Critic 模块中的 Q1 与 Q2 全部参数 | `critic_loss.backward()` 后、`critic_optimizer.step()` 前；每次 critic update 都执行 | `clip_grad_norm_(..., max_norm=100)`；未传 `norm_type`，使用 PyTorch 默认 total L2 norm |
| actor | `agent.py:374` | `self.actor.parameters()` | `actor_loss.backward()` 后、`actor_optimizer.step()` 前；仅在 `total_it % policy_freq == 0` 的 delayed actor branch 执行 | `clip_grad_norm_(..., max_norm=100)`；未传 `norm_type`，使用 PyTorch 默认 total L2 norm |

- **EvoRL/JAX 映射**：在 actor 与 critic 的 optimizer chain 中分别使用 `optax.clip_by_global_norm(100)`，且 placement 必须位于参数 update 之前。不得裁剪 loss、action noise、Q values、target parameters 或 actor/critic 合并后的共同 norm。
- **多 GPU 约束**：learner 梯度先按 EvoRL data-parallel 语义聚合，再分别对全局 critic gradient PyTree 和 actor gradient PyTree执行 global-norm clipping；不能每卡以不同 local replay gradient 独立裁剪后再用 `pmean` 掩盖分布差异。
- **映射分类**：clip 对象、时机、L2 global norm 与阈值 100 为 **SOURCE-FAITHFUL**；用 Optax 承载同一操作为 **FRAMEWORK-ADAPTATION**。
- **unit test**：分别构造 actor/critic gradient PyTree，使总 L2 norm 大于 100；断言裁剪后 norm 约为 100、方向不变、另一个网络参数不参与该 norm，并验证 `jax.jit` 前后一致。

## 5. Preference、并行采样、HER 与几何约束

### 5.1 preference sampling — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：`MORLExperienceSource` 从 `w_batch` 选 episode preference；HER 用 `abs(randn)` 后 L1 归一化并保留三位小数；`MORL_utils.generate_w_batch_test` 生成离散 simplex 网格。
- **论文位置**：Sec. 3 与 Appendix B.1.1–B.1.2 称 preference 从 simplex/子空间均匀采样。
- **JAX/EvoRL 映射**：严格区分并复现源码的两类 sampler。**Episode sampler**：预先生成与源码一致的离散 simplex `w_batch`，按 preference subspace 分配后，从对应子集中随机选一行，并在 episode 内保持不变。**HER sampler**：在 device 上生成独立 standard-normal 向量，取绝对值、按 L1 归一化，再按源码保留三位小数。不实现源码中不存在的 `Dirichlet(1)`/exponential-normalize 连续均匀 sampler。
- **JIT/vmap 风险**：离散网格及每个 subspace 的索引范围必须静态；随机选行、absolute-normal、L1 normalize 和 rounding 全部使用 JAX device 运算，避免 host NumPy RNG。极小概率的近零 L1 norm 需要 epsilon 保护；rounding 后应检查各分量和是否仍满足源码容差，而不要静默二次归一化并改变源码分布。
- **论文/源码不一致**：连续均匀、离散网格、absolute-normal 三种行为并不相同，且源码 rounding 改变分布。
- **映射分类**：**SOURCE-FAITHFUL**；使用 JAX PRNG/device arrays 实现相同分布为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无。`weight_num`、网格顺序与分配规则均从目标任务官方 settings/训练脚本读取；不加入论文均匀 sampler。

### 5.2 preference-space subdivision — **NEW**

- **EvoRL 当前实现**：无 preference space。
- **PD-MORL 官方源码**：各 `train_*_MO_TD3_HER.py` 生成 preference 网格，并用 `np.array_split` 分给 `process_count` 个 child processes。
- **论文位置**：Appendix B.1.2：将 preference space 分成 `C_p=10` 个不重叠子空间，每个 process 在其子空间采样。
- **JAX/EvoRL 映射**：按源码顺序生成离散 `w_batch`，先在与设备无关的逻辑层执行等价于 `np.array_split(w_batch,C_p)` 的静态分段，其中 `C_p=10`。建立固定的一一映射 `logical_worker_id 0..9 → preference_subspace 0..9`；episode 结束时，每个逻辑 worker 只从自己的离散子集选一行。完成逻辑分割后才决定 worker 的 GPU placement。
- **JIT/vmap 风险**：固定 worker→subspace 和离散索引表易于 JIT；不同子集长度可能相差 1，可使用 padding+valid-count 或静态 start/end 索引。设备数绝不能参与 `array_split`。三卡标准均匀 sharding 无法直接整除 10 个逻辑 workers 时，可使用 12 个物理 slots（每卡 4 个）和 2 个 inactive masks，或其他等价的 4/3/3 placement；inactive slots 不得拥有新 subspace、产生 transition、计入指标或改变更新次数。
- **论文/源码不一致**：源码只是对按字典序排列的离散网格 `array_split`；这不保证一般高维 simplex 上的几何等体积或严格局部邻域。
- **映射分类**：**SOURCE-FAITHFUL**；以静态 JAX 索引替代 NumPy `array_split` 为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无。`C_p=10` 固定来自源码；GPU 数量不是该配置的输入。只复现源码离散分割，不新增高维几何分区算法。

### 5.3 parallel child processes — **REPLACE**

- **EvoRL 当前实现**：Brax 已通过 batched env、JAX transform 与 jitted workflow 并行采样，不需要 Python child process。`evorl/workflows/rl_workflow.py` 已支持 `enable_multi_devices`、`NamedSharding`/device mesh 和逐设备 PRNG；`evorl/distributed/` 提供 `pmean`、`psum`、`all_gather` 等通信原语，TD3 也已用 `pmean/psum` 聚合指标与计数。
- **PD-MORL 官方源码**：训练脚本用 `torch.multiprocessing`、共享 actor/critic、`JoinableQueue(maxsize=1)` 和 10 个 CPU child processes；每轮收集 10 个 transitions，再做 10 次 learner updates。
- **论文位置**：Appendix B.1.2 描述 `C_p` child processes、周期同步到 main process。
- **JAX/EvoRL 映射**：保留固定 10 个逻辑 preference workers，替换的只是执行载体。单卡时一张 GPU 承载全部 10 个 workers；三卡时将同一组 worker IDs 映射为 4/3/3，或用每卡 4 slots 加 inactive mask 表达。各设备产生的有效 transitions 必须先进入同一个逻辑全局 replay pool；之后 learner 才从全局混合分布取得 minibatch，并在设备轴上分片计算。模型参数复制，梯度通过现有 `pmean` 同步；每个逻辑 worker使用独立 PRNG、episode state 和官方 subspace。
- **JIT/vmap 风险**：不要在 jitted step 中建立队列、进程或可变共享状态。设备 placement 可因 1/3 GPU 改变，但逻辑 worker axis、10 个 subspace IDs、有效 transition 数、逻辑全局 replay 内容和全局采样/更新比必须不变。若使用 padded physical slots，rollout/replay add/loss/metrics 全部应用 valid mask。Replay batch 可以在全局采样**之后**沿 data-parallel 轴分片；禁止先按设备隔离 replay 再各自 local-sample。
- **论文/源码不一致**：无算法不一致；源码并行机制是 PyTorch/CPU 工程选择，不是 PD-MORL 必要组成。
- **映射分类**：**FRAMEWORK-ADAPTATION**（10 个 CPU child workers 转换为同样 10 个逻辑 Brax workers，再部署到 1 或 3 GPU）。
- **ASSUMPTION / 待确认**：无。`C_p=10` 与 GPU 数彻底解耦；源码“每轮 10 个 logical-worker transitions 后做 10 次 updates”的有效采样/更新比必须复现。仅物理 placement 和 wall-clock 并发顺序允许因 JAX batching/sharding 改写。

> **硬性实现约束：`process_count` 不是 `device_count`。** 禁止令 `C_p=jax.device_count()`，禁止按 GPU 数重新执行 `np.array_split`，也禁止三卡运行时把 preference space 改成 3 个子空间。逻辑划分始终是 `np.array_split(w_batch,10)`；设备层只保存 `logical_worker_id → device_id/local_slot` 映射。后续 unit test 必须比较单卡与三卡的 10 份 subspace 内容完全相同，验证 worker IDs `0..9` 各出现一次，并确认 inactive physical slots 不进入 replay、loss、metrics 或采样/更新计数。

> **硬性实现约束：data-parallel batch sharding 发生在全局 replay sampling 之后。** “三卡 replay batch 分片”只表示把一个来自逻辑全局池的 minibatch 分给三张卡计算；不表示三张卡各自拥有互不交换的 replay。任何 local-only replay 方案必须标为 **DEVIATION**，并且不属于默认实现。

### 5.4 HER / preference relabeling — **NEW**

- **EvoRL 当前实现**：无 preference relabeling；replay sample 原样进入 TD3 loss。
- **PD-MORL 官方源码**：`ExperienceReplayBuffer_HER._add` 对同一 `(s,a,r,s',done)` 添加若干随机 preference 版本，仅替换 `preference`。
- **论文位置**：Appendix B.1.1：原 transition 加 `N_w` 个随机 preference relabel；Algorithm 3 的 replay 写入步骤。
- **JAX/EvoRL 建议**：分两阶段实现。**GPU 简化版（第一阶段）**：采样 base batch 后生成 `w_relabel:[N,N_w,L]`，广播物理 transition，拼上原 `w` 后展平到 `[N(1+N_w),...]`；replay 中只存 base transition。**源码忠实版（后续消融）**：收集时先写原 transition，达到源码 warm-up 条件后，再为每条 transition 生成 `N_w` 个 preference 并作为独立条目物理写入 replay。两版都保持 reward、state、action、next state 和 done 不变，只替换 `w`。
- **JIT/vmap 风险**：GPU 简化版的 `N_w`/展平倍数必须静态；有效 batch 增大会改变编译内存和 loss scale，应在 config 明确 base batch 与有效 batch。源码忠实版不扩张单次 sample shape，但会更快占满 replay，并改变设备内存、写指针和样本年龄分布；物理写入仍应使用固定 shape 的批量 add，而不是 Python 循环。
- **论文/源码不一致**：源码有 warm-up 门槛，且不是严格 uniform sampler；论文没有这两个限定。
- **映射分类**：第一阶段 sample-time relabel 为已批准的 **DEVIATION**；后续写入时复制、warm-up 门槛及 preference-only relabel 为 **SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。两阶段及消融控制变量已固定；不得把它改造成 goal-state HER，也不得重算 reward/next state。

#### HER 两种实现的实验边界

| 项目 | GPU 简化版（第一阶段） | 源码忠实版（框架完成后消融） |
|---|---|---|
| relabel 时机 | replay sample 后 | transition 写入 replay 时 |
| replay 内容 | 只存 base transitions | 存原 transition 与独立 relabeled entries |
| warm-up | 无额外 HER warm-up | 复现 `len(buffer) > start_timesteps * process_count`，其中 `process_count` 固定为逻辑 `C_p=10`，与 GPU 数无关 |
| learner 输入 | `[N(1+N_w),...]` 固定展开 | 从已混合 replay 采样 `[N,...]` |
| replay 容量消耗 | 每个环境 step 占 1 条 | warm-up 后每个环境 step 最多占 `1+N_w` 条 |
| 主要偏差 | 同一 base sample 的 relabels 在一次 update 内相关 | 样本年龄、覆盖/淘汰概率与源码一致 |
| 用途 | 先闭合 JAX/GPU 数据链 | 判断简化是否改变算法结果 |

### 5.5 interpolator `I(w)` — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：训练脚本用 SciPy `RBFInterpolator(..., kernel='linear')`；key-preference 预训练脚本生成 key objective solutions。在线 evaluation 仅当 key preference 下以原 preference 标量化的 return 改善时替换对应 objective solution，随后按脚本的归一化行为重建 interpolator。
- **论文位置**：Appendix B.1.3：用 key preferences 的近似 Pareto 解建立 RBF interpolator，并在训练中更新。
- **JAX/EvoRL 建议**：划定训练数据面与低频控制面的边界。**JIT 内调用**：将固定数量的 knots、目标方向和线性 RBF 系数作为普通 JAX arrays 保存于 workflow state；training step 只执行固定参数、固定 shape、纯 JAX 的前向 `w_p=I(w)`。**JIT 外重拟合**：在 evaluation 或显式 host boundary 收集 key-preference returns，判断是否更新 key solutions，并用 SciPy 或独立的 JAX 线性代数过程低频重拟合；拟合完成后一次性把同 shape 的新参数写回 workflow state，供下一段已编译训练使用。不要使用 host callback，也不要求一个 training-step JIT 同时负责拟合和调用。
- **JIT/vmap 风险**：SciPy/Python callable 不能被 training JIT trace；在 step 内改变 knot 数量、Python 对象或参数 PyTree 结构会触发重编译。即使重拟合使用 JAX，也应留在独立 evaluation/control boundary。JAX 前向/拟合必须用固定 knots 与查询点逐值对照 SciPy `RBFInterpolator(kernel='linear')`；未通过容差测试前不能标为等价适配。若参数 shape 不变，仅更新数值通常无需重新编译。
- **论文/源码不一致**：源码初始化 key objectives 时做 L2 normalize，在线更新路径却做 L1 normalize；论文写“unit vector”，更接近 L2。
- **映射分类**：RBF kernel、key solutions 的生成/更新条件和调用语义必须为 **SOURCE-FAITHFUL**；纯 JAX 固定-shape 前向与 host-boundary 重拟合为 **FRAMEWORK-ADAPTATION**。统一修正源码初始化 L2/在线 L1 会构成 **DEVIATION**，本计划不实施。
- **ASSUMPTION / 待确认**：不得自行选择 key solutions 来源；按官方流程先运行 fixed-preference key training 并加载其产物。初始化采用源码 L2、在线更新采用源码 L1。

#### Interpolator 执行边界

| 边界 | 允许操作 | 禁止操作 |
|---|---|---|
| training-step JIT 内 | 固定参数 `I(w)` 前向；batched distance/kernel；输出 `w_p:[B,L]` | SciPy 调用、host callback、增删 knots、动态 shape、根据 Python 条件重建对象 |
| evaluation / host boundary | 评估 key preferences、更新 key solutions、低频重拟合、记录拟合诊断 | 在环境每 step 或每个 gradient step 都重拟合 |
| 返回训练前 | 将同结构、同 shape 的新 knots/coefficients 写回 workflow state | 改变 PyTree schema 或让设备持有不可序列化 Python interpolator |

### 5.6 projected preference `w_p` — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 调用 interpolator，将 batch preference 映射为 `w_p`，再用于 cosine/angle loss。
- **论文位置**：Appendix B.1.3 与 B.1.5，`w_p=I(w)`。
- **JAX/EvoRL 映射**：直接使用官方 interpolator 输出 `w_p:[B,L]`；不要在调用后额外 L2 normalize，因为源码把 `self.interp(w)` 直接传给 `F.cosine_similarity`。`w_p` 仅用于方向角，TD target critic 选择和 `w^TQ` 必须继续用原 `w`。
- **JIT/vmap 风险**：插值结果可能接近零；`F.cosine_similarity` 的等价实现需在范数分母使用与 PyTorch 默认行为一致的 epsilon，但不得借此额外 normalize 或改写 `w_p`。所有 knots/coefficients shape 固定。
- **论文/源码不一致**：归一化范数在源码不同路径不一致。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无；`w_p` 的数值处理严格跟随官方 interpolator 输出及 angle 调用，不额外投影到 simplex。

### 5.7 cosine similarity `S_c` — **NEW**

- **EvoRL 当前实现**：无多目标向量相似度。
- **PD-MORL 官方源码**：连续 `MO_TD3_HER` 没有单独实现 Eq. (5) 的 preference-driven optimality operator，但角度 loss 内调用 cosine similarity。
- **论文位置**：Sec. 4.1 Eq. (5) 用 `S_c(w,Q)` 选择 MO-DDQN 动作；Appendix B.1.5 的连续扩展改为方向角约束。
- **JAX/EvoRL 建议**：新增最小的 batched cosine 原语供 `g` 使用即可；不要把 Eq. (5) 的离散动作选择规则额外塞进 TD3 target。
- **JIT/vmap 风险**：零范数需 epsilon；对 cosine 的 clamp 需保证梯度定义。
- **论文/源码不一致**：论文的 `S_c` 是离散算法核心；官方连续 TD3 只间接通过 angle 使用 cosine。二者不应混为同一 target 规则。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。官方连续 MO-TD3 路径未使用 MO-DDQN Eq. (5) operator，因此不新增该算法模块；只实现源码 angle 所调用的 cosine。

### 5.8 directional angle `g(w_p,Q)` — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：`MO_TD3_HER.learn` 使用 `rad2deg(acos(clamp(cosine_similarity(w_p,Q),0,0.9999)))`，同时加入 actor 和两个 critic loss。
- **论文位置**：Appendix B.1.5 Eq. (10)–(12)。
- **JAX/EvoRL 建议**：按 source-faithful 版本实现 batched angle，带 norm epsilon 与 `[0,0.9999]` clamp；先保持“度”单位以对齐 `α=10` 的损失尺度。
- **JIT/vmap 风险**：`acos` 在 1 附近梯度发散；clamp 是必要数值保护。负 cosine 被截为 0 会丢失大于 90° 的区别，但这是官方行为。
- **论文/源码不一致**：Eq. (10) 左侧写 `g(w_p,Q)`，分子却印成 `w^TQ`；源码明确使用 `w_p`。论文给原始弧度公式，源码转成角度且 clamp。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。固定复现 degree conversion 与 `[0,0.9999]` clamp；不得改成弧度或其他 clamp，也不自行重标定 `α`。

## 6. Evaluation 与 Pareto 指标

### 6.1 evaluation — **REPLACE**

- **EvoRL 当前实现**：`evorl/evaluators/evaluator.py::Evaluator.evaluate` 汇总标量 episode return/length；TD3 workflow 定期调用。
- **PD-MORL 官方源码**：`lib/utilities/MORL_utils.py::eval_agent`、`eval_agent_test`、`eval_agent_interp` 对 preference 网格逐个运行确定性策略，累计 vector return；`eval_benchmarks_MO_TD3_HER.py` 汇总多次评估。
- **论文位置**：Sec. 3 的 HV/sparsity 定义、Sec. 5 实验、Appendix B.2。
- **JAX/EvoRL 建议**：evaluation 接收固定 `W_eval:[M,L]`，对每个 preference 和 seed 运行 deterministic actor，输出 `returns:[M,S,L]`；优先把 preference/seed 轴并入 Brax 原生 batch 后统一 JIT。只有尚未 batch 化的纯函数边界才局部使用 `vmap`。
- **JIT/vmap 风险**：不同 episode 长度用 done mask；`M*S` 过大时分固定 chunk，避免动态循环和显存峰值。
- **论文/源码不一致**：源码常在 CPU 串行遍历 preference；论文不要求该执行方式。
- **映射分类**：evaluation 网格、确定性动作、episode return 聚合与重复次数为 **SOURCE-FAITHFUL**；用 Brax batch/JIT 执行相同评估为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无。网格、seed/repeat 数、undiscounted vector return 和聚合顺序必须从官方评估脚本逐项提取，不自行选择。

### 6.2 Pareto front construction — **NEW**

- **EvoRL 当前实现**：无多目标 non-dominated filtering。
- **PD-MORL 官方源码**：`eval_benchmarks_MO_TD3_HER.py` 对 mean vector returns 做 non-dominated sorting；`MORL_utils.py` 中指标函数也先处理非支配点。
- **论文位置**：Sec. 3 的 Pareto set/front 背景与实验图。
- **JAX/EvoRL 建议**：对最大化目标，点 `x` 被支配当且仅当存在 `y` 满足 `y>=x` 且至少一维 `y>x`。首版在 evaluation 后用固定 `[M,M,L]` 比较生成 mask；无需进入 training JIT。
- **JIT/vmap 风险**：压缩成动态长度数组会破坏静态 shape；在 JAX 内保留 `[M]` mask，写文件/画图时再 host 端压缩。`O(M^2L)` 对小网格足够。
- **论文/源码不一致**：源码借助 pymoo 的最小化约定，对 return 取负；数学上与最大化筛选等价。
- **映射分类**：**SOURCE-FAITHFUL**；若以 JAX fixed mask 等价替代官方 non-dominated-sort 依赖，则为 **FRAMEWORK-ADAPTATION**，必须用相同输入做结果一致性测试。
- **ASSUMPTION / 待确认**：无。重复点、比较符号、排序和 tolerance 均遵循官方依赖的实际行为，不自行增加 tolerance 或预去重。

### 6.3 hypervolume — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：`MORL_utils.py::eval_agent` 使用 pymoo HV，对 returns 取负，并以零向量为 reference point。
- **论文位置**：Sec. 3 Eq. (1) 定义 hypervolume。
- **JAX/EvoRL 映射**：指标放在 evaluation/postprocess，不进入训练 step。可继续在明确的 host evaluation boundary 使用官方 pymoo 计算路径；若改成二目标 JAX/NumPy 精确面积算法，只能作为 **FRAMEWORK-ADAPTATION**，且必须与 pymoo 在固定测试前沿上逐值对齐。不得自行更换 reference point、目标符号或预处理。
- **JIT/vmap 风险**：排序和动态 Pareto 数量不适合核心 training JIT；用 fixed mask 或 host postprocess。二目标静态排序可 JIT，但没有必要。
- **论文/源码不一致**：论文是一般 reference point；源码默认正向最大化目标且 reference 为零，经取负适配 pymoo。
- **映射分类**：官方取负和零 reference 的评估行为为 **SOURCE-FAITHFUL**；等价指标实现为 **FRAMEWORK-ADAPTATION**。
- **ASSUMPTION / 待确认**：无算法假设；第一版严格使用目标任务官方评估脚本的符号和 reference。若 Brax reward 映射导致 reference 不再具有相同含义，该环境整体映射标为 **DEVIATION** 并停止等待确认，不自行改 reference。

### 6.4 sparsity — **NEW**

- **EvoRL 当前实现**：无。
- **PD-MORL 官方源码**：`lib/utilities/MORL_utils.py::compute_sparsity` 对 Pareto 点逐目标排序，累计相邻差的平方并除以 `N-1`。
- **论文位置**：Sec. 3 Eq. (2)。
- **JAX/EvoRL 建议**：严格按 Eq. (2) 对 non-dominated returns 计算；指标越低表示覆盖越均匀。与 HV 一样放在 evaluation 后处理。
- **JIT/vmap 风险**：masked sort 在 JIT 中复杂，因此放在 host evaluation boundary 即可。源码在 non-dominated 点数 `N<=1` 时返回 sparsity `0`，JAX/NumPy 等价实现必须保持该行为。
- **论文/源码不一致**：源码有时对取负后的 Pareto 点计算；平方间隔对整体符号翻转不敏感，因此结果等价。
- **映射分类**：**SOURCE-FAITHFUL**。
- **ASSUMPTION / 待确认**：无。严格保持官方未归一化公式、排序和 reduction；不增加 objective-range normalization。

## 7. 关键 shape 与一次训练 step 数据流

设逻辑 preference-worker 数为固定 `C_p=10`，物理 GPU 数为 `D∈{1,3}`，有效并行环境 batch `B=C_p`；从 replay 采样的原始 batch 为 `N`，目标数为 `L`，observation 维为 `O`，action 维为 `A`，HER 数为 `N_w`。若三卡采用均匀 physical sharding，可令物理 slot 数 `P=ceil(C_p/D)D=12`，并用 `[P]` valid mask 保证只有前述 10 个逻辑 workers 生效。

| 张量 | shape |
|---|---|
| logical worker ids | `[C_p] = [10]`，恒为 `0..9` |
| worker→subspace ids | `[C_p] = [10]`，与 GPU placement 无关 |
| optional physical valid mask | 单卡无需 padding；三卡均匀 slots 时为 `[D,P/D]=[3,4]`，恰有 10 个 `True` |
| logical global replay | 一个全局容量/size/write-order/index 空间；物理上可集中、复制或 sharding |
| environment observation `s,s'` | `[B,O]`；训练 batch 为 `[N,O]` |
| action `a` | `[B,A]`；训练 batch 为 `[N,A]` |
| vector reward `r` | `[B,L]`；训练 batch 为 `[N,L]` |
| termination/done | `[B]`；训练 batch 为 `[N]` |
| episode preference `w` | `[B,L]`；训练 batch 为 `[N,L]` |
| actor output | `[B,A]` 或 `[N,A]` |
| twin critic output | `[N,2,L]` |
| scalarized twin Q | `[N,2]` |
| selected target vector Q | `[N,L]` |
| vector TD target `y` | `[N,L]` |
| interpolated direction `w_p` | `[N,L]` |
| relabeled effective batch | `[N(1+N_w), ...]` |

按官方源码语义映射后的数据流为：

1. 10 个逻辑 Brax worker lanes 各自持有固定到 episode 结束的 `w`，actor 以 `(s,w)` 产生 `[B,A]` action，再加 exploration noise。GPU placement 不参与 preference 选择。
2. Brax step 产生 `s'`、`r:[B,L]` 和经确认与官方 `done` 等价的结束信号；来自全部 worker IDs `0..9` 的有效 transitions 汇入同一个逻辑全局 PyTree replay。环境对应关系未确认前，本步不得实现。
3. 从逻辑全局 replay 分布采样 `N` 条 base transitions，再把 minibatch 分片到 learner devices；第一阶段 HER 随后在设备上生成额外 preferences，并沿 relabel 轴广播/展平。
4. target actor 以 `(s',w)` 产生 next action，加 target smoothing noise。
5. target critics 产生 `[N_eff,2,L]`；用原 `w` 标量化为 `[N_eff,2]`，选标量较小的 critic 的完整 Q 向量。
6. 使用经确认的官方 `done` 等价 mask，计算 vector target `y=r+γ(1-done)Q'_selected`。
7. `w_p=I(w)`；critic loss 严格为两个 Smooth-L1 vector losses 与两个未乘 `actor_loss_coeff` 的 angle means 之和。
8. 到源码 `policy_freq` 时，actor 最小化 `-mean(w^TQ_1)+actor_loss_coeff*mean(angle)`；随后 soft-update target actor 和 target critics。
9. evaluation 在固定 preference 网格上得到 vector returns，再构造 Pareto mask、HV 与 sparsity；这些指标不回流训练。

## 8. 论文与官方源码不一致清单

| 事项 | 论文 | 官方源码 | 实现规则 |
|---|---|---|---|
| preference sampling | simplex/子空间均匀 | episode 离散网格；HER absolute-normal + L1 normalize + 三位 rounding | 固定复现源码；不实现论文均匀 sampler |
| HER 开始时间 | 每个 transition relabel | replay warm-up 后才物理写入 relabel 条目 | 第一阶段用 sample-time relabel；完整框架后必须做源码忠实版消融 |
| angle 的向量 | Eq. (10) 标题是 `w_p`，分子印成 `w` | 使用 `w_p` | 使用 `w_p` |
| angle 单位/clip | 原始 `acos` 公式 | degree，cosine clip 到 `[0,0.9999]` | 首轮使用源码行为 |
| critic regression | 公式写 squared error并统一表达方向系数 | Smooth L1；critic angles 不乘 `actor_loss_coeff` | 逐操作复现源码 |
| interpolator normalization | “unit vector” | 初始化 L2，在线更新 L1 | 复现源码 L2/L1 差异；任何一致化另标 `DEVIATION` |
| target update timing | 一处文字近似“每步” | delayed actor 分支 | 遵循 Algorithm 3 与源码的 delayed 分支 |
| target critic selection | `arg_Q min_i w^TQ_i` 记号含糊 | 按 scalar score 选完整 vector | 明确 gather 完整 vector |
| 子空间 | 不重叠/均匀子空间 | lexicographic 离散网格 `np.array_split(w_batch,C_p)`，`C_p=10` | 永远保持 10 个逻辑子空间；GPU 数不参与划分 |
| 并行方式 | 10 个 child processes | PyTorch CPU multiprocessing | 10 个逻辑 workers 映射到 1 或 3 GPU；仅 placement 改变 |

## 9. 按依赖关系排序的最小实现顺序

以下顺序只描述后续工作，不表示本文已经实现：

1. **冻结官方源码行为表**：逐文件提取目标任务的网络、settings、reward/state/done、preference、HER、更新比、interpolator 和 evaluation 实际行为；每项保留源码文件与行号，不先用论文补齐源码。
2. **Brax 对应关系确认门**：并排列出官方旧 MuJoCo state/reward/termination/action bounds 与 Brax 可用量。任何一项不能直接对应时标为 `DEVIATION` 并停止该环境实现，等待确认；确认后才固定 `[B,L]` reward 和 done mask，并写 shape/finite-value 检查。
3. **数据通路与逻辑全局 replay**：在 workflow 维护 episode `w`，通过 `SampleBatch.extras` 写入 transition；汇集全部 10 个 workers 的有效数据到一个逻辑 pool。验证单卡/三卡都能从全体 worker IDs 采样，且经验频率与全局条目占比一致；禁止默认 local-only per-GPU replay。
4. **条件网络**：actor 输入扩为 `(s,w)`；twin critics 输入 `(s,w,a)` 并输出 `[B,2,L]`；层数、hidden size、ReLU、初始化和 action scaling 对齐源码/settings。先验证 init/apply/JIT shape 与 PyTorch 参考输出的结构。
5. **核心 MO-TD3 target（硬约束 + unit test）**：实现 `w^TQ`、按 scalarized score 选择完整 target-Q vector、vector Bellman target。先用交叉大小的人工 Q 验证输出只能是某个 critic 的整条向量、绝不等于 elementwise minimum；覆盖 batch 内不同选择，并验证 eager/JIT 一致后才接入训练。
6. **源码损失与更新时序**：critic 使用两个默认-mean Smooth-L1 加两个未加权 angle means；actor 使用 `-mean(w^TQ_1)+actor_loss_coeff*mean(angle)`；noise、`policy_freq`、delayed target update 和 Polyak 参数逐项对齐源码。保留官方 gradient clipping：critic 每次更新、actor 每次 delayed update 分别对各自完整参数梯度做 total-L2 global-norm clip，阈值均为 100。
7. **固定 10 个逻辑 preference workers，再做 GPU placement**：先独立于设备生成源码离散网格与 `np.array_split(w_batch,10)`，固定 worker ID 0–9 与 subspace ID 0–9 的映射；episode 只在对应 worker lane 从所属子集重采样。单卡承载全部 10 个 workers；三卡用 4/3/3 或 masked 12-slot placement，但有效 workers 仍为 10。测试必须证明单卡与三卡得到相同的 subspace 划分、ID 覆盖和每轮有效 transition/update 数。
8. **第一阶段 HER**：实现固定 `N_w` 的 GPU 简化版 sample-time expansion；验证 replay 只存 base transition，展开后 reward/其余 transition 不变且只有 `w` 改变。
9. **官方 key-preference 流程与 interpolator 边界**：按 key-training 脚本产生 objective solutions，复现 linear RBF、初始化 L2 与在线更新 L1。training JIT 内只做固定参数 `I(w)` 前向；evaluation/host boundary 低频重拟合，再以同 shape 参数返回训练。
10. **`w_p`、cosine 与 directional angle**：复现源码 interpolator 直接输出、cosine 默认 epsilon、`[0,0.9999]` clamp、`acos` 和 degree conversion；不得额外 normalize `w_p`。用 PyTorch/JAX 对照与有限梯度检查确认一致且无 NaN。
11. **源码 evaluation 行为**：复现官方 preference 网格、确定性动作、episode vector return、repeat/seed 与聚合顺序，输出 `[M,S,L]` returns；仅执行方式改为 Brax batch/JIT。
12. **Pareto/HV/sparsity**：在 evaluation/host boundary 复现官方 non-dominated sorting、取负、零 reference、pymoo HV 与 sparsity；任何等价替代先通过固定输入逐值对照，且 `N<=1` 的 sparsity 返回 0。
13. **端到端 smoke test**：先用单卡隔离算法正确性，再用三卡验证初始化、分片 rollout、逻辑全局 replay sampling、全局 minibatch 的 learner 分片、同步后的 vector critic/actor 更新、soft targets、JIT、HER 和若干持续训练 steps；记录每卡 local shape、全局有效 batch及 replay 中各 worker/subspace 的覆盖频率。
14. **源码忠实版 HER 消融（必做）**：完整训练与评估链路稳定后，实现写入时物理 relabel、源码 warm-up 门槛及满 buffer 的 `1+N_w` 淘汰/写入行为；在控制 seed、环境步数、base batch、`N_w`、sampler 与更新次数后，与 sample-time relabel 比较学习质量、吞吐量和设备内存。除非另有明确实验要求，不增加 preference、normalization、angle 或 replay 行为的自设计变体。

## 10. 明确不应在首版实现的内容

- 不新增另一套 replay buffer；现有 PyTree buffer 足够。
- 不移植 PyTorch multiprocessing/queue/shared-memory 架构；Brax/JAX batching 已覆盖其目的。
- 不把 SciPy interpolator 或 pymoo 放进 jitted training step。
- 不把 preference 混入环境 observation space，也不为其预先设计 encoder/hypernetwork。
- 不实现离散动作 MO-DDQN 的完整 Eq. (5) operator；连续 PD-MORL 只需要其 cosine 几何原语和角度项。
- 不同时支持任意目标数的高维 simplex 精确分区与通用高维 HV；首个二目标 Brax 基线验证后再扩展。

## 11. 结论

EvoRL 原版 TD3 可保留的骨架是 twin critics、连续 actor、两类动作噪声、delayed actor update、target soft update、JAX PyTree replay 和 Brax batched execution。真正需要替换的是标量 reward/Q/TD target/loss/evaluation；需要新增的是 preference 生命周期、relabeling、interpolated objective direction、角度约束和 Pareto 指标。最大工程风险不是 TD3 公式本身，而是同时保持固定 10 个逻辑 preference workers、跨设备的逻辑全局 replay distribution、固定 shape HER 和低频 interpolator/evaluation boundary，而不破坏 JIT 静态结构。按第 9 节顺序实现可以先闭合最小的向量 TD3 数据链，再逐层加入 PD-MORL 特有机制。
