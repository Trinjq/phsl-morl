# Step3：PD-MORL 官方连续控制行为冻结与 Brax 映射

## 1. Step3 结论

### 1.1 首个任务

首个复现任务建议选择官方 **`MO-Walker2d-v2`**，在 EvoRL 中以 Brax 0.14.2 的 **`walker2d`** 承载。选择依据是源码可证的语义接近程度，而不是沿用此前对 Hopper 的预设：

| 官方任务 | 官方维度与目标 | Brax 对应环境 | 关键映射问题 | 首任务判断 |
|---|---|---|---|---|
| `MO-Walker2d-v2` | `obs=17`，`action=6`；速度、能效 | `walker2d` | 动作界限、`frame_skip=4`、`dt=0.008`、observation、速度量和健康条件均有明确对应；XML/物理 pipeline 不同 | **最佳候选** |
| `MO-HalfCheetah-v2` | `obs=17`，`action=6`；速度、能效 | `halfcheetah` | 官方速度目标有上限 `min(4,v)` 且有角度 termination；Brax 默认均无此行为 | 次选，需增加更多环境语义 |
| `MO-Ant-v2` | `obs=27`，`action=8`；x/y 速度 | `ant` | 官方只在 state 非有限时结束；Brax 默认按 torso 高度结束，且 Brax XML actuator gear 为 150、官方为 100 | 非首选 |
| `MO-Swimmer-v2` | `obs=8`，`action=2`；速度、能效 | `swimmer` | 任务最小，但 Brax XML actuator gear 为 150、官方为 300，动力学尺度已直接改变 | 非首选 |
| `MO-Hopper-v2` | `obs=11`，`action=3`；速度、跳高 | `hopper` | 官方动作范围为 `[-2,-2,-4]..[2,2,4]`，Brax 三个动作维均为 `[-1,1]`；termination 与 actuator 也不同 | 非首选 |

上表 Hopper 的动作维为三维；“统一 `[-1,1]`”指三个动作维均使用该范围。

### 1.2 最终判定

**B. READY WITH EXPLICIT DEVIATIONS**

`MO-Walker2d-v2` 的 observation、两项 reward、action、物理 termination、time limit 和 TD mask 均能找到明确的 Brax/JAX 承载量，不存在需要猜测 reward decomposition 的核心 `BLOCKED` 项。但是，当前 Brax 不能被称为旧 MuJoCo 2.1 的逐轨迹复现：

1. 官方 XML 使用 RK4；Brax XML 明确写明移除 RK4，当前环境运行在 `generalized` pipeline。
2. Brax XML 为适配 Brax 重写了 Walker2d 根关节坐标和部分几何表达；Brax 源码再以 torso world-z 回填 observation 的 height。
3. 因此，公式和变量语义可以保持，数值轨迹、接触解算与最终 Pareto front 不应声明为 bitwise 或物理数值等价。

这两项属于已知 **DEVIATION**，不是未识别的假设。进入 Step4 前必须接受“复现 PD-MORL 算法语义，但运行于 Brax Walker2d 动力学”这一实验边界；否则结论退回 `BLOCKED`，且在“只允许 Brax/JAX”约束下没有完全等价的旧 MuJoCo 后端可用。

实验室核验环境：`/home/qiuquanj/miniforge3/envs/evorl`，Brax 0.14.2，JAX 0.10.2，三张 CUDA GPU 可见。GPU 0 的实际构造结果为：`obs=(17,)`、`action=6`、`ctrl_range=[-1,1]^6`、`n_frames=4`、`dt=0.008`、`q=(9,)`、`qd=(9,)`。本阶段只读取并运行环境构造检查，未修改训练代码。

## 2. 官方任务源码索引

以下行号均对应当前本地官方仓库快照。

| 文件 | 行号 | 职责 |
|---|---:|---|
| `lib/utilities/morl/MOEnvs/moenvs/__init__.py` | 38-42 | 注册 `MO-Walker2d-v2`，外层 `TimeLimit(max_episode_steps=500)` |
| `lib/utilities/morl/MOEnvs/moenvs/MujocoEnvs/walker2d.py` | 10-16 | 定义二目标 Walker2d，`obj_dim=2`、`frame_skip=4` |
| 同上 | 18-34 | action clipping、精确 vector reward、物理 done |
| 同上 | 36-48 | observation 与 reset 分布 |
| `lib/utilities/morl/MOEnvs/moenvs/MujocoEnvs/assets/walker2d.xml` | 7, 46-51 | `timestep=0.002`、RK4、六个 `[-1,1]` 且 gear=100 的 motor |
| `lib/utilities/settings.py` | 56-84 | `Walker2d_MO_TD3_HER` 主训练配置 |
| 同上 | 86-108 | key-preference 预训练配置 |
| `PD-MORL/train_Walker2d_MO_TD3_HER.py` | 31-52 | 每个逻辑 preference worker 的 seed、env、采样与队列写入 |
| 同上 | 80-85 | 从实际 env 冻结 observation/action/reward/action-bound/time-limit |
| 同上 | 112-144 | preference 网格、10 个子空间和初始 interpolator |
| 同上 | 154-206 | 10-lane sampling、全局 replay、更新比、插值器更新与 evaluation 时序 |
| `PD-MORL/train_Walker2d_MO_TD3_HER_Key.py` | 25-35, 175-200 | key preference 网格与三个 key workers |
| 同上 | 126-170 | key policy 的训练、10-episode evaluation 与最优 objective 保存 |
| `lib/models/networks.py` | 12-15 | Linear 使用 Xavier-normal，bias=0 |
| 同上 | 62-97 | Actor 结构、拼接顺序和 `tanh*max_action` |
| 同上 | 99-164 | 两个完全独立的 vector critics 与 Q1 路径 |
| `lib/common_ptan/agent.py` | 189-257 | target copy、Adam、episode preference 与 exploration noise |
| 同上 | 260-383 | MO-TD3 target、loss、gradient clip、delay 与 Polyak update |
| `lib/common_ptan/experience.py` | 8, 35-85 | transition 字段、random warm-up、combined done 写入 replay |
| 同上 | 144-198 | 全局 HER replay、`N_w=3` relabel 和满 buffer 行为 |
| `lib/utilities/MORL_utils.py` | 297-326 | sparsity 与 evaluation preference 网格 |
| 同上 | 355-446 | train-time 与 benchmark evaluation 的 seed、return、HV 和 sparsity |
| `PD-MORL/eval_benchmarks_MO_TD3_HER.py` | 23-40 | benchmark 名称与默认 evaluation repeat |
| 同上 | 52-90 | env-derived shape、模型加载、评估与 mean Pareto front |
| `environment.yml` | 56, 90, 100, 114 | Python 3.8.11、Gym 0.21.0、mujoco-py 2.1.2.14、pymoo 0.5.0 |
| `README.md` | 40-53 | 连续任务来源与 Gym `TimeLimit.truncated` 注意事项 |

论文只用于交叉核对：Appendix B.2 把 `MO-Walker2d-v2` 定义为 `S⊆R^17`、`A⊆R^6`、forward speed 与 energy efficiency；Appendix B.3 Table 5 给出 MO-TD3-HER 的主要超参；Algorithm 3 给出连续动作训练时序。精确 reward、done 和 wrapper 行为以源码为准。

## 3. 官方任务完整配置表

### 3.1 主训练配置

| 参数 | 冻结值 | 来源 | 默认/override 与说明 |
|---|---:|---|---|
| scenario | `MO-Walker2d-v2` | `settings.py:57` | 任务特定 |
| reward/objective count `L` | 2 | `walker2d.py:12,16`; train script `:83` | env-derived |
| observation dimension `O` | 17 | `walker2d.py:36-39`; train script `:81`; paper App. B.2 | env-derived |
| action dimension `A` | 6 | XML `:46-51`; train script `:82`; paper App. B.2 | env-derived |
| action low | `[-1,-1,-1,-1,-1,-1]` | XML `:46-51`; `walker2d.py:24` | env-specific |
| action high / `max_action` | `[1,1,1,1,1,1]` | train script `:84`; XML `:46-51` | env-derived |
| actor hidden layers | 2 × 400 | `settings.py:80,82`; `networks.py:74-78,92-95` | `layer_N_actor=1` 表示 input hidden 后再循环 1 个 hidden，不是总计 1 层 |
| critic hidden layers | 每个 critic 2 × 400 | `settings.py:79,82`; `networks.py:112-123,141-149` | Q1/Q2 各自独立 |
| hidden activation | ReLU | `networks.py:92-94,141-148` | 全局网络实现 |
| actor output activation | `tanh` 后逐维乘 `max_action` | `networks.py:85,95-97` | 全局网络实现 |
| initialization | weight: Xavier normal；bias: 0 | `networks.py:12-15,81-83,126-132` | 显式源码行为 |
| actor LR | `3e-4` | `settings.py:67` | 任务配置；Adam见 `agent.py:210` |
| critic LR | `3e-4` | `settings.py:68` | 任务配置；Adam见 `agent.py:212` |
| gamma | `0.995` | `settings.py:69` | 任务配置 |
| tau | `0.005` | `settings.py:73` | 任务配置 |
| batch size | 256 | `settings.py:70` | 任务配置 |
| replay capacity | 2,000,000 | `settings.py:61` | 逻辑全局 pool |
| `start_timesteps` | 10,000 **每个 worker** | `settings.py:63`; `experience.py:38,55-57` | 每个 child 的 `global_steps` 都从 0 开始 |
| configured `time_steps` | 1,000,000 **每个 worker** | `settings.py:62`; train script `:154-160` | 10 workers 实际合计约 10,000,000 env transitions；论文 App. B.1.4 也说明总量为 `N*C_p` |
| exploration noise `expl_noise` | 0.1 | `settings.py:66`; `agent.py:252-254` | actor action noise std=`max_action*0.1` |
| target policy noise | 0.2 | `settings.py:74`; `agent.py:312` | 未乘 `max_action` |
| target noise clip | 0.5 | `settings.py:75`; `agent.py:312` | 先 clip noise，再 clip action |
| policy delay | 10 learner updates | `settings.py:76`; `agent.py:357` | Walker task override；Hopper 才是 20 |
| actor angle coefficient | 10 | `settings.py:81`; `agent.py:369` | 只乘 actor angle term |
| critic gradient clip | global norm 100 | `agent.py:351-354` | 每个 critic update，覆盖 Q1/Q2 全部参数；论文 Table 5 未报告 |
| actor gradient clip | global norm 100 | `agent.py:370-376` | 只在 delayed actor branch；论文 Table 5 未报告 |
| process count `C_p` | 10 | `settings.py:71`; train script `:62,115-125` | 逻辑 preference workers，不等于 GPU 数 |
| preference training grid step | 0.001 | `settings.py:64`; train script `:113` | 二目标时 1001 个离散 preference |
| preference subspaces | `np.array_split(W,10)` | train script `:115` | 1001 个点按顺序分成 10 个逻辑子空间，size 为 `[101,100,100,100,100,100,100,100,100,100]` |
| HER preference count `N_w` | 3 | `settings.py:65`; `experience.py:176-188` | `abs(N(0,1))`、L1 normalize、round 3 decimals |
| learner-start replay threshold | 1536 entries | train script `:162` | 实际表达式 `2*batch_size*weight_num`；不是 `start_timesteps` |
| updates per collection round | 10 | train script `:156-171` | 每轮收集 10 条 base transitions，再做 10 次 batch update |
| HER activation threshold | replay length >100,000 | `experience.py:184` | `start_timesteps*process_count` |
| episode limit | 500 | registry `moenvs/__init__.py:38-42`; train script `:85` | Gym TimeLimit override；settings `:78` 同值但脚本再从 env 覆盖 |
| physical frame skip | 4 | `walker2d.py:14` | env-specific |
| XML simulation timestep | 0.002 s | official XML `:7` | env-specific |
| control-step `dt` | 0.008 s | `frame_skip*timestep` | 源码公式使用 `self.dt` |
| load model | false | `settings.py:59` | 任务配置 |
| official training device | CPU (`cuda=False`) | `settings.py:58`; train script `:71` | JAX/Brax GPU 替代属于 framework adaptation |
| seed | 1 | `settings.py:83` | worker p 使用 `p_id*seed`，实际 seeds 0..9，见 train script `:32-38` |

### 3.2 Evaluation 与 key-preference 配置

| 项目 | 冻结行为 | 来源 |
|---|---|---|
| training 中间 evaluation 网格 | step 0.005，即二目标 201 preferences | train script `:114` |
| training 最终 evaluation 网格 | step 0.001，即二目标 1001 preferences | train script `:113,205-206` |
| training evaluation repeats | 3 | `settings.py:77`; train script `:179,193,206` |
| offline benchmark 网格 | step 0.001，即 1001 preferences | eval script `:62`; `settings.py:64` |
| offline benchmark repeats | CLI 默认 6，可覆盖 | eval script `:26,77` |
| seed 序列 | repeat `k` 使用 `k*11` 同时 seed env、action space、Torch、NumPy | `MORL_utils.py:361-365,407-411` |
| preference 间 reset | 每个 repeat 只 seed 一次；该 repeat 内不同 preference 顺序调用 reset，因此不是所有 preference 共用同一初态 | `MORL_utils.py:361-387,407-434` |
| policy | deterministic actor，无 exploration noise | `MORL_utils.py:378-383,425-431`; `agent.py:255-256` |
| return | 500-step以内 vector reward 的**未折扣**和 | `MORL_utils.py:375-387,423-434` |
| train-time metric aggregation | 先逐 repeat 算 HV/sparsity，再对 repeats 取 mean；objectives 也取 mean | `MORL_utils.py:359-398` |
| offline metric aggregation | 返回每个 repeat 的 HV/sparsity/objectives；脚本再报告 mean/std，并对 mean objective 做 non-dominated filter | eval script `:77-93` |
| evaluation episode limit | TimeLimit combined done，最大 500 | registry `:38-42`; evaluation while-loop |
| key preferences | 3 个：从 0.001 网格等距索引，二目标即 `[0,1]`、`[0.5,0.5]`、`[1,0]` | key script `:175-200` |
| key evaluation repeats | 每次 10，seeds `0,10,...,90` | key script `:38-69,143-158` |
| key-training total steps | 2,000,000/worker | `settings.py:91`; key script `:126` |
| key-training start random | 25,000/worker | `settings.py:92`; `experience.py:55-57` |
| key-training batch/replay | 100 / 500,000 | `settings.py:90,97` |
| 未找到的 evaluation 行为 | 独立训练 run 的六个具体 seeds：**NOT FOUND** | 论文只报告六次运行，仓库脚本未提供六个 run seed 列表 |

## 4. 官方 Actor / Critic 结构

记 batch 为 `B`，state 维 `O=17`，preference 维 `L=2`，action 维 `A=6`，hidden size `H=400`。

### 4.1 OFFICIAL

Actor（`networks.py:62-97`）：

```text
state [B,17], preference [B,2]
  -> cat((state, preference), dim=1)             [B,19]
  -> Linear(19,400), ReLU                        [B,400]
  -> Linear(400,400), ReLU                       [B,400]
  -> Linear(400,6), tanh                         [B,6]
  -> elementwise * max_action=[1,1,1,1,1,1]    [B,6]
```

Critic（`networks.py:99-164`）：

```text
state [B,17], preference [B,2], action [B,6]
  -> cat((state, preference, action), dim=1)     [B,25]

Q1: Linear(25,400) -> ReLU -> Linear(400,400) -> ReLU -> Linear(400,2)
Q2: Linear(25,400) -> ReLU -> Linear(400,400) -> ReLU -> Linear(400,2)

output: Q1 [B,2], Q2 [B,2]
```

Q1/Q2 从输入层到输出层都不共享参数。`Critic.Q1` 复用 Q1 的参数路径给 actor loss。所有 Linear weight 用 Xavier-normal，bias 为 0。

### 4.2 OFFICIAL / EVORL / DIFFERENCE / REQUIRED CHANGE

| 项目 | OFFICIAL PD-MORL | 当前 EvoRL TD3 | DIFFERENCE | Step4 REQUIRED CHANGE（本阶段不实现） |
|---|---|---|---|---|
| actor input | `[state, preference]`，state 在前 | 只输入 observation，`td3.py:116` | 少 `w:[B,2]` | 输入尾维从 17 扩到 19，顺序固定为 `(s,w)` |
| actor hidden | 400,400 + ReLU | 256,256 + ReLU，配置 `td3.yaml:36-37` | 宽度不同 | 任务配置改为 400,400 |
| actor init | Xavier-normal/bias 0 | LeCun-uniform，`linear.py:165-170` | 初始化不同 | 为 PD-MORL 网络显式使用 Xavier-normal/bias 0；不改原 TD3 默认 |
| actor output | `tanh*max_action` | tanh，Brax 范围当前为 `[-1,1]` | Walker 数值等价，通用接口语义不同 | 保留逐维 `max_action` 表达，Walker 为全 1 |
| critic input | `cat(s,w,a)` | `cat(obs,a)`，`linear.py:226-230` | 少 `w` | 顺序固定为 `(s,w,a)` |
| critic hidden | Q1/Q2 各自 400,400 | 两个 vmapped 独立 MLP，各 256,256 | 宽度不同，独立性相同 | 改成 400,400，保留两个独立 parameter sets |
| critic output | 每个 critic `[B,2]` vector Q | 合计 `[B,2 critics]`，每个 critic 标量 | 输出轴语义不同 | 输出显式为 `[B,2 critics,2 objectives]` 或等价 PyTree |
| actor loss critic | 只用 Q1 | config 默认 first critic | 骨架一致 | vector scalarization/angle 按 Step2 冻结行为替换 |

网络映射分类：拼接顺序、层数、宽度、激活、输出和初始化为 **SOURCE-FAITHFUL**；用 Flax/JAX 表达以及以静态 `[B,2,2]` 承载 twin vector Q 为 **FRAMEWORK-ADAPTATION**。

## 5. 官方 observation 定义

### 5.1 三种 state 必须分开

1. **MuJoCo internal state**：`state_vector()` 是完整 `qpos || qvel`；Walker 的 `qpos.shape=(9,)`、`qvel.shape=(9,)`，合计 18。官方 step 中 `qpos0_sum`、`qvel0_sum`（`walker2d.py:19-20`）没有被后续使用。
2. **policy observation**：

   ```python
   concat(qpos[1:], clip(qvel, -10, 10)).ravel()
   ```

   shape 为 `(8+9,)=(17,)`。全局 x 位置 `qpos[0]` 被删除；height `qpos[1]` 和 torso angle `qpos[2]` 保留；所有 qvel 保留但逐元素 clip 到 `[-10,10]`。
3. **reward/termination 直接读取的内部量**：reward speed 读取动作前后的 `qpos[0]`，该量不在 observation；physical done 读取动作后的 `qpos[1]` 和 `qpos[2]`，两者在 observation 的前两维。

### 5.2 Reset

`reset_model` 使用：

```text
qpos = init_qpos + Uniform(-0.005, 0.005, size=nq)
qvel = init_qvel + Uniform(-0.005, 0.005, size=nv)
obs  = concat(qpos[1:], clip(qvel,-10,10))
```

reset 不返回完整 internal state，只返回 17 维 observation。

### 5.3 Brax 对应

Brax 0.14.2 `walker2d.py:242-251` 使用 `pipeline_state.q`、`pipeline_state.qd`，先把 `position[1]` 替换为 torso world-z，再删除 position[0]，并把 qd clip 到 `[-10,10]`。这是因为 Brax XML 把旧 XML 的 root joints 改成相对根 body 的表示。最终 observation 仍是 17 维，字段语义和顺序可以对应。

结论：policy observation 语义为 **FRAMEWORK-ADAPTATION**，shape/order 可严格保持；internal generalized coordinates 与旧 MuJoCo qpos 的数值轨迹为 **DEVIATION**。

## 6. 官方 vector reward 定义

官方环境在 `walker2d.py:18-34` 先保存动作前 root-x，再把动作 clip 到 `[-1,1]^6`，执行 4 个 MuJoCo substeps，然后读取动作后的 root-x、height、angle。

令：

- `x_t = qpos_before[0]`
- `x_{t+1} = qpos_after[0]`
- `dt = 4 * 0.002 = 0.008 s`
- `a_c = clip(a, -1, 1)`
- `alive_bonus = 1.0`

则精确 vector reward 为：

```text
r_t = [
  (x_{t+1} - x_t) / 0.008 + 1.0,
  4.0 - sum_j(a_c[j]^2) + 1.0
]
    = [v_x + 1.0, 5.0 - ||a_c||_2^2]
```

### Objective 1：forward speed

- 源码：`reward_speed=(posafter-posbefore)/self.dt+alive_bonus`。
- 使用动作前后内部 `qpos[0]`；该字段不在 policy observation。
- 含 `dt=0.008`。
- 含 `alive_bonus=1`，包括发生 physical termination 的最后一步。
- 不含 control cost，不含速度 clipping。
- 与 previous state 比较。
- 单位主体为 m/s，再加无量纲每步 bonus；源码未做单位统一。

### Objective 2：energy efficiency

- 源码：`reward_energy=4.0-sum(square(a_c))+alive_bonus`。
- 只使用 clip 后的 action，不读取 qpos/qvel/body。
- 不含 `dt`。
- 含 `alive_bonus=1`，包括 terminal step。
- control 项系数严格为 1.0；不是 Brax 默认 `0.001`。
- 无额外 clipping；但输入 action 已先 clip。
- 不与 previous state 比较。
- 尺度范围在 action 全部落于 `[-1,1]` 时为 `[-1,5]`。

论文 Appendix B.2 只称两项目标为 forward speed 和 energy efficiency，没有给出 `+1`、`4` 或 action-square 系数；这些常数必须来自源码，不能由论文名称推断。

### Brax 未来映射（设计确认，不实现）

Brax 当前 scalar reward 是 `x_velocity + 1 - 0.001*sum(action^2)`，不能直接复用为任一官方 vector component。可在 JIT/device 内使用现有量构造：

```text
r1 = brax_state.metrics.x_velocity + 1.0
r2 = 5.0 - sum(clip(action,-1,1)^2)
reward = stack([r1,r2], axis=-1)
```

这保持官方公式，属于 **FRAMEWORK-ADAPTATION**；Brax 轨迹本身来自不同 physics pipeline，属于 **DEVIATION**。禁止把 Brax 默认 `reward_ctrl=-0.001||a||²` 当作官方 energy objective。

## 7. 官方 action 定义

| 项目 | 官方实际行为 | Brax 对应 | 判定 |
|---|---|---|---|
| action shape | `(6,)` | `(6,)`，实验室运行已验证 | YES |
| `action_space.low` | `[-1]*6` | `sys.actuator.ctrl_range[:,0]=[-1]*6` | YES |
| `action_space.high` / `max_action` | `[1]*6` | `[1]*6` | YES |
| actor output | `tanh(z)*max_action` | tanh 输出对 Walker 数值相同 | YES |
| env boundary clip | `clip(a,-1,1)` 后才 simulation/reward | Brax actuator range 相同；未来 vector reward 应显式使用 clip 后 action | YES（语义） |
| exploration noise | `Normal(0,max_action*0.1)`，加到 actor action 后 clip | `Normal(0,0.1)` 后 clip | YES |
| warm-up action | 每个 worker 前 10,000 steps 使用 `action_space.sample()` | JAX uniform box sample 可等价 | YES |
| target smoothing | `Normal(0,0.2)` -> noise clip `[-0.5,0.5]` -> actor+noise -> action clip `[-1,1]` | JAX 可原顺序表达 | YES |

官方所有动作维共享相同上下界，但这是 Walker 的源码事实，不能推广到 Hopper。

## 8. 官方 done / termination / time-limit 语义

### 8.1 Physical termination

动作执行后：

```text
healthy = (qpos_after[1] > 0.8)
       and (qpos_after[1] < 2.0)
       and (qpos_after[2] > -1.0)
       and (qpos_after[2] < 1.0)
done_env = not healthy
```

边界均为严格不等号。非有限 state 没有单独检查。terminal step 仍获得两项目标中的 alive bonus。

### 8.2 Time limit

注册代码用 Gym 0.21 `TimeLimit(max_episode_steps=500)` 包装环境。Gym 0.21 在第 500 step 把外层 `done=True`；当底层此时没有 physical done 时，`info["TimeLimit.truncated"]=True`。官方 README `:50-53` 只说必要时可以注释写入这条 info 的语句，并未取消 `done=True` 本身。

### 8.3 Replay 和 TD target 实际使用什么

调用链为：

```text
base env done_env
  -> Gym TimeLimit: done = done_env OR elapsed_steps>=500
  -> MORLExperienceSource stores terminal=done, ignores returned info
  -> replay stores the same terminal
  -> find_non_terminal_idx(terminal==0)
  -> target_Q = reward + gamma * selected_Q only on non-terminal rows
```

因此官方 TD target 对 **physical termination 和 500-step truncation 都不 bootstrap**。源码没有读取 `TimeLimit.truncated` 来恢复 bootstrap。

### 8.4 EvoRL/Brax 必需差异

EvoRL `EpisodeWrapper` 明确分开 `termination` 与 `truncation`（`training_wrapper.py:81-93`），而当前原版 TD3 target 只用 `termination`（`td3.py:178`），即会在 time limit bootstrap。PD-MORL source-faithful target 必须使用 combined episode `done`，不能直接沿用原 TD3 的 termination-only mask。

Brax Walker physical done 使用 torso world-z 和 `q[2]`，范围同样是 `(0.8,2.0)` 与 `(-1,1)`（实验室 Brax `walker2d.py:214-229`）。公式映射为 **FRAMEWORK-ADAPTATION**；combined time-limit mask 是 **SOURCE-FAITHFUL** 的必要行为。若选择保留 EvoRL termination-only bootstrap，必须标为 **DEVIATION**。

## 9. 官方 training 时序

1. 先生成 0.001 的 1001-point preference grid，再用 `np.array_split(...,10)` 固定成 10 个逻辑子空间。
2. worker `p` 用 seed `p` 创建一个 `MO-Walker2d-v2`；每个 episode 首次 policy call 从自己的子空间离散均匀抽一个 `w`，round 到 3 decimals，并保持到 combined done。
3. 每个 worker 前 10,000 local steps 用 action-space random action；transition 中仍保存该 episode 的 `w`。
4. 每个主循环从每个 worker 各取一条 transition，共 10 条，按 worker 顺序汇入同一个 main-process replay。
5. replay 未达到 1536 entries 时不学习。超过后，每个主循环执行 10 次 learner update，每次从逻辑全局 replay 有放回均匀采样 256 条。
6. 每个 `learn()` 都先令 `total_it += 1`，计算 target actor、target smoothing、两个 vector target critics，并按 `w^TQ_i` 较小者选择**整条** Q vector。
7. target 使用 combined terminal mask：`y=r+0.995*(1-terminal)*Q_selected`。
8. 每次 learner update 更新 critic；loss 为两个 Smooth-L1 vector losses 和两个 angle means 的和。backward 后对 Q1+Q2 全部 critic 参数做 global-norm clip 100，再 Adam step。
9. 当 `total_it % 10 == 0` 时才更新 actor。actor loss 使用 Q1；backward 后对完整 actor gradient 做 global-norm clip 100，再 Adam step。
10. 仅在同一个 delayed actor branch 内，以 `target = 0.005*online + 0.995*target` 同时更新 target critic 与 target actor。
11. 当所有 10 个 worker 的 episode count 超过下一个整数阈值时，在 host evaluation 路径更新 interpolator；当所有 worker 超过 `100*eval_cnt` episodes 时，用 0.005 grid 做 training evaluation。

GPU/JAX 映射继续保持 Step2 约束：10 是逻辑 preference lanes，不是物理 GPU 数；1 张或 3 张 GPU 只改变 placement。所有 lanes 写入一个逻辑全局 replay，sample 全局 batch 后才能做 learner device sharding。

## 10. MuJoCo → Brax 对照表

实验室 Brax 源码根：`/home/qiuquanj/miniforge3/envs/evorl/lib/python3.11/site-packages/brax/envs/`。

| 官方 PD-MORL / MuJoCo | Brax 0.14.2 候选 | 是否等价 | 证据 | 结论/分类 |
|---|---|---|---|---|
| observation `concat(qpos[1:],clip(qvel,-10,10))`，17维 | `walker2d.py:242-251`：q、qd；用 torso world-z 回填 height；删 root-x；clip qd | YES（字段语义/shape），NO（内部坐标数值） | 官方 `walker2d.py:36-39`；Brax 源码与 GPU runtime shape | **FRAMEWORK-ADAPTATION** + physics **DEVIATION** |
| forward displacement `qpos_after[0]-qpos_before[0]` | torso `x.pos[0,0]` 前后差；metrics `x_velocity` | YES（物理量/公式） | 官方 `:21,26,28`；Brax `:209-212,234-235` | **FRAMEWORK-ADAPTATION** |
| height=`qpos[1]` | torso world-z=`x.pos[0,2]` | YES（物理语义） | 官方 `:26,30`；Brax `:214-218,245` | **FRAMEWORK-ADAPTATION** |
| torso angle=`qpos[2]` | `pipeline_state.q[2]` | YES（语义） | 官方 `:26,30`；Brax `:214-218` | **FRAMEWORK-ADAPTATION** |
| qpos shape 9；root-x excluded from obs | q shape 9；position[0] excluded | YES（shape/order at policy boundary） | 官方 `_get_obs`; lab runtime | **SOURCE-FAITHFUL** at interface |
| qvel shape 9；clip to `[-10,10]` | qd shape 9；相同 clip | YES | 两侧 `_get_obs`; lab runtime | **SOURCE-FAITHFUL** |
| action bounds `[-1,1]^6`；gear=100 | ctrl range `[-1,1]^6`；gear=100 | YES | 两侧 XML；lab runtime | **SOURCE-FAITHFUL** |
| physical termination height `(0.8,2)` 且 angle `(-1,1)` | 相同范围，使用 torso world-z 和 q[2] | YES（条件） | 两侧 step 源码 | **FRAMEWORK-ADAPTATION** |
| XML timestep=0.002，frame_skip=4，`dt=0.008` | timestep=0.002，n_frames=4，runtime `dt=0.008` | YES（control-step duration） | XML、两侧 env、lab runtime | **SOURCE-FAITHFUL** |
| objective 1=`v_x+1` | 可由 metrics.x_velocity 构造；Brax 默认 scalar reward 还减 `0.001||a||²` | YES（候选量），默认 reward NO | 两侧 step 源码 | vector 构造为 **FRAMEWORK-ADAPTATION**；复用默认 scalar reward 为错误 |
| objective 2=`5-||clip(a)||²` | action 在 device 内可用；Brax 默认只提供 `-0.001||a||²` metric | YES（候选量），默认 metric NO | 两侧 step 源码 | exact vector 构造为 **FRAMEWORK-ADAPTATION** |
| 500-step Gym TimeLimit 产生 combined done | EvoRL wrapper 产生 done、termination、truncation 三者 | YES（可表达） | registry；`training_wrapper.py:81-93` | target 使用 combined done 才是 **SOURCE-FAITHFUL** |
| RK4 + mujoco-py 2.1.2.14 | Brax generalized pipeline；XML 明示 removed RK4 | NO | official XML `:7`; Brax XML option/comment；Brax class默认 backend | **DEVIATION** |
| 旧 XML global root joint/ref 表达 | Brax XML 移动 rootx/rootz 到 rooty 位置并移除 joint ref | NO（内部表示），意图保持相同机构 | Brax XML 内注释；Brax `_get_obs` height 回填 | **DEVIATION** at dynamics；interface 可适配 |
| reset qpos/qvel 均 Uniform ±0.005 around init | Brax q/qd 均 Uniform ±0.005 around adapted init | YES（分布规则），NO（不同 simulator 的 state realization） | 官方 `:41-48`; Brax `:178-190` | **FRAMEWORK-ADAPTATION** + dynamics **DEVIATION** |

关键判断：名字相同没有被当作等价证据。这里的 `YES` 只表示公式所需的物理量和边界能在 Brax 中明确找到；它不消除 simulator、积分器和接触数值的差异。

## 11. Mapping Classification

| 映射项 | 分类 | 约束 |
|---|---|---|
| 17维 policy observation 字段与顺序 | **FRAMEWORK-ADAPTATION** | Brax world-z 回填替代旧 qpos height，但输出语义固定 |
| 两目标定义、常数与 reduction | **SOURCE-FAITHFUL** | 只能是 `[v_x+1, 5-||a||²]` |
| 在 Brax step/device 内构造 vector reward | **FRAMEWORK-ADAPTATION** | 固定 `[B,2]`，无 host callback |
| 动作 shape、bounds、clip、actor scale | **SOURCE-FAITHFUL** | Walker 六维均 `[-1,1]` |
| physical done 条件 | **FRAMEWORK-ADAPTATION** | 使用 Brax torso world-z 与 q[2] 实现相同严格区间 |
| time-limit combined done 进入 TD target | **SOURCE-FAITHFUL** | 500-step truncation 不 bootstrap |
| JAX/Brax batch 与 JIT | **FRAMEWORK-ADAPTATION** | Brax 原生 batch 优先，`vmap` 非强制 |
| 10 logical preference lanes 到 1/3 GPU | **FRAMEWORK-ADAPTATION** | 不改变 `C_p=10` 或 `np.array_split` |
| 逻辑全局 replay | **SOURCE-FAITHFUL** | 禁止默认三个隔离 replay pools |
| 旧 MuJoCo RK4 → Brax generalized dynamics | **DEVIATION** | 不宣称轨迹或 Pareto 数值等价 |
| 旧 XML root coordinates → Brax adapted XML | **DEVIATION** | 保持外部字段语义，承认内部 dynamics 不同 |
| 保留 EvoRL termination-only bootstrap | **DEVIATION** | 默认不采用；若采用必须单列实验 |
| 直接复用 Brax scalar reward | **DEVIATION** | 且不符合官方 objective，默认禁止 |

## 12. BLOCKED / 需要人工确认的问题

### 12.1 BLOCKED 汇总

在“复现算法语义、使用 Brax Walker2d 动力学”的边界内，**没有核心 reward/state/action/done 的 BLOCKED 项**。

若目标被定义为“复现旧 MuJoCo 2.1 的数值轨迹/接触结果”，则存在不可消除的 BLOCKED：当前允许的训练后端只有 Brax/JAX，而 Brax 使用不同 pipeline、积分器和适配后的 XML。本文不自行把两者判成物理数值等价。

### 12.2 进入 Step4 前唯一需要确认的实验边界

需要人工确认是否接受以下表述：

> Step4 source-faithful 复现 PD-MORL 的网络、preference、reward 公式、done/time-limit mask、replay、loss、delay、target update 和 evaluation 语义；环境动力学采用 Brax 0.14.2 Walker2d，因此与官方 mujoco-py 2.1.2.14 的轨迹和最终指标存在显式 deviation。

不接受该边界时，不应开始 Step4。

## 13. Step4 实现前置条件

1. 明确接受第 12.2 节的 Brax physics deviation；否则停止。
2. 为 PD-MORL 单独冻结 Walker 配置，不修改 EvoRL 原版 TD3 默认配置或公式。
3. 环境侧必须在 JIT/device 内输出 `[B,2]`：`[x_velocity+1, 5-sum(action²)]`；不得复用 Brax scalar reward，也不得把 reward 拉回 host。
4. observation 必须保持 `[B,17]`；preference 单独保持 `[B,2]`，不并入环境 observation space。
5. action 必须为 `[B,6]`，逐维 `[-1,1]`；噪声和 clip 顺序按第 7 节冻结。
6. physical termination 使用严格 height/angle 区间；episode limit 为 500；replay 和 TD target 使用 combined done，使 time-limit transition 不 bootstrap。
7. Actor/Critic 层、Xavier-normal、bias 0、拼接顺序和 vector-Q shape 按第 4 节冻结。
8. 保持 `C_p=10` 与 1001-point grid 的 `np.array_split` 语义；1/3 GPU 只负责 placement。
9. replay 在逻辑上必须汇集全部 10 lanes；learner 从全局分布 sample 后再做多 GPU sharding。
10. Step2 已批准的第一阶段 sample-time HER 仍明确标为 **DEVIATION**；完整框架后补 add-time source-faithful HER 消融。
11. TD target 必须先分别计算 `w^TQ1`、`w^TQ2`，再 gather 被选 critic 的完整 2维 Q；禁止 elementwise minimum，并保留对应 unit test。
12. interpolator training-step 内只执行固定参数 `I(w)`；低频重拟合留在 evaluation/host boundary。
13. 单 GPU 先验证公式与 shapes，再用三 GPU 验证相同 10-lane preference 划分、全局 replay distribution 与同步 learner；GPU 数不能改变算法采样语义。

满足以上条件后才进入 Step4；本文没有实现任何 PD-MORL、vector critic、HER、interpolator 或多目标 Brax 环境代码。
