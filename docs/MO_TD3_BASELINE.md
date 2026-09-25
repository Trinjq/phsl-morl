# Preference-Conditioned MO-TD3 Baseline

## 结论

**STEP 4.4 PASS。** 当前实现以 EvoRL 原生 TD3 为骨架，增加了 preference-conditioned Actor、twin vector Critic、whole-vector pessimistic target selection、vector Bellman target、Smooth-L1 critic loss 和 scalarized actor objective。单 GPU Brax Walker2d smoke test 与已有 Step4.1–4.3 回归均通过。

这仍然是最小 MO-TD3 baseline，不是完整 PD-MORL。

## Actor

输入：

```text
observation  [B,17]
preference   [B,2]
concat       [B,19]
```

网络严格采用：

```text
Linear(19,400) -> ReLU
Linear(400,400) -> ReLU
Linear(400,6) -> tanh -> × max_action
```

Walker2d 的 `max_action=1`，输出为 `[B,6]`。在线 actor 与 target actor 使用相同结构；target actor 接收同一 transition 的原始 preference：`pi_target(s', w)`。

## Twin vector Critic

每个 critic 的输入顺序为：

```text
concat(observation, preference, action)
[B,17] + [B,2] + [B,6] -> [B,25]
```

两个 critic 参数完全独立：

```text
Q1: 25 -> 400 -> 400 -> 2
Q2: 25 -> 400 -> 400 -> 2
```

统一输出：

```text
Q.shape == [B,2 critics,2 objectives]
```

Actor 和 Critic 均使用 ReLU、Xavier-normal kernel 与零 bias。原 scalar TD3 仍使用原有 256×256/LeCun 默认值，没有被全局修改。

## Scalarization

纯函数 `scalarize(Q, w)` 实现：

```text
w^T Q = sum(w * Q, axis=-1)
```

支持：

```text
Q [B,2]   + w [B,2] -> [B]
Q [B,2,2] + w [B,2] -> [B,2 critics]
```

训练始终使用 replay transition 中的原始 `SampleBatch.extras.policy_extras.preference`。

## Target action 与 twin target selection

Target action 保留 TD3 顺序：

```text
a' = pi_target(s', w)
noise = clip(N(0, policy_noise), -noise_clip, noise_clip)
a' = clip(a' + noise, -1, 1)
```

Walker 配置沿用官方值：`policy_noise=0.2`、`noise_clip=0.5`。

Target critics 产生 `Q' [B,2,2]`。先分别 scalarize：

```text
z1 = w^T Q1'
z2 = w^T Q2'
idx = argmin([z1,z2])
```

随后按 `idx` 选择该 critic 的完整二目标向量。实现禁止 elementwise minimum；测试明确验证交叉向量 `[1,2]` 不会被生成。

## Vector Bellman target 与 done mask

```text
done = maximum(termination, truncation)
y_vec = reward_vec + gamma * (1-done)[...,None] * Q'_selected
```

shape：

```text
reward_vec [B,2]
done       [B]
y_vec      [B,2]
```

physical termination 和 time-limit truncation 均不 bootstrap。MO-TD3 workflow 只扩展其收集字段为 `ori_obs, termination, truncation`；原 scalar TD3 仍收集原来的 `ori_obs, termination`。

## Loss 与更新

Critic 使用两个独立 mean-reduced Smooth-L1 loss 的和：

```text
L_Q = mean(SmoothL1(Q1,y_vec)) + mean(SmoothL1(Q2,y_vec))
```

Actor 只使用 critic 1：

```text
a = pi(s,w)
L_actor = -mean(w^T Q1(s,w,a))
```

优化器为 Adam，actor/critic gradient clipping 均为 global norm 100。

`MOTD3Workflow` 继承 `TD3Workflow.step`：

- `actor_update_interval=10`；
- EvoRL 没有持久化 update counter，也没有执行 `counter % policy_freq`；
- 每个 learner update block 先通过 `lax.scan(length=9)` 执行 9 次 critic-only update，再执行第 10 次 critic update；
- 第 10 次 critic update 后立即执行一次 actor update，并更新 target actor 和 target critics；
- 因此每个 block 的实际比例是 10 次 critic update 对应 1 次 actor/target update；
- target 参数继续使用 EvoRL 原生 `soft_target_update(..., tau=0.005)`。

这与 EvoRL 原生 scalar TD3 的 delayed-update 实现完全一致。它在工程上使用固定长度的局部 update block 表达 `policy_freq=10` 的更新比例，而不是 PD-MORL PyTorch 源码中的持久化 `total_it % policy_freq == 0` counter。

## EvoRL TD3 复用关系

直接复用：

- `TD3NetworkParams` 与 `AgentState`；
- `TD3Workflow.step` 的 rollout、replay sampling 和 update skeleton；
- `agent_gradient_update`；
- TD3 exploration noise；
- delayed actor update；
- Polyak soft target update；
- `ReplayBuffer`、`SampleBatch`、JAX scan/JIT 与 Brax batch execution。

新增的 MO 专用实现位于 `evorl/algorithms/mo_td3.py`。没有从零建立另一套 training loop。

为了让 Actor 在选择 rollout action 时获得当前 episode 的 preference，`env_step` 和 `eval_env_step` 仅在 action 调用输入的 `SampleBatch.extras.policy_extras` 中透传 `env_state.info.preference`。transition 保存位置、episode lifecycle、rollout 顺序和 replay 结构没有改变。

## 配置

入口：

```bash
python scripts/train.py agent=mo-td3 env=brax/walker2d
```

`configs/agent/mo-td3.yaml` 使用：

```text
reward_size=2
process_count=10
hidden=[400,400]
gamma=0.995
exploration_noise=0.1
policy_noise=0.2
noise_clip=0.5
actor_update_interval=10  # 对应官方 policy_freq=10 的更新比例
tau=0.005
grad_clip_norm=100
```

Hydra `--cfg job` 已验证配置可正确组合到 `MOTD3Workflow`。

## 修改文件

- `evorl/algorithms/mo_td3.py`：MO networks、agent、纯算法函数和专用 workflow。
- `configs/agent/mo-td3.yaml`：MO-TD3 官方 Walker 参数。
- `evorl/algorithms/offpolicy_utils.py`：允许 workflow 指定需要保存的 env-extra 字段，默认值不变。
- `evorl/algorithms/td3.py`：使用上述默认字段列表；scalar TD3 行为不变。
- `evorl/rollout.py`：向 action function 透传已存在的 episode preference。
- `tests/test_mo_td3.py`：算法、JIT、scalar TD3 regression 和真实 Walker smoke tests。

未修改 vector reward、official preference grid、process_count、subspace、episode preference lifecycle、SampleBatch 或 replay buffer。

## GPU/JIT 测试结果

实验室设备：

```text
backend: gpu
device: CudaDevice(id=0)
GPU: NVIDIA GeForce RTX 4090
```

MO-TD3 网络和纯算法测试：

```text
4 passed, 1 deselected, 1 warning in 32.39s
```

覆盖 Actor/Critic shape 与网络层、critic 参数独立性、scalarization、whole-vector target selection、vector Bellman target、combined done、Smooth-L1、target smoothing、actor/critic loss 和 scalar TD3 shape regression。

真实 Brax Walker2d smoke test：

```text
1 passed, 10 warnings in 548.84s (0:09:08)
```

该测试使用 vector reward、official preference、真实 Walker rollout 和 replay sample。测试先显式执行一次 critic-only update 并确认 actor 不变，再执行第二次 critic update，然后由测试代码无条件显式调用一次 actor update 和 target soft update，验证 loss finite 与预期参数变化。

这里的第二次 critic update **不会**通过 `policy_freq=10` 自动触发 actor update；测试没有设置或推进 workflow counter。该 smoke test 验证的是 critic-only、actor update 和 target update 各组成操作可以在真实 Walker batch 上执行，不验证完整的 10:1 workflow cadence。正式 `MOTD3Workflow` 的 cadence 由上述继承自 EvoRL TD3 的 `lax.scan(length=9) + final critic/actor/target update` 实现。

Step4.1–4.3 GPU 回归：

```text
19 passed, 31 warnings in 555.02s (0:09:15)
```

警告来自 JAXopt/Brax 维护状态和 Brax contact API deprecation，不是测试失败。

## 与完整 PD-MORL 的差异

当前未实现：

- HER 或 preference relabeling；
- interpolator `I(w)`；
- projected preference `w_p`；
- cosine similarity 与 directional angle；
- Pareto front、hypervolume、sparsity；
- multi-GPU placement/sharding；
- PSL-MORL 或 hypernetwork。

因此当前结果只证明 preference-conditioned vector MO-TD3 baseline 能在 EvoRL + Brax + JAX GPU 路径正确执行，不代表完整 PD-MORL 已复现。
