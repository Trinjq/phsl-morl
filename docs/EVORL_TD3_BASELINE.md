# EvoRL 原版 TD3 基线确认

## 结论

本阶段确认的核心价值是：在不修改 TD3 算法公式的前提下，EvoRL 原版 TD3 能通过官方训练入口在最小连续控制任务上完成初始化、采样、回放与参数更新，为后续 MORL 改造提供可信基线。

**Step1 单 GPU smoke test 通过。** 测试使用提交 `387d589c7412de7aa0f15dfe6e72ab1616dffb24`、实验室环境 `/home/qiuquanj/miniforge3/envs/evorl`（Python 3.11.16、JAX 0.10.2）、物理 GPU 2（NVIDIA GeForce RTX 4090）和 Brax `inverted_pendulum`。环境、rollout、网络、优化和 JIT 均使用 JAX GPU 路径，没有修改 TD3 算法、环境奖励或网络定义。此前的 CPU 运行仅保留为兼容性 smoke test，不作为 Step1 最终验收依据。

## 官方最小训练入口

官方通用入口是 `scripts/train.py`，TD3 配置中的 `workflow_cls` 指向 `evorl.algorithms.td3.TD3Workflow`。本次在实验室主机固定使用物理 GPU 2；`CUDA_VISIBLE_DEVICES=2` 使它在 JAX 进程内显示为 `CudaDevice(id=0)`。关闭整卡预分配只用于避免抢占空闲显存，不改变计算 backend：

```bash
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
python scripts/train.py \
  agent=td3 \
  env=brax/inverted_pendulum \
  'recorders=[log]' \
  random_timesteps=8 \
  learning_start_timesteps=8 \
  batch_size=8 \
  replay_buffer_capacity=64 \
  rollout_length=2 \
  num_envs=1 \
  num_eval_envs=1 \
  eval_episodes=1 \
  total_timesteps=24 \
  fold_iters=4 \
  eval_interval=4 \
  env.max_episode_steps=20 \
  save_replay_buffer=false \
  checkpoint.enable=false \
  hydra.run.dir=/tmp/evorl_td3_brax_gpu_smoke
```

选择 Brax `inverted_pendulum` 是因为它是仓库已有配置中状态和动作维度都很小、可直接被原版 `BraxAdapter` 支持的连续控制环境。更轻的 Brax `fast` 配置不能用于原版 TD3：它没有 `sys.actuator.ctrl_range`，而当前 `BraxAdapter.action_space` 依赖该属性。为保持原版代码不变，本次不修补 adapter，直接使用 `inverted_pendulum`。上述覆盖项只减少预填充、batch、buffer、episode 和总训练步数；`discount=0.99`、`tau=0.005`、target policy smoothing、双 critic、`actor_update_interval=2` 等 TD3 定义保持原值。

## 核心文件与职责

| 文件 | 核心类/函数 | 职责 |
| --- | --- | --- |
| `scripts/train.py` | `train`, `setup_recorders` | Hydra 入口；构建 workflow，初始化、训练、关闭资源，并配置日志/WandB recorder。 |
| `configs/config.yaml` | 全局配置 | 随机种子、JIT、checkpoint、recorder 和 Hydra 输出目录。 |
| `configs/agent/td3.yaml` | TD3 配置 | 选择 `TD3Workflow`，定义 rollout、buffer、优化器、网络、延迟更新和 soft update 参数。 |
| `configs/env/brax/inverted_pendulum.yaml` | 环境配置 | 选择 Brax `inverted_pendulum`，默认 episode 上限为 1000。 |
| `evorl/algorithms/td3.py` | `TD3Agent` | 初始化 actor/twin critics 及 target 副本；生成训练/评估动作；计算原版 TD3 actor 和 critic loss。 |
| `evorl/algorithms/td3.py` | `make_mlp_td3_agent` | 按配置创建 tanh actor、stacked critic 和可选 observation normalization。 |
| `evorl/algorithms/td3.py` | `TD3Workflow` | 组装 env、agent、optimizer、uniform replay buffer、evaluator；实现一次 TD3 training step。 |
| `evorl/algorithms/offpolicy_utils.py` | `OffPolicyWorkflowTemplate` | 用随机策略和初始 TD3 策略预填充 buffer；折叠多个 step；训练循环、evaluation、logging、checkpoint；补充 JIT 包装。 |
| `evorl/workflows/rl_workflow.py` | `OffPolicyWorkflow` | 创建 agent/env/optimizer/replay 状态，提供通用 evaluation，并将 `step`、`evaluate` 包装为 `jax.jit`。 |
| `evorl/networks/linear.py` | `MLP`, `make_policy_network` | actor MLP：observation 映射到 tanh 限幅的连续动作。 |
| `evorl/networks/linear.py` | `make_q_network`, `make_vmap_mlp` | 将 observation 与 action 拼接，并行计算两个独立 critic 的 Q 值。 |
| `evorl/replay_buffers/replay_buffer.py` | `ReplayBuffer`, `ReplayBufferState` | 预分配环形 buffer，维护写指针/有效长度，执行批量写入和均匀随机采样。 |
| `evorl/sample_batch.py` | `SampleBatch` | 统一承载 `obs/actions/rewards/next_obs/dones/extras` 的 JAX pytree。 |
| `evorl/rollout.py` | `env_step`, `rollout` | 调 actor、推进 env，并用 `jax.lax.scan` 形成 `[T, B, ...]` trajectory。 |
| `evorl/envs/env.py` | `Env`, `EnvState` | EvoRL 环境接口和纯数据状态定义。 |
| `evorl/envs/brax.py` | `BraxAdapter`, `create_wrapped_brax_env` | 将 Brax state 转为 `EnvState`，并用 JAX-compatible wrapper 实现 episode limit、向量化、autoreset 和原始 observation 记录。 |
| `evorl/evaluators/evaluator.py` | `Evaluator` | 用无探索噪声的 `evaluate_actions` 跑完整 episode，汇总 return 和 length。 |
| `evorl/recorders/log_recorder.py` | `LogRecorder` | 将 train/workflow/eval metrics 写入控制台和日志文件。 |
| `evorl/utils/rl_toolkits.py` | `flatten_rollout_trajectory`, `soft_target_update` | 将 `[T,B,...]` 压平为 replay 维度，并执行 Polyak target update。 |

## 一次 training step 的数据流

1. `OffPolicyWorkflow.setup` 初始化 actor、两个 critics、对应 target、两个 optimizer state、训练 env 和空 replay buffer。
2. `OffPolicyWorkflowTemplate._postsetup_replaybuffer` 先收集 `random_timesteps`，不足 `learning_start_timesteps` 的部分再由初始 TD3 actor 收集。本次初始化结束时 buffer 含 8 条 transition。
3. `TD3Workflow.step` 拆分 PRNG key，调用 `rollout` 连续采集 `T=2` 步。`compute_actions` 计算 tanh actor 输出，加入高斯探索噪声并裁剪到 `[-1,1]`。
4. trajectory 中的 `next_obs` 和 `dones` 被移除；真正用于 bootstrap 的 episode 结束前 observation 和 termination 分别保存在 `extras.env_extras.ori_obs` 与 `termination`。随后 `[T,B,...]` 被压平为 `[T*B,...]` 并写入环形 buffer。
5. 每个 `_sample_and_update_fn` 从 buffer 均匀采样。默认 `actor_update_interval=2` 时，先执行 1 次仅 critic update，再重新采样并执行第 2 次 critic update，之后执行 1 次 actor update。因此 actor 相对 critic 延迟更新，optimizer count 比为 `1:2`。
6. critic target 使用 target actor、裁剪后的 target policy noise、两个 target Q 的最小值和仅由 `termination` 控制的 bootstrap discount；当前两个 critics 对该 target 最小化平方误差。
7. actor 经当前 critic 求值，默认最大化第一个 critic 的 Q，即最小化 `-mean(Q1)`。
8. actor 更新后，actor target 和 critic target 都执行 `target = tau * online + (1 - tau) * target`。最后更新 workflow 的 timestep、episode 和 iteration 计数。

## Shape 与 replay batch

以下 shape 来自本次 Brax `inverted_pendulum`、`num_envs=1`、`rollout_length=2`、`batch_size=8` 的实际探针。`B` 表示并行环境数，`T` 表示 rollout 长度，`N` 表示 replay sample batch。

| 数据 | 单环境 shape | rollout shape | replay sample shape |
| --- | --- | --- | --- |
| observation | `(4,)` | `(T,B,4) = (2,1,4)` | `(N,4) = (8,4)` |
| action | `(1,)` | `(T,B,1) = (2,1,1)` | `(N,1) = (8,1)` |
| reward | `()` | `(T,B) = (2,1)` | `(N,) = (8,)` |
| done | `()` | `(T,B) = (2,1)` | 不存储 |
| next observation | `(4,)` | `(T,B,4) = (2,1,4)` | 不以 `next_obs` 字段存储 |
| `extras.env_extras.ori_obs` | `(4,)` | `(T,B,4) = (2,1,4)` | `(N,4) = (8,4)` |
| `extras.env_extras.termination` | `()` | `(T,B) = (2,1)` | `(N,) = (8,)` |

actor 对环境 batch 的输出为 `(B, action_dim) = (1,1)`，对 replay batch 的输出为 `(N,1)`。stacked twin critic 的输出为 `(N, num_critics) = (8,2)`。

实际 replay 元素结构为：

```text
SampleBatch(
  obs,
  actions,
  rewards,
  next_obs=None,
  dones=None,
  extras={
    policy_extras={},
    env_extras={
      ori_obs,
      termination,
    },
  },
)
```

## Smoke test 证据

官方入口在实验室 JAX GPU backend 上退出码为 0，运行期间持续报告 `Using JAX default device: cuda:0`，并产生以下关键结果：

- replay post-setup 正常开始并完成；
- iteration 4：`sampled_timesteps=16`，actor loss `-0.45143`，critic loss `0.52767`；
- iteration 8：`sampled_timesteps=24`，actor loss `-0.53822`，critic loss `0.78841`；
- 两次 evaluation 均完成，episode length 分别为 9 和 5；
- 训练持续 8 个 iteration，无训练异常。

同一物理 GPU 2 上的状态探针进一步确认：

- `jax.default_backend()` 为 `gpu`，且进程仅可见一个 `CudaDevice`；
- 初始化后 replay size 为 8，采样 batch 的各字段 shape 与上表一致；
- `TD3Workflow.step` 的底层包装类型为 `PjitFunction`，且暴露 JIT `lower` 接口；
- 一次 step 后 actor optimizer count 为 1、critic optimizer count 为 2；
- actor/critic target 最大参数变化分别为 `1.55e-06` 和 `3.04e-06`；
- 两个 target 与 `tau=0.005` 软更新公式的最大数值误差均为 `0.0`；
- 一次 step 后 `sampled_timesteps=10`、`iterations=1`，actor/critic loss 分别为 `-0.38993` 和 `1.66564`，均为有限值。

此前相同配置在 JAX CPU backend 也通过 8 个 iteration；该结果只说明代码具备 CPU 兼容性，已降级为辅助 smoke test。Step1 的最终结论以上述单 GPU 运行和 GPU 状态探针为准。

## 后续 MORL 最可能改动的位置

本阶段未加入 preference、vector reward 或 PD-MORL。若后续正式改造，优先检查以下边界：

1. `SampleBatch.rewards`、`OffPolicyWorkflowTemplate._setup_replaybuffer` 的标量 dummy reward，以及 replay buffer spec：决定 reward 从标量变为 objective vector 后的存储 shape。
2. `TD3Agent.critic_loss`：当前标量 reward 与标量 `min(Q1,Q2)` 直接相加，并将 target 广播到两个 critics；这是多目标 Bellman target 最直接的算法改造点。
3. `make_q_network` 与 `TD3NetworkParams`：当前 critic 末层每个 critic 只输出一个 Q；是否增加 objective 维，以及是否让 actor/critic 接收条件变量，应由未来 MORL 算法定义决定。
4. `TD3Agent.compute_actions`、`evaluate_actions` 和 `actor_loss`：若策略需要额外条件，输入组装和 actor objective 会在这里变化。
5. `BraxAdapter`、training wrappers 或未来 MORL env adapter：需要保证 vector reward 的 shape、dtype、termination 与 episode truncation 语义保持明确，并维持全 JAX 数据路径。
6. `Evaluator`、`EvaluateMetric`、`TD3TrainMetric`、recorder：当前 episode return 最终按标量求均值；多目标评估需保留 objective 轴并定义相应汇总指标。
7. `configs/agent/td3.yaml` 与环境配置：未来只添加算法确实需要的 objective/conditioning 参数，原版 TD3 配置保留为可复现 baseline。

`TD3Workflow.step` 的 rollout、uniform replay、延迟更新调度、JIT 和 target soft-update 框架本身可优先复用；是否需要改变它们应由具体 MORL 算法而不是基础设施预先决定。
