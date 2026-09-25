# Step 4.1: Vector Reward Data Path

## 结论

**STEP4.1 PASS。**

实验室环境 `/home/qiuquanj/miniforge3/envs/evorl` 上，使用单张物理 GPU（`CUDA_VISIBLE_DEVICES=0`）完成了 Brax `walker2d` 的

`reset -> action -> step -> vector reward -> rollout -> SampleBatch -> replay add -> replay sample`

完整 smoke test。JAX 报告 `backend: gpu`，可见设备为 `[CudaDevice(id=0)]`。最终代码没有加入 preference、vector Q、scalarization、HER 或任何 learner 改动。

## Reward 定义与 shape

修改前，普通 Brax `walker2d` 保持标量奖励：

- 单环境：`reward.shape == ()`
- batched 环境：`reward.shape == (B,)`

显式创建 `vector_reward=True` 的 Walker 后：

- 单环境：`reward.shape == (2,)`
- batched 环境：`reward.shape == (B, 2)`
- rollout：`rewards.shape == (T, B, 2)`
- replay sample：`rewards.shape == (batch_size, 2)`

动作先按官方实现裁剪：

```text
a = clip(action, -1, 1)
r[0] = x_velocity + 1.0
r[1] = 5.0 - sum(a^2)
reward = stack([r[0], r[1]], axis=-1)
```

目标顺序固定为：

1. forward-speed objective；
2. energy-efficiency objective。

没有复用 Brax 默认 scalar reward，没有使用默认 `0.001` control-cost coefficient，也没有 normalization、reward clipping 或 objectives 求和。

## 影响面检查

| 数据链位置 | 修改前假设/行为 | Vector reward 兼容性 | 处理 |
|---|---|---|---|
| Brax Walker | 原生输出 scalar reward | 必须产生固定尾维 2 | 在 adapter 内显式、device-side 构造两目标 reward |
| `BraxAdapter` / wrappers | adapter 透明传递 reward；`EpisodeWrapper.reset` 把 episode return 写死为标量零 | adapter 透明；episode return 初始化不兼容 JIT scan carry shape | 仅把 episode return 初始化改为 `zeros_like(state.reward)` |
| `env_step` / `rollout` | 直接写入 `env_nstate.reward`；`lax.scan` stack 时间维 | 已透明支持 trailing objective axis | 不修改 |
| `SampleBatch` | `rewards` 是任意 JAX reward PyTree leaf | 已能表达 `[N]` 和 `[N,2]` | 不修改，不新增 MORL batch 类型 |
| `flatten_rollout_trajectory` | `jax.lax.collapse(x, 0, 2)` 只合并 `T,B` | 保留后续 objective axis | 不修改 |
| `ReplayBuffer` | 按 sample spec 为每个 PyTree leaf 添加 capacity 首维 | 固定 leaf shape `(2,)` 自动成为 `(capacity,2)` | 不修改；用 vector-reward dummy sample 初始化 |
| replay add/sample | `tree_set` / `tree_get` 操作首维索引 | objective axis 不变 | 不修改 |
| episode recorder | episode return 原先按 scalar 初始化 | 需随 reward shape 初始化 | 修改为 shape-transparent 初始化；不对目标求和 |
| scalar evaluator / workflow reduction | 多处以 scalar episode return 为语义并做 `.mean()` / `.flatten()` | 不能作为正式 MORL evaluation 使用 | 本阶段不修改；Step 4.1 不运行或冒充正式 MORL evaluation |
| Actor、Critic、TD target | 当前 TD3 是 scalar-return learner | 属于后续阶段 | 不修改 |

## 修改文件与必要性

### `evorl/envs/brax.py`

- 新增最小 `MOWalker2dAdapter`。
- `reset` 只把外层 reward 初始化为 `(2,)` 零向量；observation、Brax internal state 和 termination 不变。
- `step` 在 JAX device path 中裁剪 action，并使用 Brax step 后的 `metrics.x_velocity` 构造精确 reward vector。
- `create_brax_env` 与 `create_wrapped_brax_env` 新增默认关闭的 `vector_reward` 参数。
- 对非 `walker2d` 使用该参数会立即报错，防止把 Walker 公式误用于其他环境。

默认值为 `False`，因此所有现有单目标调用保持原行为。

### `evorl/envs/wrappers/training_wrapper.py`

`EpisodeWrapper.reset` 的 episode-return 初值从 scalar `jnp.zeros(())` 改为 `zeros_like(state.reward)`。对 scalar reward 结果仍为 scalar；对 vector reward 结果为 `(2,)`，从而保证 `jax.lax.scan` carry 的 PyTree leaf shape 不变。

### `tests/test_morl_vector_reward.py`

覆盖单 transition 数值、batched lanes、rollout、SampleBatch flatten、replay add/sample、JIT、finite 检查和原 scalar Walker 回归。文件可由 pytest 收集，也提供无需 pytest 的直接断言入口，因为实验室环境未安装 pytest。

## 未修改文件及原因

- `evorl/sample_batch.py`：类型与 PyTree 结构已经支持任意固定尾维 reward。
- `evorl/rollout.py`：直接保留 reward leaf，`lax.scan` 自动产生 `T` 维。
- `evorl/utils/rl_toolkits.py`：`flatten_rollout_trajectory` 只 collapse 前两个维度，不丢失 objective axis。
- `evorl/replay_buffers/replay_buffer.py`：按 spec 分配和按首维索引，对 `(2,)` leaf 透明。
- `evorl/algorithms/td3.py`：本阶段禁止修改 TD3 数学逻辑；当前 scalar learner 也没有用于 vector reward 训练。
- Actor/Critic/network/config：preference 和 vector Q 尚未进入本阶段。
- evaluator、Pareto/HV/sparsity：正式 MORL evaluation 明确留到后续阶段；没有把两个目标求和或平均后冒充 return。

## SampleBatch 与 replay 兼容方式

实际 shape 流为：

```text
env step reward       (B, 2)
rollout rewards       (T, B, 2)
collapse T,B          (T*B, 2)
replay single spec    (2,)
replay storage        (capacity, 2)
replay sample         (batch_size, 2)
```

因此无需 `MORLSampleBatch`，也无需重写 replay。objective axis 从环境到 sample 始终是最后一维。

## 单目标兼容策略

- `vector_reward=False` 是默认值。
- 普通 Walker batched reset/step 的 reward 仍为 `(B,)`，没有变为 `(B,1)`。
- `zeros_like(state.reward)` 对 scalar reward 仍返回 scalar，不改变现有 episode recorder schema。
- 单目标 TD3 Agent、network、loss、target update 和 workflow 均未修改。

## 测试

### 静态检查

Windows 工作区：

```powershell
python -m compileall -q evorl tests/test_morl_vector_reward.py
python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['evorl/envs/brax.py','evorl/envs/wrappers/training_wrapper.py','tests/test_morl_vector_reward.py']]; print('syntax ok')"
git diff --check
```

结果：语法检查 `syntax ok`；`git diff --check` 无 whitespace error。工作区 Python 和实验室环境均未安装 pytest；实验室环境也未安装 ruff，因此没有为了本阶段增加依赖。

### 最终 GPU smoke test

同步最小改动到实验室干净的同提交工作区后执行：

```bash
cd /home/qiuquanj/projects/evorl
CUDA_VISIBLE_DEVICES=0 /home/qiuquanj/miniforge3/envs/evorl/bin/python -u tests/test_morl_vector_reward.py
```

关键输出：

```text
backend: gpu
devices: [CudaDevice(id=0)]
all vector-reward checks passed
```

各断言结果：

| Test | 验证内容 | 结果 |
|---|---|---|
| 1 | 单 transition；reward `(2,)`；精确速度/能耗公式；finite | PASS |
| 2 | `B=3`；reward `(3,2)`；三种 action 的能耗分别计算且互不混淆 | PASS |
| 3 | `T=4, B=3`；rollout rewards `(4,3,2)` | PASS |
| 4 | flatten 后 `(12,2)`；replay sample `(5,2)` | PASS |
| 5 | reward step、batched step、rollout、replay add/sample 均经 `jax.jit`；无 tracer error/host callback；reward finite | PASS |
| 6 | 默认 scalar Walker batched reset/step reward 均保持 `(2,)`（这里的 `2` 是 env batch，不是 objective axis） | PASS |
| 7 | `jax.default_backend() == "gpu"`，可见 `CudaDevice` | PASS |

首次 Brax Walker generalized pipeline 的完整 XLA 编译耗时约 19 分钟；编译和执行最终均成功。另一次在 GPU 2 上的预验证也通过，但最终验收以上述 GPU 0、最终代码版本为准。

## DEVIATION / BLOCKED

- Reward 数据链：无算法语义 deviation；公式和 objective 顺序严格来自已冻结的官方 PD-MORL Walker 实现。
- Framework adaptation：reward 在 EvoRL Brax adapter 内以 JAX array 构造，并通过现有 PyTree rollout/replay 传递。
- 环境动力学：仍继承 Step 3 已记录的 Brax generalized pipeline 与旧 MuJoCo RK4 数值动力学差异；本阶段没有扩大或修正该差异。
- BLOCKED：无。

Step 4.1 到此停止；未进入 Step 4.2。
