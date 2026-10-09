# PHSL Brax Key Artifact 预训练修正计划

## 1. 修正目标

本次修正的目标是把 PHSL 的 Brax Key Artifact 预训练改成“算法语义尽量对齐 PD-MORL Key Trainer、仅保留必要的 JAX/Brax 向量化差异”，并重新生成可信的 `interp_objs_walker2d_brax.txt`。

修正范围仅包括 Key Artifact 预训练、验证和产物生成，不修改正式 PHSL 主训练流程。当前主要差异包括 preference noise 缺失、学习启动时机错误、训练预算和更新密度不足、`next_obs` 构造不正确、评估协议不一致，以及 Key Trainer 超参数误用了部分主训练参数。

## 2. 修正步骤

### 第 1 步：固定目标协议和向量化边界

修改文件：

- `scripts/train_brax_key_solutions.py`

以 PD-MORL 的 `MO_TD3_HER_Key` 为目标协议，将 Key Trainer 参数改为：

```python
gamma = 0.99
batch_size = 100
replay_capacity = 500_000
policy_freq = 2
eval_episodes = 10
```

每个 key 的目标训练预算和行为阶段为：

- 约 2,000,000 条环境 transition；
- 约 25,000 条随机动作 transition；
- replay 达到 `2 * batch_size = 200` 后开始更新；
- 每新增一条 transition，对应一次 Critic 更新，即 UTD 约为 1。

当前 JAX 采样块大小为：

```text
num_envs * rollout_len = 64 * 4 = 256
```

为避免尾部数据被静默丢弃，使用能够被 256 整除的预算：

```python
random_action_steps = 25_088
total_env_steps = 2_000_128
```

这两个数分别是对原始 25,000 和 2,000,000 的最小向上取整。必须在代码注释和训练元数据中明确记录该向量化差异。

### 第 2 步：恢复训练期 preference noise

修改位置：

- `train_single_key()`；
- `warmup_action_fn()`；
- `train_action_fn()`；
- `eval_action_fn()`；
- replay `SampleBatch` 的构造位置。

在 `train_single_key()` 内增加训练 preference 采样函数：

```python
def sample_training_preference(key, batch_shape):
    noise = jnp.clip(
        jax.random.normal(key, (*batch_shape, reward_size)) * 0.05,
        -0.05,
        0.05,
    )
    preference = jnp.abs(w + noise)
    return preference / jnp.maximum(
        preference.sum(axis=-1, keepdims=True),
        jnp.finfo(preference.dtype).eps,
    )
```

具体修改如下：

1. `warmup_action_fn()` 继续生成随机动作，但同时为每条 transition 生成扰动后的 preference，并通过 `PyTreeDict(preference=...)` 返回。
2. `train_action_fn()` 使用本次扰动后的 preference 调用 Actor，而不是始终广播固定 key。
3. 将 rollout 中实际使用的 preference 写入 `SampleBatch.extras.preference`。
4. `update_step()` 从 `sample.extras.preference` 读取 preference，删除固定的 `broadcast_to(w, ...)`。
5. `eval_action_fn()` 始终使用未经扰动的原始 key，确保评估目标稳定。

归一化必须严格采用：

```text
abs(w + noise) / L1(abs(w + noise))
```

不能只除以绝对值之和而保留负分量，否则端点 key 可能产生负 preference，与 PD-MORL 源码不一致。

### 第 3 步：修正 replay 中的 `next_obs` 和 episode boundary

修改位置：

- `train_single_key()` 中训练环境的创建；
- `rollout()` 调用；
- replay batch 构造逻辑。

创建训练环境时启用真实后继状态记录：

```python
train_env = create_wrapped_brax_env(
    "walker2d",
    episode_length=horizon,
    parallel=num_envs,
    autoreset_mode=AutoresetMode.NORMAL,
    record_ori_obs=True,
    vector_reward=True,
)
```

调用 `rollout()` 时显式请求：

```python
env_extra_fields=("ori_obs", "termination", "truncation")
```

新增以下辅助函数：

```python
def replay_next_obs_and_done(trajectory, obs_dim):
    extras = trajectory.extras.env_extras
    required = ("ori_obs", "termination", "truncation")
    missing = [name for name in required if name not in extras]
    if missing:
        raise ValueError(f"rollout is missing required env extras: {missing}")

    next_obs = extras.ori_obs.reshape(-1, obs_dim)
    done = jnp.maximum(
        extras.termination,
        extras.truncation,
    ).reshape(-1)
    return next_obs, done
```

删除当前逻辑中的以下回退：

```python
flat_next_obs = ... if "ori_obs" in ... else flat_obs
```

当真实 `next_obs` 不存在时应立即报错，不能把当前 `obs` 当作 `next_obs` 继续训练。

### 第 4 步：拆分随机动作阶段和学习启动条件

修改函数：

- `make_train_chunk()`；
- `train_chunk()`；
- `update_step()`。

当前 `is_warmup` 同时控制“是否使用随机动作”和“是否更新网络”，这与 PD-MORL 不一致。将接口改为：

```python
def make_train_chunk(random_actions: bool, update_count: int):
```

建立三类训练块：

1. 首个 replay 填充块：使用随机动作，收集 256 条 transition，执行 57 次 Critic 更新；
2. 后续随机动作块：继续使用随机动作，每块执行 256 次 Critic 更新；
3. 策略动作块：使用 Actor 加动作探索噪声，每块执行 256 次 Critic 更新。

首块执行 57 次更新的依据是：官方实现在 replay 达到第 200 条 transition 时开始更新，因此在第一个 256-transition 块内，对应第 200 至第 256 条，共 57 次更新。

由此恢复以下训练语义：

- replay 满 200 后开始学习；
- 随机动作继续到约 25,000 条 transition；
- “开始学习”和“结束随机动作”不再是同一事件；
- 后续训练保持每条新 transition 对应一次 Critic 更新；
- Actor 继续由全局 `policy_freq = 2` 控制更新频率。

完整训练的预期更新计数为：

```text
Critic updates = 57 + (7,813 - 1) * 256 = 1,999,929
Actor updates  = floor(1,999,929 / 2) = 999,964
```

第一块在收集满 256 条数据后统一执行 57 次更新，是保留 JAX 静态批处理所需的框架适配；该差异需要写入元数据。

### 第 5 步：重写训练循环，保证完整执行预算

修改函数：

- `train_single_key()` 的训练循环。

显式计算并检查 chunk 数量：

```python
chunk_size = num_envs * rollout_len
total_chunks = total_env_steps // chunk_size
random_action_chunks = random_action_steps // chunk_size

assert total_env_steps % chunk_size == 0
assert random_action_steps % chunk_size == 0
assert total_chunks == 7_813
assert random_action_chunks == 98
```

训练顺序固定为：

1. 第一个随机动作块：57 次更新；
2. 剩余 97 个随机动作块：每块 256 次更新；
3. 后续 7,715 个策略动作块：每块 256 次更新。

删除当前由下面代码决定训练轮数的结构：

```python
num_rounds = learning_chunks // eval_chunk_freq
```

评估间隔不能决定训练是否继续。所有 `total_chunks` 必须完整执行，评估只能插入训练循环，不能导致不足一个评估区间的尾部 chunk 被丢弃。

训练结束时记录并断言：

- 实际环境 transition 数；
- 随机动作 transition 数；
- Critic optimizer step 数；
- Actor optimizer step 数；
- replay 当前大小；
- 完成 episode 数。

### 第 6 步：恢复按 completed episodes 触发的评估

修改位置：

- `train_chunk()` 的返回值；
- `train_single_key()` 的 host 训练循环；
- `evaluate()`；
- `record_evaluation()`。

在每个训练块中统计：

```python
completed_episodes = jnp.sum(trajectory.dones, dtype=jnp.uint32)
```

在 host 训练循环中累计 completed episodes，并在计数跨过每 100 个 episode 时调用 `evaluate()`。

评估协议固定为：

- 每次 10 个 episode；
- 使用未经扰动的原始 key；
- 不加入 preference noise；
- 不加入动作探索噪声；
- 记录 mean/std vector return；
- 使用 `w @ mean_vector_return` 计算 scalarized return；
- 保存 scalarized return 最优时对应的 vector return；
- 训练结束时额外执行一次最终评估，避免最后一个评估区间之后的策略没有被检查。

同时修正文档与实现不一致的问题：当前报告写的是 10 个 evaluation episodes，但旧代码实际配置为 16，修正后统一为 10。

### 第 7 步：增加最小测试、计数检查和产物元数据

新增文件：

- `tests/test_pd_morl_key_artifact.py`

测试至少覆盖：

1. 扰动后的 preference 全部非负；
2. 扰动后的 preference L1 范数为 1；
3. evaluation preference 严格等于原始 key；
4. replay 保存的 preference 与动作生成时使用的 preference 一致；
5. termination 和 truncation 均正确进入 `done`；
6. `next_obs` 来自 `ori_obs`，不再回退为当前 `obs`；
7. 总 transition 数为 2,000,128；
8. 随机动作 transition 数为 25,088；
9. Critic 更新数为 1,999,929；
10. Actor 更新数为 999,964；
11. 三个 key 的顺序严格为 `[0, 1]`、`[0.5, 0.5]`、`[1, 0]`。

修改 `main()`，使输出路径可以通过命令行参数指定，并在三个 key 全部成功后再写最终文本文件。除目标 artifact 外，只保存一个配套元数据文件，例如：

```text
interp_objs_walker2d_brax.metadata.json
```

元数据至少记录：

- 三个 key 及其顺序；
- 每个 key 的训练 seed；
- 全部训练超参数；
- 向量化取整说明；
- 实际 transition 和 optimizer step 计数；
- 最佳 step、vector return 和 scalarized return；
- evaluation history；
- 代码 Git commit；
- artifact SHA-256。

正式长训练前先执行一次缩小预算的 smoke test，检查 loss、回报、preference、计数和输出均为有限值，并确认 warm-up 随机动作阶段已经发生 learner update。

### 第 8 步：在远程服务器完整重训并生成新 Artifact

在远程 4090 服务器环境中执行完整的三个 key 训练。正式训练前记录：

- GPU 型号和 CUDA/JAX 版本；
- 当前 Git commit；
- 实际启动命令；
- 三个训练 seed；
- 输出目录。

不要直接覆盖当前 artifact。首先生成候选文件：

```text
outputs/key_retrain/interp_objs_walker2d_brax.candidate.txt
```

训练完成后，使用：

```text
scripts/validate_brax_key_solutions.py
```

验证以下项目：

- 文件形状严格为 `3 x 2`；
- 所有数值均为有限值；
- 三行与三个 key 的顺序严格对应；
- 每个 key 的最佳向量回报来自固定 key、无噪声评估；
- RBF 在三个锚点处能够还原对应方向；
- 实际 transition、Critic update 和 Actor update 计数与计划一致；
- 元数据中的参数、seed、Git commit 和 artifact 哈希完整。

只有全部验证通过后，才用候选文件替换：

```text
configs/artifacts/interp_objs_walker2d_brax.txt
```

最终交付物为：

- 新的 `configs/artifacts/interp_objs_walker2d_brax.txt`；
- 配套的 `interp_objs_walker2d_brax.metadata.json`；
- 三个 key 的最佳 vector return、scalarized return 和最佳训练步数；
- 新 artifact 的 SHA-256；
- smoke test 和完整验证结果。

## 3. 完成标准

满足以下条件后，本次 Key Artifact 修正即可停止，不再追加额外调参：

1. preference noise、replay preference、学习启动时机和更新密度已与 PD-MORL Key Trainer 对齐；
2. `next_obs` 和 episode boundary 已正确写入 replay；
3. 每个 key 完整执行约 2M transition，且更新计数可验证；
4. 评估按每 100 个 completed episodes 触发，并使用 10 个无噪声 episode；
5. 三个远程完整训练均成功完成；
6. 候选 artifact 通过数值、顺序、RBF 和来源验证；
7. 新的 `interp_objs_walker2d_brax.txt` 已写入正式配置路径并记录 SHA-256。
