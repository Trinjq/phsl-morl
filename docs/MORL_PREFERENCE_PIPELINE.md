# Step 4.2: Preference Data Path

## 结论

**STEP4.2 PASS。**

本阶段建立了以下数据链：

```text
episode preference w
-> EnvState metadata
-> rollout transition
-> SampleBatch.extras.policy_extras.preference
-> flatten
-> replay add/sample
```

实验室单 GPU pytest 验证通过。没有修改 observation、Actor、Critic、TD target 或 learner 数学逻辑，也没有实现 HER、preference subspace 或正式 PD-MORL sampler。

## Preference shape 与约束

Walker2d 的目标数固定为 `L=2`：

- single preference：`(2,)`
- batched preference：`(B,2)`
- rollout preference：`(T,B,2)`
- flattened preference：`(T*B,2)`
- replay sample preference：`(batch_size,2)`

所有 preference 满足：

```text
w[i] >= 0
sum(w) ~= 1
```

Observation 与 preference 始终分离：

```text
obs:        (B,17)
preference: (B,2)
```

没有修改 Brax observation 或 observation space，物理环境也不读取 preference。

## 临时 sampler

`sample_preference(key, batch_shape)` 使用纯 JAX 生成：

```text
u ~ Uniform(0,1)
w = [u, 1-u]
```

对两个目标而言，这是连续 simplex 上的均匀采样。它支持单个和任意 batch shape，并在 device/JIT 内执行。

**DEVIATION / TEMPORARY：该 sampler 不是 PD-MORL 官方正式 episode sampler。**

后续必须替换为已经冻结的官方语义：

```text
discrete w_batch
+ np.array_split(w_batch, 10)
+ C_p=10 logical worker subspace sampling
```

本阶段没有实现上述 grid、subspace 或 10 logical workers。

## Episode 生命周期

`EpisodePreferenceWrapper` 放在现有 Episode/autoreset/batched wrappers 外层，因此它看到的是 physical termination 与 time-limit truncation 合并后的 `done`。

生命周期为：

1. reset 时，为每个环境 lane 建立独立 JAX PRNG key 并采样一个 preference；
2. 每个 step 前，当前 preference 是该 transition 所属 episode 的 preference；
3. `env_step` 在执行环境 step 前，把当前 preference 写入 `extras.policy_extras.preference`；
4. 环境 step 得到 combined done 后，用 `done[...,None]` mask 选择性更新 preference；
5. 未结束 lane 的 preference 和 preference PRNG key 都保持不变；
6. done lane 获得新 preference，供下一 episode 的 transition 使用。

没有 Python per-lane 分支，没有 host RNG，也没有因一个 lane done 而重采样全部 lanes。

## Preference 存储位置

运行时当前 episode preference 临时保存在：

```text
EnvState.info.preference
```

对应的 per-lane JAX keys 保存在：

```text
EnvState._internal.preference_key
```

transition 与 replay 中冻结的位置严格为：

```text
SampleBatch.extras.policy_extras.preference
```

`EnvState` 中的字段只是 wrapper/workflow metadata；Brax physics、observation 和 reward 公式均不依赖它。

## 修改文件

### `evorl/envs/wrappers/preference_wrapper.py`

新增：

- `sample_preference`：临时纯 JAX 两目标 simplex sampler；
- `EpisodePreferenceWrapper`：管理 batched per-lane episode preference 和 PRNG lifecycle。

### `evorl/envs/wrappers/__init__.py`

导出上述 wrapper 和 sampler，遵循现有 wrapper 组织方式。

### `evorl/envs/brax.py`

`create_wrapped_brax_env` 新增默认关闭的 `episode_preference=False` 参数。开启后只在已有 batched/autoreset 环境外添加 preference wrapper；默认单目标路径完全不变。

### `evorl/rollout.py`

`env_step` 将当前 `env_state.info.preference` 复制到 `policy_extras.preference`。若环境没有该字段，原 rollout 行为不变。

### `tests/test_morl_preference_pipeline.py`

覆盖 sampler、per-lane lifecycle、combined time-limit done、rollout、flatten、replay add/sample、transition-preference 对应关系、JIT、GPU 和 Step 4.1 回归。

## 未修改文件

- `evorl/sample_batch.py`：`extras` 已能承载 preference PyTree leaf，无需新建 `MORLSampleBatch`。
- `evorl/replay_buffers/replay_buffer.py`：固定尾维 `(2,)` 对现有 spec/tree set/tree get 完全透明。
- `evorl/utils/rl_toolkits.py`：collapse 只合并 `T,B`，保留 preference 最后一维。
- TD3 Agent、Actor、Critic、network、loss、target update：本阶段禁止修改，且 preference 尚未作为网络输入。
- Brax observation、action、physics、vector reward：Step 4.1 行为保持不变。
- evaluation/HV/Pareto/sparsity：尚未进入正式 MORL evaluation 阶段。

## Replay shape 与对应关系

```text
rollout reward       (T,B,2)
rollout preference   (T,B,2)
flatten reward       (T*B,2)
flatten preference   (T*B,2)
storage preference   (capacity,2)
sample reward        (batch_size,2)
sample preference    (batch_size,2)
```

合成 replay 测试给每个 transition 一个可恢复 ID，并验证 sample 后的 preference 和 vector reward 都与同一原 transition ID 对应，排除了 leaf 独立采样或错位。

## 测试与结果

### 环境依赖

按用户指示，在实验室环境安装：

```bash
/home/qiuquanj/miniforge3/envs/evorl/bin/python -m pip install pytest
```

结果：成功安装 `pytest 9.1.1`、`pluggy 1.6.0` 和 `iniconfig 2.3.0`。

### 轻量 GPU 检查

正式 Brax 测试前，单独运行 sampler 和 lifecycle 断言，结果：

```text
gpu
[CudaDevice(id=0)]
lightweight-preference-checks-PASS
```

### 正式 Step 4.1 + Step 4.2 GPU pytest

仓库 `tests/conftest.py` 默认使用 CPU，因此最终命令显式包含 `--device gpu`：

```bash
cd /home/qiuquanj/projects/evorl
CUDA_VISIBLE_DEVICES=0 \
  /home/qiuquanj/miniforge3/envs/evorl/bin/python -m pytest \
  tests/test_morl_vector_reward.py \
  tests/test_morl_preference_pipeline.py \
  --device gpu -q -s
```

关键输出：

```text
Use GPU!
Turn off jax GPU preallocation!
6 passed, 31 warnings in 666.70s (0:11:06)
```

警告来自 Brax/JAXopt 的维护或弃用提示，不是测试失败。运行期间通过 `nvidia-smi` 确认物理 GPU 0 有显存和计算利用率活动。

新增的 replay correspondence 测试随后单独运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
  /home/qiuquanj/miniforge3/envs/evorl/bin/python -m pytest \
  tests/test_morl_preference_pipeline.py::test_replay_preserves_preference_transition_pairs \
  --device gpu -q -s
```

结果：

```text
Use GPU!
1 passed, 1 warning in 6.68s
```

### 测试覆盖

| Test | 验证内容 | 结果 |
|---|---|---|
| 1 | single preference `(2,)`、非负、和为 1 | PASS |
| 2 | batched preference `(5,2)`、各 lane simplex、独立随机值 | PASS |
| 3 | 未 done lane 不变；单一 done lane 独立更新；transition 保留旧 w | PASS |
| 4 | Walker `episode_length=1` combined time-limit done 后更新下一 episode w；rollout `(2,3,2)` | PASS |
| 5 | flatten 后 reward 与 preference 均为 `(6,2)` | PASS |
| 6 | replay storage/sample 保留 `(batch,2)`，并保持 transition 对应关系 | PASS |
| 7 | sampler、mask resample、rollout、replay 路径均通过 JIT | PASS |
| 8 | pytest 明确 `Use GPU!`，物理 GPU 0 活跃 | PASS |
| 9 | Step 4.1 vector reward 和原 scalar Walker 回归 | PASS |

## DEVIATION / BLOCKED

- `DEVIATION / TEMPORARY`：连续 uniform-simplex sampler 仅用于 Step 4.2 数据链验证，不是官方 PD-MORL sampler。
- `FRAMEWORK-ADAPTATION`：用 batched JAX keys、mask 和外层 wrapper 替代 Python per-lane 状态管理；算法语义是每 lane 每 episode 固定 preference。
- Step 3/4.1 已记录的 Brax 与旧 MuJoCo dynamics 差异继续存在，本阶段未改变。
- BLOCKED：无。

## 尚未实现

- 官方 discrete `w_batch`；
- `np.array_split(w_batch,10)` 的逻辑 preference subspaces；
- `C_p=10` logical workers 到 1/3 GPU 的映射；
- preference 条件 Actor/Critic；
- HER、interpolator、projected preference、angle loss、vector Q、scalarization 与 MO-TD3 target。

Step 4.2 到此停止；未进入 Step 4.3。
