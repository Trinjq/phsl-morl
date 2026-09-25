# Step 4.3: Official Preference Sampling

## 结论

**PATCH PASS。** PD-MORL 的 `process_count` 是可配置超参数，默认值为 10；**10 不是算法常量**。EvoRL 的正式 preference grid 现按 `K = process_count` 动态执行等价于 `np.array_split(w_batch, K)` 的划分。

本次只修正 preference subspace 与 Brax lane 映射。Episode lifecycle、rollout、SampleBatch、replay、vector reward、Actor、Critic、TD target、HER、interpolator、evaluation 和 multi-GPU sharding 均未修改。

## 官方源码证据

### 主训练入口

- `lib/utilities/settings.py:56-75`：Walker2d 配置块；`:71` 定义默认值 `process_count: 10`。
- `PD-MORL/train_Walker2d_MO_TD3_HER.py:60-63`：读取配置，并令 `PROCESSES_COUNT = args.process_count`。
- 同文件 `:113-115`：生成 preference grid，并执行 `np.array_split(w_batch_test, PROCESSES_COUNT)`。
- 同文件 `:119-129`：按 `range(PROCESSES_COUNT)` 动态创建 worker、分配第 `p_id` 个 subspace 并 join queue。
- 同文件 `:149-174`：统计数组长度、采样循环、learner 更新次数和 queue 完成循环均依赖 `PROCESSES_COUNT`。

### HER 下游依赖

- `lib/common_ptan/experience.py:155`：`ep_p` 长度为 `args.process_count`。
- 同文件 `:184`：HER warm-up 阈值为 `start_timesteps * process_count`。

因此 `process_count` 不只控制物理 CPU child-process 数，还控制逻辑 preference worker/subspace 数和相关训练计数。源码没有要求它必须等于 10。

### Key-pretraining 入口

- `PD-MORL/train_Walker2d_MO_TD3_HER_Key.py:175-194`：`main_parallel(process_count, reward_size)` 根据参数动态选择 key preferences 并创建相同数量的进程。
- 同文件 `:198-200`：局部设置 `process_count = 3`，表示该独立 key-pretraining 入口使用 3 个 key preferences。

该入口不使用主训练的 K-way `array_split`，也不能作为主训练必须固定为 10 的依据。

## 审查结果与修正

| 位置 | 修正前 | 官方语义 | 修正后 |
|---|---|---|---|
| `preference_wrapper.py` subspace sizes/starts | 固定 `[101, 100×9]` 与 10 个起点 | `np.array_split(grid, K)` | 根据 K 动态计算 quotient、remainder、starts 和 sizes |
| `official_preference_subspaces` | 无 K 参数 | subspace 数等于 K | 接收 `process_count=10`，返回 K 个 slices |
| `sample_official_preference` | 固定索引 10 个 subspaces | worker i 只从动态 subspace i 采样 | 接收 K，使用对应动态边界 |
| `brax.py` worker IDs | `lane_id % 10` | K 与物理 batch/device 分离 | `lane_id % process_count` |
| `brax.py` batch 合法性 | 无检查 | 等量 replicas 要求 `B % K == 0` | preference 模式创建 env 前显式校验 |
| tests | 只接受 10；preference batch 曾为 B=3 | K 可配置 | 覆盖 K=4/8/10/16 与指定 B/K 组合 |

`create_wrapped_brax_env(..., process_count=10)` 是当前最小配置入口。尚未存在 PD-MORL workflow 配置，因此没有向通用单目标 Brax YAML 增加一个当前不会被消费的字段。

## K 与 B

```text
K = process_count = logical preference worker 数 = preference subspace 数
B = Brax physical batched environment lane 数
logical_worker_id = lane_id % K
```

默认 `K=10`，但 K 可以是其他合法正整数。由于当前映射要求每个 logical worker 获得相同数量的 environment replicas，必须满足：

```text
B % K == 0
```

不满足时会在创建 Brax 环境前抛出包含 B、K 的 `ValueError`。GPU 数量不参与 grid 划分，也不改变 K；本阶段没有实现或修改设备 placement。

为避免产生空 subspace，当前二目标 1001-point grid 要求 `1 <= K <= 1001`。

## Grid 与采样语义

官方 Walker2d grid 是按第一目标权重递增的 1001 个点：

```text
[0.000, 1.000]
[0.001, 0.999]
...
[1.000, 0.000]
```

动态边界计算与 NumPy `array_split` 一致：若 `1001 = qK + r`，前 r 个 subspaces 的长度为 `q+1`，其余长度为 q。worker i 只用 JAX PRNG 从 subspace i 中均匀选择索引；JIT 内没有 NumPy RNG、host callback 或 device-to-host sampling。

## 未修改的数据链

- 每个 lane 在 episode reset 时从自己的 subspace 采样 preference；
- episode 内 preference 保持不变；
- done lane 推进 key 并重新采样，未 done lane 保持不变；
- transition 保存 step 前 episode 的 preference；
- preference 仍位于 `SampleBatch.extras.policy_extras.preference`；
- replay 仍保持 reward/preference/transition 对应关系；
- observation 不拼接 preference。

## 测试结果

实验室环境：Brax + JAX，单 GPU 0，`pytest --device gpu`。

### 参数化与数据链测试

```text
K = 4, 8, 10, 16:
  subspace count == K                         PASS
  slices == np.array_split(w_batch, K)        PASS
  worker IDs == 0..K-1                        PASS
  worker i samples only from subspace i       PASS

K=4, B=4                                      PASS
K=4, B=8                                      PASS
K=8, B=16                                     PASS
K=8, B=10                                     FAIL AS EXPECTED
```

轻量 GPU 集合结果：

```text
10 passed, 6 deselected, 1 warning in 26.80s
```

完整 preference pipeline 结果：

```text
16 passed, 8 warnings in 200.69s (0:03:20)
```

完整集合覆盖正式 sampler、动态 split、lane mapping、episode lifecycle、真实 Brax Walker2d rollout、replay preservation 和 JIT/GPU 执行。警告来自 JAXopt/Brax 维护状态及 Brax contact API deprecation，不是测试失败。

## 分类与边界

- 动态 `np.array_split(w_batch, process_count)`、默认 K=10、worker/subspace 一一对应：`SOURCE-FAITHFUL`。
- CPU processes 到 Brax batched lanes、`lane_id % K` 和 `B % K == 0` 校验：`FRAMEWORK-ADAPTATION`。
- `DEVIATION`：无。
- `BLOCKED`：无。

Step 4.3 参数化修正到此停止；未进入 Actor/Critic 或下一阶段。
