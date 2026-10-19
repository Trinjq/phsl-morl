> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# GPU-native PD-MORL 修改计划

## 1. 目标与边界

本任务要解决的问题是：在保留 PD-MORL 单个偏好条件策略、向量价值学习和 Pareto 前沿覆盖能力的前提下，将原本面向 CPU 多进程的执行结构改造成适合 JAX、Brax 和 GPU 的批量训练结构，以缩短达到同等 Pareto 质量所需的墙钟时间。

新增模式命名为 `pd_morl_gpu_native`。现有 source-faithful PD-MORL 必须保持行为不变，继续作为正确性与论文复现基线。GPU-native 模式允许改变采样规模和 optimizer 调度，但不得暗示其训练轨迹与原始实现逐步等价。

保留的 PD-MORL 核心语义包括：

- preference-conditioned actor；
- vector-valued twin critic；
- scalarization 与 pessimistic whole-vector target selection；
- directional-angle loss；
- preference relabeling HER；
- online interpolator；
- key/full Pareto evaluation。

GPU-native 主动引入的算法偏差必须在报告中明确列出，包括大规模并行环境、大 batch、不同 optimizer update 节奏，以及 chunk 边界上的控制面更新延迟。

## 2. 阶段 0：只读审计与基线测量

首先阅读并追踪以下现有实现，不立即修改代码：

- `evorl/algorithms/mo_td3.py`；
- PD-MORL 与 TD3 workflow；
- `evorl/replay_buffers/replay_buffer.py`；
- `evorl/replay_buffers/her.py`；
- Brax adapter 与 preference wrapper；
- `evorl/evaluators/pd_morl.py`；
- interpolator；
- `configs/experiment/pd_morl_walker_reproduction.yaml`。

输出一份简洁的数据流记录，至少包含：

- environment、transition、replay sample、preference、vector reward 和 vector Q 的 shape；
- rollout、replay add/sample、learner update、evaluation 和 interpolator update 的调用关系；
- 所有训练 inner loop 中的 `np.asarray`、`jax.device_get`、Python 循环、Python 条件、SciPy 调用和隐式同步；
- compilation time、训练吞吐、evaluation time 和峰值显存的 faithful 基线。

编译时间必须单独报告，稳态吞吐测量前必须 warm up 并调用 `block_until_ready()`。

## 3. 阶段 1：批量 evaluation

先处理当前最明确的瓶颈。保留现有 `PDMORLEvaluator`，新增独立的 GPU-native batched evaluator，避免改变 faithful 路径。

要求：

- Key evaluation 的 `3 preferences × 3 repeats` 使用 9 个并行 lane；
- Full evaluation 的 `201 × 3 = 603` 个 episode 按固定 `eval_batch_size` 分块；
- 最后一批通过 padding 和 mask 保持静态 shape，避免额外 JAX 编译；
- 每个 batch 只允许一次 Python→JAX 调用和一次最终 device→host 同步；
- 已完成 lane 使用 done mask 冻结状态和 return，批次运行到所有有效 lane 完成；
- serial 与 batched evaluator 使用相同的 preference、repeat 和 seed 映射。

通过逐项数值一致测试后，GPU-native workflow 才能使用该 evaluator。faithful workflow 继续使用原 evaluator。

## 4. 阶段 2：环境并行与 rollout block

保持 PD-MORL 的 10 个 logical preference groups，但允许每组包含多个环境：

```text
num_envs = 10 × envs_per_preference_group
```

每个 environment lane 固定属于一个 logical preference group；episode reset 时只从所属 preference subspace 重新采样 preference。GPU 数量不得改变 group 数量或 lane 到 group 的映射。

**GPU-native 正式实验的总环境交互预算默认固定为 10,000,000 global valid transitions，增加 parallel env 数量不得自动扩大总预算。该预算包含 prefill 和训练期 base transitions，不包含 evaluation transitions 或 HER relabeled entries。正式配置必须保证剩余预算能被静态训练 chunk 整除，否则启动时直接报错。**

**10 个 logical preference groups 与物理 sampling lanes 严格区分；`envs_per_preference_group` 只增加并行采样副本，不增加 PD-MORL preference groups。**

新增完全 JIT 的 rollout block，使用 `jax.lax.scan` 一次执行 `rollout_length` 个环境步：

```text
trajectory leaf shape: [T, N_env, ...]
flattened replay input: [T × N_env, ...]
```

rollout block 内禁止逐步返回 Python。环境 step、preference reset、transition construction 和 replay 写入应尽可能保留在 device 上。

`envs_per_preference_group` 不预设正式默认值；先提供可运行的小规模值，最终值由 benchmark 与学习效果共同决定。

## 5. 阶段 3：复用 global replay 与现有 HER

保持一个逻辑 global replay，不按 preference group 分区。

现有 replay sampling、index generation、batch construction 和 HER 已主要由 JAX、`vmap`、`scan` 与 reshape 完成。第一版应直接复用这些实现，并验证它们能正确接受 `[T × N_env, ...]` 输入。只有 profiling 证明存在 host transfer、Python 循环或明显瓶颈时才修改。

HER 必须继续满足：

- base transition 与 relabeled transition 的值和 shape 正确；
- preference relabeling 不使用 Python transition 循环；
- **第一版 HER warm-up 使用独立配置 `her_start_base_transitions`，按 global valid base transition count 定义，默认阈值为 100,000，不随 sampling lane 数自动放大；**
- replay 中可以采样到所有 preference groups 的数据。

## 6. 阶段 4：最小 GPU learner 调度

GPU-native 模式不再复刻“每轮 10 条 transition 后顺序执行 10 次 optimizer update”的 CPU 调度。

第一版只新增两个 learner 配置：

- `replay_batch_size`；
- `critic_updates_per_rollout`。

不在第一版加入 `grad_minibatches`、`minibatch_size` 或梯度累积。只有单个大 batch 因显存不足或 profiling 表明梯度累积有明确收益时再增加。

每次实验必须报告数据复用强度：

```text
replay_samples_per_transition
    = critic_updates_per_rollout × replay_batch_size
      / (rollout_length × num_envs)
```

改变 `num_envs`、`rollout_length` 或 batch size 时，不得忽略该比率的变化。最终参数不能只根据 transitions/s 决定，还必须比较学习曲线和达到目标 HV 的墙钟时间。

**第一版 actor/target delayed update 继续以 global critic optimizer step 为计数单位，并复用现有 `actor_update_interval` 表达原 `policy_freq`，避免同时引入额外调度变量。** 现有 faithful learner 的 update mask 和 optimizer 顺序不得改变。

## 7. 阶段 5：训练 chunk 与控制面边界

将多个 rollout/learner round 合并在一个编译后的训练 chunk 中，减少 Python↔JAX 往返。checkpoint、低频 logging、Pareto 指标计算和 SciPy interpolator refit 可以保留在 host。

必须明确处理以下语义变化：如果 key/full evaluation 阈值在 chunk 内被跨过，GPU-native 模式允许在 chunk 结束后处理，但这会延迟 interpolator 生效时间，属于 algorithmic deviation，必须记录实际最大延迟。

第一阶段继续使用现有正确的 SciPy interpolator，并将其限制在低频控制面边界。暂不实现 JAX-native RBF；只有 profiling 证明 interpolator host boundary 成为显著瓶颈时，才增加固定 shape 的 JAX 实现及数值等价测试。

## 8. Benchmark 方法

避免一次执行完整笛卡尔积。采用分阶段扫描：

1. 固定 learner 参数，扫描 `envs_per_preference_group = 16, 32, 64, 128`；
2. 固定最佳可行环境规模，扫描 `replay_batch_size = 512, 1024, 2048, 4096`；
3. 对少量候选组合扫描 `rollout_length = 1, 4, 8`；
4. 对最终候选运行短学习曲线，而不是只测空吞吐。

至少记录：

- environment transitions/s；
- optimizer updates/s；
- replay samples/s；
- JAX compilation time；
- steady-state wall-clock time；
- evaluation time；
- GPU utilization 与峰值显存；
- `replay_samples_per_transition`；
- 相同 environment-step budget 下的 HV、sparsity 和 EUM；
- 达到预设 HV 阈值所需的墙钟时间。

最终选择依据是 time-to-quality，而不是单独最大化 transitions/s。所有 benchmark 记录硬件、软件版本、seed、warm-up 和测量区间。

## 9. 最小正确性测试集

优先复用现有 PD-MORL、HER、parallel 和 golden tests，不重复创建已经覆盖相同行为的测试。

新增或扩展的测试只覆盖新行为：

- lane 到 10 个 logical preference groups 的稳定映射；
- 每个 group 的 preference 只落在对应 subspace；
- group 数量不随环境数或 GPU 数变化；
- `[T, N_env, ...]` rollout shape 与 flatten 后的 replay 内容；
- global replay 覆盖所有 preference groups；
- 大批量 HER shape、mask 与 relabeled values；
- serial evaluator 与 batched evaluator 的逐项一致性；
- padded evaluation batch 不影响有效结果；
- GPU-native smoke run 的 loss、gradient、Q、return 均无 NaN/Inf；
- faithful 配置的现有 regression/golden tests 保持通过。

每个阶段先运行相关 unit test 和小规模 smoke test，不直接启动完整 10M transition 训练，也不为了通过测试降低原有容差或断言标准。

## 10. 配置与交付物

保留现有 `pd_morl_walker_reproduction.yaml` 作为 faithful 配置，不复制一份内容相同但名称不同的配置。新增：

- `configs/experiment/pd_morl_brax_gpu_native.yaml`。

如果后续确实需要统一命名，可以在不破坏现有入口的前提下再增加 `pd_morl_brax_faithful.yaml`，但第一版不为名称创建重复配置。

最终交付：

- 独立的 GPU-native workflow；
- 独立的 batched evaluator；
- 一个 GPU-native 配置；
- 必要的新测试；
- `docs/PD_MORL_GPU_NATIVE_REPORT.md`。

所有阶段的实现结果、测试结果、benchmark 数据、失败项、最终参数选择和结论必须汇总到上述报告中，以文档形式交付；不得只保留在终端输出、日志或聊天记录中。

报告必须区分：

1. 保留的原始 PD-MORL semantics；
2. 不改变算法目标的 GPU framework adaptation；
3. 为吞吐主动引入的 algorithmic deviation；
4. faithful 与 GPU-native 的正确性、吞吐和 time-to-quality 对比。

## 11. 停止条件

满足以下条件后停止继续增加架构和配置：

- faithful regression/golden tests 全部通过；
- batched evaluator 与 serial evaluator 在规定容差内一致；
- GPU-native smoke run 无数值异常；
- benchmark 证明 GPU-native 稳态吞吐优于 faithful baseline；
- 短学习曲线表明加速不是通过停止学习或过度降低 update 强度获得；
- 报告已经清楚披露所有算法偏差和适用边界。

除非 profiling 或上述验证失败，不新增自定义 replay、第二套 HER、梯度累积框架、JAX-native RBF 或多 GPU 通信。
