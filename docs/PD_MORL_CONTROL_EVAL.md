# PD-MORL 控制与评估

## 控制边界

Host 只负责 episode 计数触发、日志和完整 Pareto evaluation。GPU-native Key
路径由 `BatchedPDMORLEvaluator.evaluate_keys_device` 返回 `[3,3,2]` JAX 回报，
随后 `update_key_solutions_jax` 在设备侧完成 repeat 均值、严格标量化替换和
RBF L2 refit。完整 HV、Sparsity、Pareto count 与 JSON 快照仍使用现有 Host
聚合路径；这不是声称整个 PD-MORL 已端到端 JIT。

Key trigger 仍使用每个逻辑 worker episode count 的严格 `>`，保持原有触发顺序、
3-repeat 评估和 counter/checkpoint 语义。raw Key returns 不归一化后再比较；只有
拟合目标逐行 L2 归一化。相等候选不替换，但每次触发都 refit。

## Artifact 固定

本次对照固定使用：

```text
configs/artifacts/interp_objs_walker2d_brax.txt
SHA-256: 989e74d631ba74572884c4f32b19e12c9317af847ffbc6c02bdde5ad39bfa209
keys: [[0,1], [0.5,0.5], [1,0]]
```

`interp_objs_walker2d_brax_v2.txt` 是不同 Artifact，不能在插值器迁移实验中
静默替换。旧 SciPy-L1、SciPy-L2 和目标 JAX-L2 的比较必须记录同一 Artifact、
seed、GPU 和配置。

## 实验室 benchmark

脚本 `scripts/benchmark_pd_morl_jax_rbf.py` 记录首次 JIT、稳态固定输入 refit、
1001 点前向，以及同输入的 SciPy-L1/L2 参考耗时。已在实验室 `cuda:0`、JAX
0.10.2 运行 1000 次 refit，结果写入
`docs/benchmark_pd_morl_jax_rbf_l2.json`：

| 项目 | 测量值 |
|---|---:|
| JAX 首次编译 | 0.545 s |
| JAX 稳态 refit | 0.876 ms |
| JAX 1001 点前向 | 0.598 s |
| SciPy-L1 refit | 0.0576 ms |
| SciPy-L2 refit | 0.0188 ms |

微型 4×4 线性系统上 SciPy 更快是预期结果；本实验的性能结论只针对设备侧
数据流和训练热路径，不以微小系统的单独耗时宣称加速。

## 训练验证

GPU smoke 使用相同 Artifact、seed=0、`PDMORLGPUWorkflow` 和实际 HER/Actor/Critic
路径；输出目录必须唯一，元数据记录 Git SHA、GPU、JAX、配置、Artifact SHA 和
counter。2M 配对训练应在 smoke 和数值 Golden 完成后再运行，比较 SciPy-L2 与
JAX-L2，不覆盖原 Artifact、不混用旧 checkpoint，也不将 HV 改善作为未经验证的
结论。
