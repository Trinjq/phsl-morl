# PD-MORL 控制与评估

当前基线入口：[协议与运行链路](PD_MORL_REPRODUCTION_PROTOCOL.md)。

## 动态锚点与控制边界

Host 在 chunk 边界检查采样计数，执行控制流程、日志和完整 Pareto 指标聚合。GPU key evaluator 返回 `[3,3,2]` 的 JAX 回报，`update_key_solutions_jax` 在设备侧完成三次评估均值、严格标量化改善替换与在线 L2 refit。初始化同样使用 L2，详见[插值器说明](PD_MORL_INTERPOLATOR.md)。

当前主线 key 根据各偏好组累计基础 transitions 触发，完整评估根据全局基础 transitions 触发，包含预填充，排除 HER 和评估。门槛、默认参数与恢复规则统一见[当前协议](PD_MORL_REPRODUCTION_PROTOCOL.md)。没有对应 transition 间隔配置项的调度对照继续使用 episode 规则：key 为所有组计数严格大于 `eval_cnt_ep`，full 为所有组计数严格大于 `100 * eval_cnt`。

## 评估口径

| 用途 | 偏好数 × repeats | 触发方式 |
| --- | --- | --- |
| 动态 key 候选 | 3 × 3 | 所有组达到下一基础 transition 门槛 |
| 完整训练评估及收敛曲线 | 201 × 3 | 全局基础 transition 门槛 |
| 训练结束评估 | 1001 × 3 | 结束钩子，由 run_final_evaluation 控制 |
| 独立 offline 评估 | 1001 × 6 | 显式调用 evaluate_offline |

repeat seed 为 `11 * repeat_index`，与训练 seed 区分。评估使用确定性 actor，累计至 done 或 500 步的未折扣二维回报。完整评估不替换 key，也不驱动 RBF 更新。独立 51 偏好评估入口已删除，历史 CSV 和绘图工具保留。

`source_hv/source_sparsity` 对每个偏好的平均回报取前沿并计算指标；
`mean_repeat_hv/mean_repeat_sparsity` 则先逐 repeat 计算指标再平均。
二者不可互换。HV 使用零参考点；sparsity 对非支配点按各目标排序后计算相邻差平方和，并除以点数减一。

## 记录与版本

当前 artifact、运行位置、文件解释和 checkpoint 限制只在主协议维护。历史 run metadata 中的单一 `l2` 字段曾与当时的在线 L1 源码不符，不据此重写旧实验结果。本次修改后的新运行明确记录 initial=L2 和 online=L2。

主线完整评估同时用于收敛曲线，结束时先保存再进行最终评估；旧工作区作为历史来源保留，其结果仍按当时版本解释。

## 旧 L2 微基准

此前 cuda:0/JAX 0.10.2 的 L2 微基准原始结果见 [JSON](../archive/docs/benchmark_pd_morl_jax_rbf_l2.json)：首次 JIT 0.545 s、固定输入 JAX refit 0.876 ms、1001 点前向 0.598 s；SciPy-L1/L2 refit 分别为 0.0576/0.0188 ms。它使用旧 artifact（SHA 前缀 `989e74d6`），仅描述当时微基准，不代表当前 v3/L1 的性能，也不作为整体 GPU 加速比。
