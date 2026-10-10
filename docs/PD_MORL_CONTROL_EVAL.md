# PD-MORL 控制与评估

当前基线入口：[协议与运行链路](PD_MORL_REPRODUCTION_PROTOCOL.md)。

## 动态锚点与控制边界

Host 在 chunk 边界检查 episode 计数，执行控制流程、日志和完整 Pareto 指标聚合。GPU key evaluator 返回 `[3,3,2]` 的 JAX 回报，`update_key_solutions_jax` 在设备侧完成三次评估均值、严格标量化改善替换与在线 L1 refit。初始化使用 L2，详见[插值器说明](PD_MORL_INTERPOLATOR.md)。

当前并行版的控制计数为各偏好组累计完成 episode 数整除每组环境数（默认 16）；key 触发条件为所有组计数严格大于 `eval_cnt_ep`，完整评估条件为所有组计数严格大于 `100 * eval_cnt`。每次触发递增对应阈值计数。此处保留严格大于比较，但并行分组计数不等于原版单环境 worker 的触发频率。

## 评估口径

| 用途 | 偏好数 × repeats | 触发方式 |
| --- | --- | --- |
| 动态 key 候选 | 3 × 3 | key episode 阈值 |
| 完整训练评估 | 201 × 3 | full episode 阈值 |
| 独立收敛诊断 | 51 × 3 | 当前主线默认每 250,000 环境 transitions，在 chunk 边界执行 |
| 训练结束评估 | 1001 × 3 | 结束钩子，由 run_final_evaluation 控制 |
| 独立 offline 评估 | 1001 × 6 | 显式调用 evaluate_offline |

repeat seed 为 `11 * repeat_index`，与训练 seed 区分。评估使用确定性 actor，累计至 done 或 500 步的未折扣二维回报。独立收敛诊断不替换 key，也不驱动 RBF 更新，但会增加评估耗时。

`source_hv/source_sparsity` 对每个偏好的平均回报取前沿并计算指标；
`mean_repeat_hv/mean_repeat_sparsity` 则先逐 repeat 计算指标再平均。
二者不可互换。HV 使用零参考点；sparsity 对非支配点按各目标排序后计算相邻差平方和，并除以点数减一。

## 记录与版本

当前 artifact、运行位置、文件解释和 checkpoint 限制只在主协议维护。历史 run metadata 中的单一 `l2` 字段曾与在线 L1 源码不符，不据此重写旧实验结果。新运行明确记录 initial=L2 和 online=L1。

主线已经整合独立收敛诊断与训练结束保存修复；原服务器工作区作为历史来源保留，其结果仍按当时版本解释。

## 旧 L2 微基准

此前 cuda:0/JAX 0.10.2 的 L2 微基准原始结果见 [JSON](../archive/docs/benchmark_pd_morl_jax_rbf_l2.json)：首次 JIT 0.545 s、固定输入 JAX refit 0.876 ms、1001 点前向 0.598 s；SciPy-L1/L2 refit 分别为 0.0576/0.0188 ms。它使用旧 artifact（SHA 前缀 `989e74d6`），仅描述当时微基准，不代表当前 v3/L1 的性能，也不作为整体 GPU 加速比。
