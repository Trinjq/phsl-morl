# PD-MORL GPU 并行基线：当前协议与运行入口

更新日期：2026-10-10。本文件是当前基线的统一入口；旧报告保留实验出处，不再充当当前运行指令。

## 研究路线与实现范围

当前工作是通过 EvoRL/JAX 与 Brax 建立 PD-MORL 的 GPU 并行基线。下一步在同一框架上复现 PSL-MORL，之后再实现基于 PSL-MORL 的新方法。目录、分支和旧报告中的 PHSL 名称是历史命名，不能据此判断已经实现了 PSL-MORL 或新方法。

当前主配置是 [pd_morl.yaml](../configs/experiment/pd_morl.yaml)，工作流为 `PDMORLGPUWorkflow`。保留 PD-MORL 的向量 Q、偏好条件策略、HER、方向角损失和动态 RBF 锚点机制；并行采样与优化预算另行明确，不声称逐次优化轨迹与原版相同。

## 代码与实验位置

| 位置 | 用途 |
| --- | --- |
| 本地 `E:/projects/pd-morl` | 唯一开发主线 |
| lab4090 `/home/qiuquanj/projects/pd-morl` | 同一主线的运行副本，使用现有 evorl Python 环境 |
| 分支 `codex/pd-morl-mainline` | 两端统一版本，不向远程托管平台自动推送 |
| 原 evorl、phsl-* 等工作区 | 已冻结为历史来源，原输出与 checkpoint 原位保留；完整位置见[归档说明](../archive/README.md) |

主线以本地最新实现为基础，整合独立收敛评估、锚点替换诊断、最终保存、恢复计数和历史裁剪修复。原始状态在两端 `../archive/pd-morl-20261010/` 保存源码快照及文件校验清单。通过可编辑安装确保 lab4090 的 `import evorl` 指向主线，而不是旧目录。

复现实验核对 `git_commit`、`runtime_source_sha256`、resolved config 与 artifact SHA。新的运行和验证写入主线的新输出目录，旧实验不重新标记为主线结果。

## 从 key 预训练到最终评估

1. **Key 预训练**：`scripts/train_brax_key_solutions.py` 分别训练三个单偏好策略，按各自偏好的评估标量回报保留最佳目标向量。当前 v3 元数据记录每个 key 2,000,128 条环境数据、batch 100、gamma 0.99、policy delay 2。该成本独立于主训练预算。
2. **Artifact 验证**：使用 `scripts/validate_brax_key_solutions.py` 核对文件、元数据、key 顺序及插值数值。验证通过不等于验证策略已达到全局最优。
3. **主训练**：加载固定 artifact，初始化 RBF，进行并行采样、HER 写入和 TD3 更新。
4. **动态锚点**：三个 key 各评估三次，先求回报均值，再按原始回报的严格标量化改善替换；每次触发都重新拟合。
5. **评估与结果**：区分 episode 触发的完整评估、固定 transition 间隔的收敛诊断、训练结束评估及独立 offline 评估。

当前 artifact：
[interp_objs_walker2d_brax_v3.txt](../configs/artifacts/interp_objs_walker2d_brax_v3.txt)，
[对应元数据](../configs/artifacts/interp_objs_walker2d_brax_v3.metadata.json)。

```text
keys: [[0,1], [0.5,0.5], [1,0]]
SHA-256: e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93
```

v1/v2、未版本化文件及被撤回的候选属于历史 artifact，不因文件名相似而替换 v3。

## 当前参数与计数口径

| 项目 | GPU 并行版 |
| --- | --- |
| 环境 | Brax Walker2d，观测 17，动作 6，episode 上限 500 |
| 二维奖励 | `(x_velocity + 1, 5 - sum(action**2))`；动作按环境边界处理 |
| 网络 | Actor 和 twin critic 均为两层 400 |
| 并行布局 | 10 个偏好组，每组 16 个环境，共 160 个环境 |
| 一个 rollout iteration | 每环境 4 步，共 640 条 base transitions |
| 一个 host chunk | 4 个 rollout iterations，共 2560 条 base transitions |
| 预填充 / 主训练总预算 | 4160 / 10,000,960；总预算已包含预填充 |
| 更新预算 | 每个 rollout 320 次 critic 更新，batch 512；每 10 次 critic 更新一次 actor 和 target |
| gamma / tau / 学习率 | 0.995 / 0.005 / 0.0003 |
| Replay / HER | 容量 2,000,000；超过 100,000 门槛后，每条新数据追加 3 个偏好重标记条目 |
| 主训练精度 | float32，`matmul_precision=highest` |
| 插值器归一化 | 初始目标逐行 L2；在线 refit 目标逐行 L1；原始回报保持原值 |
| key/full 触发计数 | 每个偏好组累计完成的 episode 数整除 16，再检查所有组是否超过阈值 |

环境预算只计 base transitions；HER 条目、评估 rollout 和 key 预训练单独核算。
默认 10M 配置的训练部分为 3905 个 host chunks、15620 个 rollout iterations、4,998,400 次 critic 更新和 499,840 次 actor/target 更新。host chunk 与 iteration 不能混用。

原版在优化启动后每条新 transition 约对应一次 critic 更新、batch 256；当前版为 0.5 次、batch 512。两者抽样条目总量相同不代表优化轨迹等价。按组平均 episode 触发也不同于原版十个单环境 worker 的触发频率。后续与 PSL-MORL 比较时应明确并统一这些预算和触发口径。

## 评估文件分别表示什么

| 文件或字段 | 含义 |
| --- | --- |
| `resolved_config.yaml`、`run_metadata.json` | 本次运行的配置、源码/设备和 artifact 信息 |
| `counters_summary.json` | 采样与更新计数、计时；部分计数按调度推导，不能代替独立执行验证 |
| `hv_history.csv` | episode 触发的 201 偏好 × 3 repeats 完整评估历史 |
| `hv_convergence.csv` | 独立 51 偏好 × 3 repeats 诊断，默认每 250,000 transitions 检查，在 chunk 边界执行 |
| `final_training_eval/results.json` | 1001 偏好 × 3 repeats 的训练结束评估 |
| `evaluate_offline()` | 独立 1001 偏好 × 6 repeats；普通训练不因此自动产生六次 offline 结果 |
| `source_hv` / `source_sparsity` | 先跨 repeats 求每个偏好的平均回报，再取非支配前沿并计算指标 |
| `mean_repeat_hv` / `mean_repeat_sparsity` | 各 repeat 分别计算前沿指标，再对指标求均值 |

HV 参考点为 `[0,0]`，二维目标按最大化处理。两种指标聚合口径不能混写；51 点诊断与 201/1001 点评估也不能直接拼成一条同口径曲线。不同训练 seed 的重复实验与一次评估的 repeats 是两件事。

旧 `run_metadata.json` 的 `key_objective_normalization: l2` 曾被硬编码；即使目录叫 `l1_online_*`，该字段仍可能错误。历史 JSON 保留原貌，判断当时算法应交叉核对源码及实验记录，不能单凭字段或目录名推断。新运行分别记录 initial=L2、online=L1。

## 常用入口

以下命令从实际选定的仓库根目录执行。GPU 编号只是示例，运行前选择可用设备；输出目录应为新的实验目录。

```bash
# 检查最终配置，不启动训练
python scripts/train.py --config-name experiment/pd_morl --cfg job

# 验证当前 v3 artifact，不重新预训练或替换文件
python scripts/validate_brax_key_solutions.py \
  configs/artifacts/interp_objs_walker2d_brax_v3.txt \
  --metadata configs/artifacts/interp_objs_walker2d_brax_v3.metadata.json

# 主训练；Hydra 的实际目录使用 hydra.run.dir 指定
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config-name experiment/pd_morl \
  seed=42 hydra.run.dir=outputs/pd_morl_gpu_l1_seed42_new

# 已有评估结果绘图，不重新评估
python scripts/plot_pd_morl_hv.py \
  --input outputs/RUN/hv_history.csv --output outputs/RUN/hv_history.png
```

`pd_morl_brax_reference` 用于原版调度对照，`pd_morl_key_replacement_diagnostics` 继承主配置并开启记录；其余旧配置移入 archive/configs/experiment。本次整理只执行短验证，不启动正式长训练。

## 保存、恢复与验证

当前主配置启用 `save_replay_buffer=true`，每 1564 个 rollout iterations 保存一次（1,000,960 条训练 transitions），保留最近两份。该间隔可被 fold_iters=4 整除；最终 checkpoint 在最终评估前保存，即使关闭最终评估也保存。这个持久化调整不改变学习更新预算。

恢复入口仍需显式指定完整 checkpoint，并检查原配置中影响算法的参数；目前 artifact SHA 校验不能替代完整协议一致性检查。旧 `save_replay_buffer=false` 的快照可用于评估，不能作为完整续训状态。

独立 key 预训练入口现在显式使用并记录 `matmul_precision=highest`。这只定义新运行行为，不回填历史 artifact 未记录的精度。Artifact 验证器只读，取消写回未版本化文件的旧 promote 入口。

- [x] 将已有生命周期、收敛评估及诊断实现整合到统一主线。
- [x] 明确 checkpoint 单位，并为主线启用完整续训状态。
- [x] 统一初始 L2、在线 L1 的实现说明、测试和新元数据。
- [x] 固定新 key 预训练的 matmul precision，保留历史精度信息的边界。
- [ ] 正式续训时核对影响训练的配置；当前恢复入口仍只做有限校验。
- [ ] 与 PSL-MORL 比较前冻结共用的数据预算、优化预算和评估触发口径。

2026-10-10 主线验证完成：

- 121 项 CPU 主回归通过；GPU 专项用例未运行。新增真实 checkpoint 路径测试先复现旧加载错误，公共加载入口修复后，10 项保存/恢复专项通过。
- 三份现行 Hydra 配置均可组合，v3 artifact 与默认元数据路径验证通过。
- 实际短训练采集 30 条数据并保存最终 checkpoint；关闭最终评估时保存正常。随后从同一完整状态恢复到 40 条数据，累计 critic=6、actor=3、host chunks=3，并保存 checkpoint 3。
- 修复了公共加载器对 CheckpointManager 的 step/default 目录解析；直接保存、管理器路径和管理器 API 的读取均已覆盖。第一次失败验证输出保留在新主线验证目录，不覆盖原实验。
- 两端 733 个运行相关文件逐字节一致；85 处仓库 Markdown 相对链接通过检查。
- 11 份整理前源码快照已校验；lab4090 的 10,985 个历史结果文件共 6,583,664,347 字节重新核对 SHA-256，均未变化。本地历史工作区的 39 个结果文件也未变化，根目录与已有图表目录另有 62 个保留文件纳入清单。

验证输出在 lab4090 的 `reports/mainline-validation/` 与 `outputs/mainline_verification_cpu*`。这是短链路正确性验证，不是正式训练结果或 GPU 性能测量。Brax 动力学与随机数流仍不同于原 MuJoCo 实现，不以论文 HV 数值作为接口正确性的唯一判据。

## 历史记录索引

旧报告中的“当前”“下一步”“必须”和 PASS 结论均限于其当时版本，不构成现行任务或运行授权。旧文件集中移入 archive；原位置映射见 [moves.json](../archive/moves.json)，原工作区继续保留。

| 历史阶段 | 记录 |
| --- | --- |
| 原版调度及 Step9 计数核对 | [原 Step9 协议](../archive/docs/PD_MORL_STEP9_PROTOCOL.md)、[Step9 验证](../archive/docs/PD_MORL_STEP9_0_1_VERIFICATION_REPORT.md) |
| GPU 并行改造与早期性能探索 | [GPU v2 报告](../archive/docs/PD_MORL_GPU_NATIVE_V2_REPORT.md)、[当时的后续决策](../archive/docs/PD_MORL_GPU_NATIVE_V2_NEXT_DECISION.md) |
| 初始化/在线统一 L2 的实验 | [JAX-L2 2M 报告](../archive/docs/PD_MORL_JAX_L2_2M_REPORT.md)、[旧微基准原始数据](../archive/docs/benchmark_pd_morl_jax_rbf_l2.json) |
| Key 候选撤回与 v3 修正 | [Refresh 验证记录](../archive/docs/KEY_ARTIFACT_REFRESH_VALIDATION.md)、[修正计划](../archive/docs/PD_MORL_KEY_ARTIFACT_CORRECTION_PLAN.md)、[当时的基线快照](../archive/docs/PHSL_FIX_BASELINE.md) |
| 旧稳定性、审查与阶段任务 | [稳定性复查](../archive/docs/PHSL_FIX_STABILITY_RECHECK.md)、[综合审查](../archive/docs/PD_MORL_COMPREHENSIVE_AUDIT_REPORT.md)、`archive/docs/guide/step*` 等阶段说明 |

当前模块细节见 [插值器](PD_MORL_INTERPOLATOR.md) 与 [控制评估](PD_MORL_CONTROL_EVAL.md)。新增实验只在其运行目录保存原始结果，并更新本文件必要的状态与链接，不再新增一套重复的主状态文档。
