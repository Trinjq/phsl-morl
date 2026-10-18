# PD-MORL GPU 并行基线：当前协议与运行入口

更新日期：2026-10-18。本文件是当前基线的统一入口；旧报告保留实验出处，不再充当当前运行指令。

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

主线以本地最新实现为基础，整合固定采样间隔的完整评估、锚点替换诊断、最终保存、恢复计数和历史裁剪修复。原始状态在两端 `../archive/pd-morl-20261010/` 保存源码快照及文件校验清单。通过可编辑安装确保 lab4090 的 `import evorl` 指向主线，而不是旧目录。

复现实验核对 `git_commit`、`runtime_source_sha256`、resolved config 与 artifact SHA。新的运行和验证写入主线的新输出目录，旧实验不重新标记为主线结果。

## 从 key 预训练到最终评估

1. **Key 预训练**：`scripts/train_brax_key_solutions.py` 分别训练三个单偏好策略，按各自偏好的评估标量回报保留最佳目标向量。当前 v3 元数据记录每个 key 2,000,128 条环境数据、batch 100、gamma 0.99、policy delay 2。该成本独立于主训练预算。
2. **Artifact 验证**：使用 `scripts/validate_brax_key_solutions.py` 核对文件、元数据、key 顺序及插值数值。验证通过不等于验证策略已达到全局最优。
3. **主训练**：加载固定 artifact，初始化 RBF，进行并行采样、HER 写入和 TD3 更新。
4. **动态锚点**：所有偏好组累计基础 transitions 达到 10k、20k……时，在 host chunk 边界触发。三个 key 各评估三次，先求回报均值，再按原始回报的严格标量化改善替换；每次触发都重新拟合。
5. **评估与结果**：当前主配置统一为每 250,000 基础 transitions 执行 201 偏好 × 3 repeats 的完整评估，同时用于收敛曲线；训练结束仍执行 1001 偏好 × 3 repeats。独立 offline 评估保留原入口。

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
| 一个 rollout iteration | 每环境 8 步，共 1280 条 base transitions |
| 一个 host chunk | 4 个 rollout iterations，共 5120 条 base transitions |
| 预填充 / 主训练总预算 | 4160 / 10,003,520；总预算已包含预填充 |
| 更新预算 | 每个 rollout 640 次 critic 更新，batch 512；每 10 次 critic 更新一次 actor 和 target |
| gamma / tau / 学习率 | 0.995 / 0.005 / 0.0003 |
| Replay / HER | 容量 2,000,000；超过 100,000 门槛后，每条新数据追加 3 个偏好重标记条目 |
| 主训练精度 | float32，`matmul_precision=highest` |
| 插值器归一化 | 初始与在线 refit 目标均逐行 L2；原始回报保持原值 |
| key 触发计数 | 每组 16 个环境合计的基础 transitions，包含预填充；所有组达到 10k、20k……时触发，默认等价于全局每 100k |
| full 评估触发 | 每 250,000 基础 transitions 一次；在跨过门槛后的 host chunk 边界执行，当前首个为 255,040 |

环境预算只计 base transitions；HER 条目、评估 rollout 和 key 预训练单独核算。
默认 10M 配置的训练部分为 1953 个 host chunks、7812 个 rollout iterations、4,999,680 次 critic 更新和 499,968 次 actor/target 更新。host chunk 与 iteration 不能混用。

2026-10-14 用户根据第二步筛选结果，选择将默认采样与更新参数改为 rollout_length=8、critic_updates_per_rollout=640；batch 512、actor delay 10、160 环境及 fold_iters=4 保持原值。此前总预算 10,000,960 在该布局下剩余半个 host chunk，因此向上对齐为 10,003,520，增加 2560 条基础 transitions。已有两轮 2M 实验保留当时的配置、计数和结果；修改默认配置不重新训练历史实验，也不自动启动新的 10M 训练。

修改后已同步 lab4090：15 项队列、导入及生命周期回归通过，三份 Hydra 配置组合通过，主配置与继承的锚点诊断配置均采用 8 / 640。两轮共 15 项历史实验继续按各自冻结的参数通过完成检查。

原版在优化启动后每条新 transition 约对应一次 critic 更新、batch 256；当前版为 0.5 次、batch 512。两者抽样条目总量相同不代表优化轨迹等价。当前 key 按组累计 transitions 触发，也不同于原版十个单环境 worker 的 episode 触发。后续与 PSL-MORL 比较时应明确并统一这些预算和触发口径。

2026-10-14 用户要求将主线三锚点更新改为基础 transitions 触发，配置项为 `key_update_interval_group_transitions=10000`。计数复用完整 checkpoint 中的 `worker_steps`，不包含 HER、评估或 key 预训练数据；默认首个触发边界为全局 101,440 条，10M 预算共触发 100 次。恢复时由所有组的最小累计计数推导下一未来门槛；重复边界不再触发，跨过多个门槛只评估当前权重一次，`eval_cnt_ep` 保留为实际 key 评估次数加一。三锚点 3 repeats、严格改善替换、在线 L1 refit、完整 201 × 3 评估与最终 1001 × 3 评估保持原值。没有该配置项的调度对照继续使用原 episode 触发规则。

验证：本次改动及此前删除独立 51 偏好评估入口的版本已同步 lab4090 主线，在现有 evorl 环境通过 78 项相关 CPU 回归，覆盖所有组达到门槛、重复边界去重、跨多个门槛仅评估一次、恢复后的下一门槛、严格改善替换与无改善 refit、完整评估、保存与恢复，以及三份现行 Hydra 配置组合。未启动正式训练。

## 评估文件分别表示什么

| 文件或字段 | 含义 |
| --- | --- |
| `resolved_config.yaml`、`run_metadata.json` | 本次运行的配置、源码/设备和 artifact 信息 |
| `counters_summary.json` | 采样与更新计数、计时；部分计数按调度推导，不能代替独立执行验证 |
| `hv_history.csv` | 当前固定 250,000 基础 transitions 间隔的 201 偏好 × 3 repeats 完整评估历史，同时提供收敛曲线 |
| `hv_convergence.csv` | 历史独立 51 偏好诊断；当前主配置不再生成或更新 |
| `final_training_eval/results.json` | 1001 偏好 × 3 repeats 的训练结束评估 |
| `evaluate_offline()` | 独立 1001 偏好 × 6 repeats；普通训练不因此自动产生六次 offline 结果 |
| `source_hv` / `source_sparsity` | 先跨 repeats 求每个偏好的平均回报，再取非支配前沿并计算指标 |
| `mean_repeat_hv` / `mean_repeat_sparsity` | 各 repeat 分别计算前沿指标，再对指标求均值 |

HV 参考点为 `[0,0]`，二维目标按最大化处理。两种指标聚合口径不能混写；历史 51 点诊断与 201/1001 点评估也不能直接拼成一条同口径曲线。不同训练 seed 的重复实验与一次评估的 repeats 是两件事。

2026-10-13 用户批准将完整评估改为固定 transitions 触发，并合并独立收敛评估。配置项为 `hv_history.interval_transitions=250000`；201 偏好网格、3 repeats、评估种子和指标算法保持原值，仍输出 `hv_history.csv` 与 `pd_morl_evaluations.jsonl`。正常 10M 预算约有 40 个完整评估点，实际采样步数取跨越阈值的 host 边界，首个为 252,480。恢复训练依据 checkpoint 的采样量初始化下一未来阈值；同一边界不重复评估，跨越多个门槛只评估当前权重一次，`eval_cnt` 仍只统计实际评估次数。恢复入口原有的未来历史裁剪逻辑继续有效。

三锚点在线替换、最终 1001 × 3 评估、key 预训练和 learner 数学未改变。未配置该 transition 间隔的旧参考复现配置仍采用原 episode 触发规则。用户随后要求实际删除独立 51 偏好评估：已移除训练 workflow 的旧入口、调度、自动绘图调用及主/基础配置中的 `convergence_eval` 项；历史 CSV 及其独立绘图工具保留用于查看旧结果，不再接入训练。此次仅修改代码，不重新启动已停止的旧 seed 计划或旧训练。

验证：本地及 lab4090 的 66 项评估/控制/HV 历史/训练结束/恢复回归均通过；新增检查覆盖门槛前后触发、重复边界去重、跨多个门槛仅评估当前权重一次、恢复后的下一阈值、201 × 3 及 HV/sparsity 历史写入，并确认不生成独立 51 点历史。Hydra 主配置组合通过；AST 对照确认网络、loss、GPU 采样/HER、在线锚点替换、输出字段与 offline 评估代码保持原值。本次验证未运行完整训练。

随后实际删除旧入口的版本已通过本地 74 项相关回归，包含 201 × 3 的批量评估复现检查；训练代码和配置中不再有独立收敛评估入口。lab4090 SSH 再次持续超时，删除版本的同步及服务器回归尚未完成；服务器上次已同步的版本已停用 51 偏好评估，但仍含旧入口。

2026-10-14 用户要求将在线重拟合也改为逐行 L2 归一化，初始化与在线路径现已统一。三锚点触发、严格标量化改善替换、原始回报、HER、损失公式和更新预算沿用此前设置；新运行与微基准元数据均记录 online=L2。旧 checkpoint 恢复时保留保存的插值器，下一次锚点触发后按当前 L2 规则重拟合。

验证：本地与 lab4090 的九个修改文件 SHA-256 一致；服务器现有 evorl 环境通过 85 项相关 CPU 回归，包含独立 SciPy-L2 对照、JIT/vmap、float32/float64、零向量与无改善 refit。v3 artifact 只读验证通过；微基准 CPU 短测输出 normalization=l2 且插值结果有限，该短测不作为 GPU 性能结果。未启动正式训练。

旧 `run_metadata.json` 的 `key_objective_normalization: l2` 曾被硬编码；即使目录叫 `l1_online_*`，该字段仍可能错误。历史 JSON 保留原貌，判断当时算法应交叉核对源码及实验记录，不能单凭字段或目录名推断。本次修改后的新运行分别记录 initial=L2、online=L2。

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
  seed=42 hydra.run.dir=outputs/pd_morl_gpu_l2_seed42_new

# 已有评估结果绘图，不重新评估
python scripts/plot_pd_morl_hv.py \
  --input outputs/RUN/hv_history.csv --output outputs/RUN/hv_history.png
```

`pd_morl_brax_reference` 用于原版调度对照，`pd_morl_key_replacement_diagnostics` 继承主配置并开启记录；其余旧配置移入 archive/configs/experiment。本次整理只执行短验证，不启动正式长训练。

## 保存、恢复与验证

当前主配置启用 `save_replay_buffer=true`，每 784 个 rollout iterations 保存一次（1,003,520 条训练 transitions，不含预填充），保留最近两份。该间隔可被 fold_iters=4 整除；最终 checkpoint 在最终评估前保存，即使关闭最终评估也保存。这个持久化调整不改变学习更新预算。

恢复入口仍需显式指定完整 checkpoint，并检查原配置中影响算法的参数；目前 artifact SHA 校验不能替代完整协议一致性检查。旧 `save_replay_buffer=false` 的快照可用于评估，不能作为完整续训状态。

独立 key 预训练入口现在显式使用并记录 `matmul_precision=highest`。这只定义新运行行为，不回填历史 artifact 未记录的精度。Artifact 验证器只读，取消写回未版本化文件的旧 promote 入口。

- [x] 将已有生命周期、收敛评估及诊断实现整合到统一主线。
- [x] 明确 checkpoint 单位，并为主线启用完整续训状态。
- [x] 统一初始与在线 L2 的实现说明、测试和新元数据。
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

2026-10-10 GPU 短训练与完整恢复验证通过（源码 `f439c47`）：

- lab4090 的 GPU 0 / RTX 4090，JAX 确认为 GPU 后端；保留 160 个并行环境、400×400 网络、batch 512、每 rollout 320 次 critic 更新和每 10 次 critic 更新一次 actor/target。
- 本次命令行短测覆盖参数：`replay_buffer_capacity=65536`、`env.max_episode_steps=16`、`start_timesteps=480`、`her_start_base_transitions=6720`、`convergence_eval.interval_transitions=2560`、`checkpoint.save_interval_steps=4`、`log_interval=4`、`key_diagnostics.enable=true`。默认主配置与训练代码未改动。
- 第一段采样 11,840 条，保存 checkpoint 12，critic/actor 实际优化器计数为 3,840/384；随后独立进程恢复并采样到 16,960 条，保存 checkpoint 20，计数增至 6,400/640。两段配置仅输出目录、总采样预算和恢复路径不同。
- 恢复入口逐项比较 checkpoint 的 149 个持久化数组，数值字节、形状与 dtype 全部一致，覆盖网络、优化器、replay、环境、PRNG 和计数。续训后 actor、critic、两套目标网络均改变，最终 checkpoint 与返回的完整训练状态也一致。
- 五个训练块的损失均有限；replay 最终包含 16,960 条原始数据与 30,720 条 HER 数据，共 47,680 条。检查了双目标奖励、HER 副本与偏好、有效数据的有限性及实际发生的 episode 截断。
- 在线 key 评估和插值器重拟合累计 5 次，归一化后的三组 key 的 L1 范数均约为 1。51 偏好 × 3 repeats 收敛记录由 3 行续接为 5 行，原 3 行完全保留；两段均完成 1001 偏好 × 3 repeats 最终评估与收敛绘图。
- 该短测未发生 key 替换，也未触发 201 偏好周期评估；未验证默认 500-step episode 下的长训练稳定性或与不间断训练的数值轨迹一致性。短测 HV 不用于正式性能比较。没有启动正式长实验。

原始日志、配置、评估和完整 checkpoint 保存在 lab4090 `/home/qiuquanj/projects/pd-morl/outputs/gpu_chain_smoke_20261010T154347Z/`。轻量证据副本在本地 `reports/gpu_chain_smoke_20261010T154347Z/`，逐项断言结果为 `restore_verification.json`。恢复观察脚本首次启动的导入路径错误及日志已保留；修正仅限该次验证脚本，最终两段训练均正常退出。

## 80 个 seed 的三卡队列

入口为 [run_pd_morl_seed_scan.py](../scripts/run_pd_morl_seed_scan.py)。默认 seed 为 `0, 7, 14, ..., 553`，共 80 个；每个从头训练至 10,003,520 条原始 transition，含预填充，合计 **800,281,600 条**。复用同一 v3 artifact，不从短测 checkpoint 初始化。正式 500-step episode、160 个环境、网络、HER、优化预算和全部评估保持主配置，仅额外开启 `key_diagnostics.enable=true`。seed 间隔 7 只是编号规则，不保证统计独立性或更好的性能；统计时保留全部预定 seed，不只挑选最高 HV。

以上为当前 8 / 640 的入口，新默认输出目录为 `outputs/pd_morl_80seeds_step7_10m_l8_c640`。本节后续 2026-10-11 的启动记录属于旧 4 / 320 配置及旧输出目录；新默认配置不混入该旧扫描。

每张 GPU 同时最多一个训练进程，空闲且达到显存门槛后领取下一个 seed，不把 80 个 seed 预先固定分配到三张卡。默认启动门槛为 8 GiB 空闲显存，每进程绑定 8 个不同 CPU 核；显存门槛可通过 `--min-free-gb` 调整，它不是显存预留或峰值保证，也不保证共享 GPU 没有算力竞争。当前不默认在同卡叠加多个训练进程，进一步提高并发应先测完整配置下的总吞吐和显存峰值。

新启动的子进程关闭 JAX 显存预分配，编译缓存按扫描目录和 GPU 隔离在 `jax_cache/pd_morl_seed_scan/<scan-hash>/gpu_N/`；同一卡上串行运行的 seed 可以复用编译缓存。设置 `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`，关闭发生临时文件竞争的 XLA 辅助磁盘缓存，保留普通 JAX 编译缓存。这是运行设置，不改变训练预算或算法精度。见 [JAX 缓存说明](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) 与 [显存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。耗时应依据正式配置实际完成的任务估计，不从 16-step 短测外推。

在 lab4090 执行以下命令。第一条只预览；第二条为后台启动。已有控制器运行时无需重复执行启动命令。

```bash
cd /home/qiuquanj/projects/pd-morl
/home/qiuquanj/miniforge3/envs/evorl/bin/python scripts/run_pd_morl_seed_scan.py --dry-run

mkdir -p outputs/pd_morl_80seeds_step7_10m_l8_c640
nohup /home/qiuquanj/miniforge3/envs/evorl/bin/python -u scripts/run_pd_morl_seed_scan.py \
  --gpus 0 1 2 --min-free-gb 8 --cpus-per-job 8 \
  >> outputs/pd_morl_80seeds_step7_10m_l8_c640/controller.log 2>&1 < /dev/null &
```

状态与各 seed 的最终 HV/sparsity 汇总在输出根目录的 `scan.json`；每个 seed 的原始输出位于 `seed_NNNN/attempt_NNN/`，控制台日志为 `launcher.log`。每个 seed 使用独立且跨恢复保持一致的 run_id，避免汇总时混淆不同 seed。控制器锁阻止同一输出目录被重复启动。

重新运行相同命令时，已完成 seed 经结果检查后跳过，其他 seed 从最近的完整 checkpoint 恢复到同一个总预算；无完整 checkpoint 时该 seed 从头重跑。每次重试使用新的 attempt 目录，旧日志、结果与 checkpoint 原位保留。失败任务自动重新排队，等待时间依次为 60、120、240 秒，最多 900 秒（初始值可通过 `--retry-delay-seconds` 调整）；等待期间其他 seed 继续派发，不再因累计三次失败停止队列。控制器持续调度，直到全部 80 个 seed 通过完成检查，或收到停止信号/发生控制器级错误；持续失败的根因仍需修复。

Ctrl-C、TERM 或 HUP 只终止本控制器创建或明确接管的训练进程，随后可重新运行相同命令续跑。若控制器被强杀而训练子进程仍存活，重启会核对 PID 对应的输出目录，并接管原进程，不重复启动该 seed 或占用其 GPU。接管进程的退出码不可获取，因此必须通过同样的完整结果检查才能标记完成。暂时无法读取 GPU 状态时保留训练进程并等待重试。

完成判据为：完整采样/更新计数、GPU 后端、有限的训练诊断和最终回报、40 个收敛阈值、1001 偏好 × 3 repeats 最终评估及最终完整 checkpoint 均符合预期。这只表示实验完成，不把 HV 高低作为成功筛选条件。训练代码/配置指纹或 seed 列表变化时拒绝混入原扫描目录；调度脚本自身不再计入训练指纹，CPU 核数、GPU 列表和显存门槛可在重启时调整。

队列已通过模拟子进程测试，覆盖不同完成顺序、显存等待、GPU 互斥、失败续跑及完成证据检查。2026-10-11 00:36（Asia/Shanghai）用户授权启动正式扫描，控制器 PID 为 4013647，首批 seed 0/7 分别绑定 GPU 0/1；GPU 2 当时空闲显存不足 8 GiB，等待资源后自动领取任务。当前进度以 `scan.json` 和 `controller.log` 为准；尚未完成三卡吞吐基准。

2026-10-11 00:54（Asia/Shanghai）修复共享缓存及自动重试：seed 7 的首次初始化因共享 XLA per-fusion 缓存临时文件缺失退出，尚未保存 checkpoint。替换控制器为 PID 4030497，原 seed 0/14/21 的训练 PID 4013652/4016398/4027655 保持运行；seed 7 已补入优先待运行队列，下一张卡空闲时从头重试。每卡仍最多一个训练，启动显存门槛保持 8 GiB。已运行进程保留原环境变量和旧缓存；隔离设置从新 attempt 生效，如原进程失败也会按新设置恢复。旧控制器与交接前状态保存在扫描目录 `controller_fix_20261010T165423Z/`，原始输出未改动。迁移前核对旧源码指纹一致，仅从新指纹排除调度脚本；训练源码和配置未变。6 项队列测试通过，覆盖缓存隔离、存活进程接管、失败自动续跑、退避等待和完整完成判据。

2026-10-12 01:07（Asia/Shanghai）用户要求停止这批 80-seed 扫描。控制器与三项训练均已退出；38 个 seed 已完成，266/273/280 标记为 interrupted，39 个未开始。已有结果、日志和 checkpoint 保留，后续 key 实验不自动恢复该扫描。

## 两组 key 预训练：seed 0–5

2026-10-12 用户授权重新训练两组 key，并确认沿用当前 v3 预训练配置，仅改变 seed 与 GPU 分配。

| 偏好 | GPU | 第一组 seed | 第二组 seed |
| --- | --- | --- | --- |
| [0,1] | 0 | 0 | 3 |
| [0.5,0.5] | 1 | 1 | 4 |
| [1,0] | 2 | 2 | 5 |

第一组三项并行，全部完成后启动第二组三项；每卡同一时间最多一个训练任务。每个 seed 独立初始化，沿用 64 环境、rollout 4、2,000,128 base transitions、25,088 随机动作 transitions、batch 100、replay 500,000、gamma 0.99、LR 3e-4、tau 0.005、actor delay 2、400×400 网络及原有偏好扰动/评估。每个后续 rollout 收集 256 条数据再进行 256 次 critic 更新；首批为 57 次。未增加角度损失或 HER，也未改变随机状态更新与计算逻辑。

入口为 `scripts/run_brax_key_groups.py`，新输出根目录 `outputs/key_groups_20261012_seed0_to5`。启动方式：

```bash
python scripts/run_brax_key_groups.py --output-dir outputs/key_groups_20261012_seed0_to5
```

调度状态在根目录 `run.json`；每组保存一个三行 candidate artifact 及 metadata，各 seed 保存 `progress.json`、完整评估历史及最佳/最终网络快照 `best_networks.msgpack`、`final_networks.msgpack`。快照包含 actor、双 critic、target 网络及对应 seed/偏好/步数/评估回报，供独立复评；不包含完整 replay/env/optimizer，因此不作为训练恢复 checkpoint。新候选保存在新目录，不自动替换现行 v3。

运行前检查：旧五项 key 回归通过，采样/损失/更新/评估计算函数与改动前 AST 相同；`--smoke` 使用每 seed 512 transitions 验证两组三卡调度、结果组装和快照读回，smoke 输出单独保存。

## Key 预训练采样节奏：64 × 1

2026-10-12 用户要求将 key 预训练从 64 环境 × 4 步改为 64 环境 × 1 步。当前脚本每轮采集 64 条 transitions，随后进行 64 次 critic 更新、32 次 actor 更新；actor delay 仍为 2。replay 满 200 条才开始学习：前三轮累计 192 条不更新，第四轮累计 256 条后进行 57 次 critic 更新、28 次 actor 更新，之后按正常 64/32 节奏执行。

为与上一轮直接比较，每个 seed 的总预算保持 2,000,128 transitions，随机动作预算保持 25,088；分别对应 31,252 轮和 392 轮。累计 critic/actor 更新仍为 1,999,929 / 999,964。其他网络、奖励、损失、偏好扰动和评估规则不变。上述 seed 0–5 已完成结果仍属于旧 64 × 4 配置；后续运行应使用新的输出目录。

各 seed 元数据新增 `sampling_schedule`，记录实际环境数、rollout 长度和学习启动节奏；组装 artifact 时核对三个 key 的节奏一致，并从这些记录写入组元数据。此前没有该字段的历史任务需使用当时脚本组装，避免被错误标记成新配置。

验证：lab4090 上 7 项 key 回归测试通过；`outputs/key_groups_64x1_smoke_20261012_164811` 的两组三卡短测全部完成。每个 seed 实际采集 512 条、更新 critic 313 次和 actor 156 次，两份 artifact 的 hash、最佳向量、64 × 1 元数据及最佳/最终权重均核对通过。完整预算重训尚未启动；旧两组目录保留了与其运行 hash 一致的原训练脚本快照。

2026-10-12 16:58（Asia/Shanghai）用户要求开始完整训练，按同样的 seed 0/1/2 和 3/4/5、偏好与 GPU 分配重新运行 64 × 1，便于与旧 64 × 4 比较。输出目录为 `outputs/key_groups_64x1_20261012_165825_seed0_to5`，控制器 PID 2035054；第一组三项已启动，第二组自动排队。每个 seed 采用完整 2,000,128 transitions 预算，未启用 smoke；该目录保留与运行 hash 一致的训练脚本快照。

2026-10-12 17:39:58（Asia/Shanghai）两组完整训练全部结束，耗时 41 分 33 秒；六项均正常退出，实际每 seed 2,000,128 transitions、1,999,929 critic 更新和 999,964 actor 更新。两份 candidate artifact hash 与最佳向量一致，最佳/最终权重均保存。seed 0–5 最终标量回报相对最佳分别下降 92.79%、0.31%、92.76%、0.84%、45.24%、96.82%。两组速度 key 的最佳回报向量均被同组均衡 key 支配；本轮 64 × 1 没有消除预训练退化，不据此声称采样节奏已解决稳定性问题。评估数据及两组曲线的本地副本在 `E:/projects/pd_morl_visualizations/key_groups_64x1_20261012/`。

2026-10-12 key 预训练代码检查发现：默认 GPU 计算下，同一权重、同一编译评估函数及 `PRNGKey(0)` 仍可能产生不同回报；seed 2 最佳策略两次速度回报相差约 214。开启 `--xla_gpu_deterministic_ops=true` 后，所测 seed 1/2/4 的最佳与最终六份权重重复误差均为零。命令行入口现于 JAX/backend 初始化前启用该 flag，并在各 seed 元数据记录 `xla_flags`。该修正通过七项 key 回归、CPU 命令行检查及 GPU 512-transition 短训练（critic 313、actor 156）；完整预算新训练未启动。

同一检查确认部分最终策略早停和共同 Q 偏乐观，但不是全部 seed 的一致特征；原版 learner 与当前 learner 的两次更新数值对照、35 项 key/golden 回归、实际 replay 自动重置与环形覆盖、向量评估终止累加均通过。gamma=0.99 的折扣优化与未折扣 artifact 选优仍有口径差异，原版也采用这一设置。此次未改变 gamma、更新比、损失、偏好或奖励，不能宣称确定性修正已经解决长期退化。两份新候选均未通过现有速度端点质量校验；现行 v3 保留。主报告和复核脚本在本地 `E:/projects/pd_morl_visualizations/key_pretraining_audit_20261012/`，远端 `reports/key_pretraining_audit_20261012/`。

## Key 预训练新配置：64 × 1，8/4 更新，batch 256

2026-10-12 用户指定重新训练两组 seed 0/1/2、3/4/5。每轮 64 环境 × 1 步，采集 64 条后更新 critic 8 次、actor 4 次，batch size 256，学习率 3e-4；继续启用 GPU 确定性设置。保留 replay 满两个 minibatch 才学习的条件：前七轮不更新，第八轮累计 512 条后开始完整的 8/4 更新。每 seed 仍为 2,000,128 transitions、25,088 随机动作，总 critic/actor 更新数为 249,960 / 124,980。其他网络、奖励、损失、gamma、偏好扰动及评估规则沿用此前配置。

短测改为每 seed 1,024 transitions、768 随机动作，覆盖不更新、随机动作期更新和策略动作期更新，预期实际 critic/actor 更新为 72 / 36。新元数据逐 seed 记录 batch size、学习率和每轮更新次数；历史元数据缺少这些字段时须用历史脚本组装，避免错标。新候选独立保存，不自动替换 v3；现有 `validate_brax_key_solutions.py` 的 batch 100 与旧计数断言用于原 v3 协议，不能据此判断本轮新配置是否正确。

2026-10-12 20:56:13（Asia/Shanghai）完成七项 key 回归与 `outputs/key_groups_64x1_c8_b256_smoke_20261012_204543` 的六个 GPU 短测，实际每 seed 为 1,024 transitions、72 critic / 36 actor 更新；两组 artifact hash、实际超参数、确定性 flag、GPU 分配及最佳/最终权重核对通过。完整训练启动于 `outputs/key_groups_64x1_c8_b256_20261012_205613_seed0_to5`，控制器 PID 2287759，seed 0/1/2 已运行，seed 3/4/5 自动排队。与短测一致的源代码 SHA256 为 `45e38cfebda0995b89118b4a6b13011a145f293b260f3d67bdc2e98ba11df2ab`，运行目录保存 `train_brax_key_solutions_used.py`；此处仅记录启动和短测验证，不代表完整训练已完成或稳定性已改善。

2026-10-12 21:30:04（Asia/Shanghai）上述 2M 两组均完成，耗时 33 分 51 秒。六个 seed 均正常退出，每 seed 实际 2,000,128 transitions、249,960 critic / 124,980 actor 更新，两份 artifact hash 已核对。最佳点二维目标及完整曲线在本地 `E:/projects/pd_morl_visualizations/key_groups_64x1_c8_b256_20261012/`；seed 5 最终标量回报相对最佳下降 46.90%。

用户随后要求其他设置不变，仅将采样预算改为每 seed 5,000,000 transitions，从头再训练两组 seed 0/1/2、3/4/5。5M 可被每轮 64 条整除，共 78,125 轮；仍跳过前七轮更新，预期 critic / actor 更新为 624,944 / 312,472。随机动作预算保持 25,088，64 × 1、每轮 8/4 更新、batch 256、学习率 3e-4、gamma 0.99、评估及确定性设置均沿用上一轮。旧 2M 结果保留，新结果单独保存。

2026-10-12 22:22:47（Asia/Shanghai）5M 两组重训已启动，目录为 `outputs/key_groups_5m_64x1_c8_b256_20261012_222247_seed0_to5`，控制器 PID 2394712；seed 0/1/2 分配 GPU 0/1/2 运行，seed 3/4/5 同样分配并自动排队。七项 key 回归通过，AST 对照确认训练脚本唯一变化是 `TOTAL_ENV_STEPS: 2_000_128 -> 5_000_000`；源代码 SHA256 为 `69ca15b01cc166632285d08931534eac2f2a451c84542e7060bf51f39dde0a53`，对应快照保存于本轮目录。此次仅调整预算，GPU 短测沿用已验证的相同计算与调度路径，未重复运行。

## Critic 更新比例筛选：三个 seed，三张 GPU

2026-10-14 用户要求自动运行第一轮更新比例实验：critic 每 rollout 更新 80、160、320 次，各配 seed 0、1、2，共九个独立任务。每个任务从头训练 2,000,960 条基础 transitions，包含 4160 条预填充；固定 160 环境 × 4 步、batch 512、actor delay 10、初始/在线 L2、三锚点每组 10k、完整评估每全局 250k，以及最终 1001 × 3 评估。对应更新比例为 0.125、0.25、0.5，critic 总更新数为 249,600、499,200、998,400，actor/target 为 24,960、49,920、99,840。

入口 `scripts/run_pd_morl_update_sweep.py` 复用已有 seed 队列，默认 GPU 0/1/2，每张卡最多两个本队列任务，每个任务分配四个 CPU。启动前检查可用显存，默认每次启动至少需要 8 GiB，尚未反映到显存查询的新启动任务在同一调度轮内预留该额度；每个 attempt 使用独立编译缓存并关闭预分配。`scan.json` 保存实验计划、源码指纹、PID、状态和最终 HV/sparsity；每个任务的日志、checkpoint 和评估曲线位于独立目录。相同命令重启会接管存活任务、跳过完成任务并从完整 checkpoint 恢复中断任务；参数或训练源码改变时要求使用新输出目录。仅终止本队列的子进程，不操作其他用户的任务。

```bash
# 在 lab4090 的主仓库根目录执行；先预览，再用新的输出目录运行。
PY=/home/qiuquanj/miniforge3/envs/evorl/bin/python
$PY scripts/run_pd_morl_update_sweep.py \
  --output-root outputs/pd_morl_critic_sweep_2m_l2_NEW --dry-run
$PY scripts/run_pd_morl_update_sweep.py \
  --output-root outputs/pd_morl_critic_sweep_2m_l2_NEW
```

本轮只运行上述九个 2M 任务，不自动扩展到第二轮采样节奏实验或 10M 训练。

验证与启动：lab4090 的八项队列回归通过。`outputs/pd_morl_critic_sweep_smoke_20261014_v1` 的六项 GPU 短测全部完成，三张卡各两个任务，保留正式网络、batch 512 和 replay 2,000,000；每项采集 6720 条、执行四个 rollout，critic/actor 分别为 320/32、640/64、1280/128，最终评估与 checkpoint 均核对通过，无失败。随后九项 2M 队列已启动于 `outputs/pd_morl_critic_sweep_2m_l2_20261014_025037`，控制器 PID 1900101；启动核对时每卡两项运行、其余三项排队。本段记录启动与短测验证，不表示完整预算实验已结束。

九项实验随后全部正常结束，无失败或重试。2026-10-14 按用户要求，通过 `scripts/upload_pd_morl_records_wandb.py` 将历史记录补录到已有 W&B `pd-morl` 项目的[本批实验分组](https://wandb.ai/2998847026-xiangtan-university/pd-morl/groups/pd_morl_critic_sweep_2m_l2_20261014_025037)。每个 run 保留 32 个训练诊断点、8 个 201 × 3 完整评估点、20 次三锚点评估记录及独立的 1001 × 3 最终评估，指标横轴为实际基础 transitions。配置、原始日志、诊断与评估文件作为 training-records artifact 上传，原始文件 SHA-256 在上传前后保持一致；导入缓存与九个 run 的 URL 清单保存在独立的 `outputs/wandb_imports/pd_morl_critic_sweep_2m_l2_20261014_025037`。W&B API 读回复核确认九个 run 均 finished，训练/评估点数、最终 HV/sparsity 与原记录一致。补录没有重新训练；训练原始起止时间另存于各 run summary。

本批实验的[可视化工作区](https://wandb.ai/2998847026-xiangtan-university/pd-morl?nw=l2u2iyo7ibq)单独保存了 14 个原生 W&B 面板：九项 HV 曲线、按 critic 更新次数分组的 HV/sparsity 均值与标准差、Pareto 点数、训练 loss/梯度/角度、三锚点评分，以及 1001 × 3 最终 HV/sparsity 对比和保存的 HV 图片。工作区只筛选本批九个 run，曲线横轴为实际基础 transitions，不重新训练或改写原记录。通过 W&B API 读回确认面板与分组配置已保存；浏览器未登录时无法查看私有项目，需使用已有 W&B 账号打开上述链接。

需要重建面板时，可在 evorl 环境使用官方 `wandb-workspaces` SDK（本次版本 0.4.13，独立安装于 `/tmp/pd_morl_wandb_workspace_sdk`）：

```bash
PYTHONPATH=/tmp/pd_morl_wandb_workspace_sdk $PY scripts/create_pd_morl_wandb_workspace.py \
  pd_morl_critic_sweep_2m_l2_20261014_025037
```

脚本将工作区 URL 保存在独立导入目录的 `visualization_workspace.json`，重复执行更新同一工作区。

### 第二步：采样节奏比较（六项 2M 任务已完成）

用户选择第一步的 `rollout_length=4`、`critic_updates_per_rollout=320` 作为基线。第二步比较收集与更新的交替频率，保持每条基础 transition 对应的更新量为 0.5 次 critic、0.05 次 actor；改变 rollout 长度时同比调整每次 rollout 的 critic 更新次数。

| 方案 | rollout_length | 每次采集的基础 transitions | critic 更新次数/rollout | actor 更新次数/rollout |
| --- | ---: | ---: | ---: | ---: |
| 更频繁交替 | 2 | 320 | 160 | 16 |
| 第一轮基线 | 4 | 640 | 320 | 32 |
| 更长采集间隔 | 8 | 1280 | 640 | 64 |

三组均使用 160 个环境、batch 512、actor delay 10、v3 artifact、初始/在线 L2、seed 0/1/2、2,000,960 条基础 transitions，评估规模和触发阈值沿用第一步。已通过现有计数函数核对：每项均执行 998,400 次 critic 和 99,840 次 actor 更新，预算满足完整 host chunk 的约束。

复用第一步三项 320 基线，仅新增两组共六项从头训练的任务，使用新的输出目录；三张 GPU 每卡最多两项。比较以最终 1001 × 3 的 HV 为主，同时报告各 seed 的 sparsity、均值/标准差、末期变化及运行耗时。并发耗时仅用于本轮运行成本参考。W&B 对比工作区引用原基线与新实验，不改动原 run 分组或训练记录。第二步不同时改变 replay batch 或 actor delay，也不自动启动 10M 训练。

2026-10-14 用户授权执行后，队列、导入和采样对比面板共 11 项检查通过。`outputs/pd_morl_sampling_smoke_20261014_082747` 的两项 GPU 短测同卡并发完成，各采集 9280 条基础 transitions，均执行 2560 次 critic、256 次 actor 更新；有限诊断、1001 × 3 最终评估和最终 checkpoint 均核对通过，无失败或重试。

六项 2M 队列随后启动于 `outputs/pd_morl_sampling_sweep_2m_l2_20261014_163225`，后台 pipeline PID 2622670。启动检查时六项均 running，三张 GPU 各两项、无失败。pipeline 顺序执行训练、W&B 历史记录补录及对比工作区保存；W&B 上传发生在六项全部完成后，面板按 `rollout_length` 分组并引用第一步三项 320 基线。日志见运行根目录的 `pipeline.log`，训练状态与各子进程 PID 见 `scan.json`。本段记录启动，不代表 2M 训练已经完成。

本轮队列命令为：

```bash
$PY scripts/run_pd_morl_update_sweep.py --sampling-lengths 2 8 \
  --gpus 0 1 2 --seeds 0 1 2 \
  --output-root outputs/pd_morl_sampling_sweep_2m_l2_20261014_163225
```

同一队列命令可续跑并跳过已完成任务。运行目录内的 `pipeline.sh` 包含后续 W&B 上传和工作区命令；若 pipeline 已运行，不重复启动。

六项实验于北京时间 2026-10-14 17:21 全部完成，无失败或重试；所有有限诊断通过。六项与复用的三项基线均实际执行 2,000,960 条基础 transitions、998,400 次 critic 和 99,840 次 actor 更新。最终 1001 × 3 评估的三 seed 均值及样本标准差如下，HV 单位为百万：

| rollout_length / critic 更新次数 | 最终 HV ↑ | 最终 sparsity ↓ |
| --- | ---: | ---: |
| 2 / 160 | 5.824 ± 1.108 | 1671 ± 1205 |
| 4 / 320，复用第一步基线 | 5.917 ± 0.427 | 6519 ± 7654 |
| 8 / 640 | 6.320 ± 0.309 | 2059 ± 776 |

8 / 640 的平均最终 HV 比 4 / 320 高 6.80%，平均 sparsity 低 68.42%，三个 HV 分别为 6.620M、6.335M、6.004M；其中两个 seed 高于对应基线，seed 2 比基线略低。2 / 160 的平均 sparsity 最低，但 seed 2 的 HV 为 4.550M，带来较大 seed 间波动。本轮 2M 筛选优先推荐 8 / 640，保留 2 / 160 的均匀性优势作为取舍；该结果不证明完整 10M 预算下仍然领先。新两组平均单任务耗时分别为 46.1、45.1 分钟；复用基线来自另一运行批次，不用这些并发耗时推断无竞争条件下的计算效率。

自动 pipeline 已完成 W&B 上传及[第二步对比工作区](https://wandb.ai/2998847026-xiangtan-university/pd-morl?nw=afnliq69fkv)保存，含六项新实验与三项原基线、14 个面板，按 rollout 长度分组。W&B API 读回复核确认六项新 run 均 finished 且 import_complete，最终 HV/sparsity 与原文件一致，面板配置及九项 run 设置已保存。原训练文件与原基线分组保持不变。

## 完整预算：seed 0、10、20、30、40、50、60

2026-10-14 用户授权上述七个 seed 分别执行完整主训练及训练结束评估。每项从头训练 10,003,520 条基础 transitions（含 4160 预填充），采用当前 160 环境、rollout_length=8、critic_updates_per_rollout=640、batch 512、actor delay 10、初始/在线 L2 和固定 v3 artifact；key 预训练成本不重复计入主训练。每项预期执行 7812 个 rollout、1953 个 host chunks、4,999,680 次 critic 和 499,968 次 actor/target 更新，保留每 250k 的 201 × 3 评估与结束时的 1001 × 3 评估。

输出根目录为 `outputs/pd_morl_seeds_step10_10m_l8_c640_20261014_191539`，后台 pipeline PID 2774178，队列控制器 PID 2774179。三张 GPU 每卡最多两个任务，启动核对分配如下；seed 60 在空位释放后自动领取可用 GPU，不预先绑定某张卡。

| GPU | 首批 seed |
| --- | --- |
| 0 | 20、40 |
| 1 | 0、30 |
| 2 | 10、50 |
| 排队 | 60 |

复用现有队列入口，启动参数为：

```bash
$PY scripts/run_pd_morl_update_sweep.py --sampling-lengths 8 \
  --seeds 0 10 20 30 40 50 60 --total-timesteps 10003520 \
  --gpus 0 1 2 --max-jobs-per-gpu 2 \
  --output-root outputs/pd_morl_seeds_step10_10m_l8_c640_20261014_191539
```

启动前 15 项队列、导入、图表及生命周期检查通过，七项预算与当前 Hydra 主配置一致。启动检查时六项 running、一项排队，无失败或重试，所有实际运行配置均开启最终评估。此处仅记录启动，完成状态以运行根目录的 `scan.json` 与 `pipeline.log` 为准。pipeline 在七项训练和最终评估全部通过完成检查后，自动补录 W&B 并保存七 seed 的 14 面板工作区；横轴覆盖完整 10M，完整评估曲线同时显示各 seed 与总体均值/标准差，最终指标按 seed 对比。

截至北京时间 2026-10-14 23:12，六项完整预算任务已完成，seed 60 的最新诊断为 3,972,160 条基础 transitions（约 40%），全队列无失败或重试。已完成六项的 1001 × 3 最终评估 HV 为 6.563 ± 0.835M（均值 ± seed 间样本标准差）；sparsity 均值为 70,451，中位数为 5,054。seed 50 的最终 HV 为 7.250M、sparsity 为 1,159、前沿点数为 209，是当前六项中两项主要指标最好的；seed 20 的最终 HV 为 4.972M、sparsity 为 401,837、前沿仅 9 点，显著影响聚合结果。

同口径 201 × 3 训练评估中，seed 20 在约 8M 时 HV 为 7.016M，终点降至 4.749M（下降 32.3%），中后期也有多次较大波动。六项平均训练评估 HV 从约 8M 的 6.871M 降为终点的 6.483M，说明延长训练不保证终点改善；这些数值不能与 1001 × 3 的最终评估直接混作同一口径。该统计仍为六项的阶段结果，等待 seed 60 后更新七项汇总；本轮没有其他超参数方案的完整 10M 对照，不能据此声称 8 / 640 在 10M 下仍优于其他配置。

已完成六个 seed 的最终 Pareto front 图片保存在两端 `outputs/analysis/pareto_front_10m_20261014/`：`pareto_front_comparison.png` 为六前沿对比，`pareto_front_by_seed.png` 为同尺度分图，另有各 seed 的独立 PNG。图片直接读取最终 1001 × 3 评估中跨回合平均的回报与非支配点，核对前沿与原始 `mean_returns[pareto_indices]` 完全一致。对比图仅画实际点；分图用灰色表示全部偏好平均回报、彩色表示前沿，坐标均为原始累计目标回报。原实验文件保持原位不变。

### 完整预算任务的提前发布

2026-10-14 用户要求在整个队列结束前就能查看 W&B 图片。上传入口新增 `--completed-only`，仅处理 `scan.json` 中已 complete 且通过完整计数/评估/checkpoint 检查的任务；默认仍要求全队列完成。相同分组和任务名沿用原 run ID，最后一项完成后的再次上传会跳过已导入记录。图片与最终指标来自原运行目录，不改写原文件；新导入记录按实际预算标记 2m 或 10m。

工作区入口新增 `--allow-partial`（与 `--seeds` 配合），只显示已完成导入的 seed，保留七个 seed 的固定配色，输出 `pending_seeds`；最后严格模式更新同一工作区，补齐剩余 seed。局部导入与六/七 seed 面板检查通过，共六项相关回归。

本次先发布已完成的 seed 0、10、20、30、40、50；seed 60 继续训练。为保持正在运行队列的源码指纹，实际调用 `outputs/upload_pd_morl_records_wandb_partial_20261014.py` 和 `outputs/create_pd_morl_wandb_workspace_partial_20261014.py` 的独立副本，导入日志位于 `outputs/wandb_imports/pd_morl_seeds_step10_10m_l8_c640_20261014_191539/partial_publish.log`。原训练 pipeline 继续在最后一项完成后补齐全部七项，不停止或重启 seed 60。

本次六项提前发布已全部完成，[图片与指标工作区](https://wandb.ai/2998847026-xiangtan-university/pd-morl?nw=ffokf4yt10d)包含六张原始 HV 图片、完整指标与 14 个面板，待完成 seed 为 60。W&B API 读回确认六个 run 均 finished/import_complete，六张 PNG 文件存在且非空，最终 HV/sparsity 与原文件一致；训练源码指纹未变，seed 60 进程仍存活。

## 随机 18 个新 seed 的完整预算队列

2026-10-15 用户授权从闭区间 0–200 中无放回随机选 18 个 seed，排除 0、10、20、30、40、50、60，三卡每卡最多两个任务。使用系统随机熵生成抽样 RNG seed `6798771474113542478`，再用 Python `random.Random(...).sample` 抽样；抽样方法、排除列表和执行顺序记录在新运行根目录的 `selected_seeds.json`，已重建核对一致。抽中的 seed 按数值排序为：

```text
18, 56, 62, 64, 71, 87, 95, 113, 121,
125, 136, 142, 147, 150, 153, 159, 162, 171
```

实际队列按抽样顺序执行：142、147、71、56、87、64、125、153、136、171、18、113、150、95、121、159、162、62。每项从头运行当前 8 / 640、batch 512、actor delay 10、160 环境和初始/在线 L2，复用 v3 artifact；预算 10,003,520 条基础 transitions，保留每 250k 的 201 × 3 完整评估、完整 checkpoint 和结束时 1001 × 3 最终评估。十八项总基础预算为 180,063,360 条，技术完成判据不以 HV 高低筛除 seed。

新输出根目录为 `outputs/pd_morl_random18_10m_l8_c640_20261015_000058`，后台 pipeline PID 3023174，控制器 PID 3023176。运行入口 `run_pipeline.py` 和 `run_queue.py` 位于该运行目录，使用已有共享队列；队列进度在 `scan.json` / `controller.log`，上传进度在 `pipeline.log`。每项完成后自动用 `--completed-only` 补录 W&B 原始图片与记录，并用 `--allow-partial --seeds ...` 更新本批对比工作区，网络上传失败会重试；训练进程独立于上传，原实验数据不覆盖。

上一批 seed 60 在本次启动时仍占用 GPU 1 slot 0。本批动态预留这个活跃槽位，待旧进程结束后释放，启动核对结果为 GPU 0：142、56；GPU 1：新 seed 71 与旧 seed 60；GPU 2：147、87。本批五项 running、十三项排队，无失败或重试；连同旧任务总计每卡两个训练进程。共享队列新增外部槽位预留回调，相关 16 项回归通过；两队列冻结的训练源码指纹均保持一致。本段记录启动，不表示十八项训练已经结束。

## 两项迁移差异修正（2026-10-17）

用户确认修正两项已确认的迁移差异。本次只修改初始化分布与双目标 HER 偏好的舍入处理，不调整学习率、更新预算、插值器、Bellman target 或正式评估方案。

- `PreferenceActor` 和 `TwinVectorCritic` 的所有 Dense kernel 改用 `jax.nn.initializers.variance_scaling(1.0, "fan_avg", "normal")`，与原版 PyTorch 非截断 Xavier 正态的分布及方差定义一致；bias 仍为零。Target 网络仍直接复制初始化后的在线网络。分布一致不代表 PyTorch/JAX 相同 seed 生成相同权重。
- `sample_her_preferences` 保留 abs-normal 随机采样、L1 归一化及三位小数舍入；双目标时，第二项改为 `1 - 第一项`，避免两项独立舍入使总和变为 0.999 或 1.001。保留显式 key、HER 条数、插入顺序及其他字段；非双目标分支保持原行为。本次没有开启全局 float64，也不声称逐项复现原版 NumPy float64 的舍入结果。

初始化修正也适用于复用这两个网络类的后续 key 预训练；现有 v3 artifact、历史 checkpoint、replay 与实验结果原位保留。恢复旧 checkpoint 时直接加载原网络权重，初始化修正不追溯改变权重；HER 修正仅作用于新生成的重标记条目，不重写已保存 replay。两项修正尚未通过新的长期训练验证其对 HV 稳定性的影响，本次不自动启动正式训练。

验证：lab4090 现有 evorl 环境 CPU 共 62 项相关回归分批通过，覆盖 HER、初始化、损失/梯度、更新调度、checkpoint、生命周期、PyTorch golden 参照，以及 key 网络保存/恢复；GPU 10 项通过，包含 HER、初始化及真实 Brax 采样、学习更新和 target 更新。新增百万组偏好检查覆盖数值有限、范围 [0,1]、分量之和为 1 和三位小数的 float32 表示容差；网络检查覆盖 Actor 与两套 Critic 的 Xavier 方差、非截断尾部及零 bias。首轮新增测试的小数网格容差由 1e-7 调整为 float32 epsilon（1.1920929e-7）后，HER 8 项 CPU 重跑通过。原始日志保留于两端 `outputs/analysis/migration_fixes_20261017/`；两份生产源码与修正前快照比较，仅包含上述两项授权变更。

## 当前正式预算与实验清理（2026-10-18）

用户明确要求回到 10M 正式预算，并删除此前延长预算的实验及其记录。主配置 `configs/experiment/pd_morl.yaml` 与批量入口 `scripts/run_pd_morl_seed_scan.py` 统一使用 10,003,520 条基础 transitions：预填充 4160 条后，按每个 host chunk 5120 条对齐。HER 副本及评估采样不计入预算。保持 rollout 8、critic 每轮更新 640 次、batch 512、160 lanes、actor delay 10、每 250k 的 201×3 评估及最终 1001×3 评估。本次未启动新训练。

已删除六 seed 的续训批次（含其 15M 中间阶段）、从零 20M 批次和 OFN 20M 批次：服务器实验目录、checkpoint、日志、上传副本及相关派生分析均已清理；W&B 对应 24 个 run、4 个专用工作区及上传的训练记录 artifact 已删除。本地对应 8 个实验/分析目录因永久删除被自动审批拦截，改用 Windows 回收站方式删除。原来的 25 个独立 10M seed 及其完整结果保留，服务器上两批 10M 数据的文件数量、大小和修改时间清单摘要在清理前后完全一致。W&B 非目标的 81 个 run 保留。上述明确删除要求覆盖此前对这些实验的保留约定；本文件不再保留已删除实验的运行入口、结果表或失效链接。

Critic OFN 的实现、配置开关和专用测试已移除，当前为普通两层 ReLU MLP 接线性向量输出。非截断 Xavier 初始化、HER 偏好舍入修正及插值器初始/在线 L2 均保留。恢复训练使用通用网络配置兼容检查，拒绝不支持且已启用的旧网络选项。删除 OFN 后的代码验证为 54 项 CPU 回归与 14 项 GPU 检查通过；这些检查不是延长预算的训练记录，日志保存在 `outputs/analysis/migration_fixes_20261017/baseline_removal_cpu.log` 和 `baseline_removal_gpu.log`。

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
