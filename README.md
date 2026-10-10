# PD-MORL：EvoRL / Brax GPU 并行基线

当前阶段：建立 PD-MORL 并行基线；随后在同一框架上复现 PSL-MORL，再实现基于 PSL-MORL 的新方法。

**唯一开发主线**：本地 `E:/projects/pd-morl`，lab4090 `/home/qiuquanj/projects/pd-morl`，分支 `codex/pd-morl-mainline`。从[当前协议](docs/PD_MORL_REPRODUCTION_PROTOCOL.md)了解算法、预算和评估口径。

## 运行

在 lab4090 的主仓库根目录，使用已有 evorl 环境：

```bash
cd /home/qiuquanj/projects/pd-morl
PY=/home/qiuquanj/miniforge3/envs/evorl/bin/python
$PY -c 'import evorl; print(evorl.__file__)'
$PY scripts/train.py --config-name experiment/pd_morl --cfg job
$PY scripts/validate_brax_key_solutions.py configs/artifacts/interp_objs_walker2d_brax_v3.txt

# 选择可用 GPU；每次使用新的输出目录
CUDA_VISIBLE_DEVICES=0 $PY scripts/train.py \
  --config-name experiment/pd_morl seed=42 \
  hydra.run.dir=outputs/pd_morl_seed42_NEW
```

默认配置的完整 checkpoint 可恢复到新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/train.py \
  --config-name experiment/pd_morl \
  +resume_from_checkpoint=/absolute/run/checkpoints/1564 \
  hydra.run.dir=outputs/pd_morl_resume_NEW
```

若原实验覆盖了主配置参数，续训时保留相同覆盖项。历史不含 replay 的快照只用于评估。

导入路径应位于本仓库。若环境尚未指向主线，执行 `$PY -m pip install -e . --no-deps --no-build-isolation`。从其他目录运行脚本也应使用这一安装环境。

## 目录与入口

| 路径 | 用途 |
| --- | --- |
| `evorl/` | 统一算法与框架实现，供 PD-MORL 及后续 PSL-MORL 共用 |
| `configs/experiment/pd_morl.yaml` | 唯一默认 GPU 并行训练配置 |
| `configs/experiment/pd_morl_brax_reference.yaml` | 10 个单环境 worker 的调度对照，不作为默认主线 |
| `configs/experiment/pd_morl_key_replacement_diagnostics.yaml` | 继承主配置，只开启锚点记录 |
| `configs/artifacts/` | 当前 v3 artifact 与对应元数据 |
| `scripts/train.py` | 主训练及完整 checkpoint 恢复 |
| `scripts/train_brax_key_solutions.py`、`validate_brax_key_solutions.py` | 独立 key 预训练与只读验证 |
| `scripts/plot_pd_morl_hv.py`、`plot_pd_morl_convergence.py` | 两种评估历史的独立绘图 |
| `scripts/analyze_key_replacements.py` | 分析锚点替换诊断 |
| `scripts/export_pd_morl_hv_history.py`、`aggregate_pd_morl_runs.py` | 已有评估记录导出与整理 |
| `scripts/benchmark_pd_morl_jax_rbf.py` | 当前在线 L1 插值器微基准 |
| `tests/` | 当前实现的回归测试 |
| `outputs/` | 本主线的新实验；与历史实验目录分开 |
| [archive/](archive/README.md) | 旧脚本、配置、artifact、报告及迁移映射 |

已有 v3 artifact 可直接验证并用于主训练，不需要每次重新预训练。验证器不会自动替换 artifact；下一版 artifact 应使用新的版本文件和显式配置。

## 历史实验与参考资料

历史输出、checkpoint 和日志保留在原工作区，不覆盖、不迁移、不改写原元数据。归档目录 `../archive/pd-morl-20261010/` 保存整理前源码快照和文件 SHA-256 清单；旧工作区 README 标记为历史用途。详细位置见[归档说明](archive/README.md)。

本地 `../pdmorl_source` 是原版 PD-MORL 参考源码；`../psl_morl_analysis` 是下一阶段的论文/方法资料。已有结果图与分析目录原位保留，不能当作新主线重新运行所得结果。

主线默认保存完整 replay checkpoint，约每 100 万条新训练数据保存一次，保留最近两份；结束时先完成保存，再进行最终评估。主训练的 160 环境、K=320、batch=512、在线 L1 等学习设置保持当前并行基线口径。

当前协议内维护唯一的待核对清单。历史报告中的“下一步”“必须”和旧 PASS 状态不再驱动新实验。
