# 历史归档

当前运行入口是 [项目 README](../README.md) 与[当前协议](../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)。

## 仓库内归档

- `scripts/`：早期性能扫描、旧多 seed 启动器、阶段 smoke 与旧框架入口。
- `configs/experiment/`：被当前 pd_morl 配置取代的版本及历史实验变体。
- `configs/artifacts/`：原 MuJoCo、旧 Brax artifact 及被撤回候选；保留原始字节，不用于默认训练。
- `docs/`：旧审查、稳定性、性能、L2 对照、Step9 协议与阶段任务。
- `upstream/`：EvoRL 上游 README。
- [moves.json](moves.json)：整理前后相对路径映射，用于定位旧引用。

归档脚本与配置保留当时内容，可能依赖原工作区路径或旧 API，不属于当前受支持入口。需要重现旧实验时使用下面的原工作区或整理前快照，不把旧脚本直接混入当前算法。lab4090 的主 Python 环境已绑定新主线；复现旧实验须在隔离环境安装对应旧版本，或显式设置旧源码根目录的 PYTHONPATH 并核对导入路径，不能仅切换工作目录。

## 整理前工作区快照

两端各自的 `../archive/pd-morl-20261010/` 包含完整源码 ZIP（包括未提交/未跟踪的有效文件）、Git 状态/补丁和 inventory JSON。ZIP 不复制大型原始实验结果；清单逐文件记录原结果位置、大小和 SHA-256，原结果留在原目录。原 Git 仓库和分支也保留，可恢复历史版本。

本地历史工作区：`../evorl`、`../evorl-phsl-convergence`、`../phsl-morl-exp8-k320-v2-verify`。

lab4090 历史工作区：

| 目录 | 历史用途 |
| --- | --- |
| `../evorl` | 最早并行实现、性能对照及历史实验输出 |
| `../evorl-step92-seed1-code` | Step9.2 代码副本 |
| `../phsl-morl-exp8-k320-v2-minimal-verify` | 调度与最小验证实验 |
| `../phsl-scipy-l1`、`../phsl-scipy-l2` | SciPy 插值归一化对照 |
| `../phsl-jax-rbf-l2` | JAX/L2 与 key artifact 更新历史 |
| `../phsl-stability-hv-history` | 稳定性、在线 L1 和 HV 记录 |
| `../phsl-key-replacement-diagnostics` | 锚点替换诊断 |

这些目录原地冻结为历史来源；保留目录是为了维持 Git worktree、已有结果和外部引用，后续开发及新实验只进入 `pd-morl`。旧工作区独有的绘图、性能诊断脚本和规格文件已收入各自源码快照，没有因未合并到主线而丢失。

本地项目根目录的散落计划、补丁和远程代码副本移入上述日期归档的 `loose/`；原始 CSV、图片和结果目录继续原位保留。外部归档清单记录这些位置变化。
