# 当前项目约定

- 每次回复先称呼用户 boss。
- 唯一开发主线：本地 E:/projects/pd-morl，lab4090 /home/qiuquanj/projects/pd-morl；分支 codex/pd-morl-mainline。
- 当前阶段是 PD-MORL GPU 并行基线；PSL-MORL 和后续新方法属于下一阶段。
- 默认训练配置为 configs/experiment/pd_morl.yaml。参数、评估口径与待核对项只在 docs/PD_MORL_REPRODUCTION_PROTOCOL.md 维护。
- archive/ 与其他旧工作区属于历史资料，旧阶段指令不作为当前任务要求。更改前检查现有未提交修改。
- 原始实验输出、checkpoint、日志与旧元数据原位保留；新实验使用新目录，分析结果与输入分开。
- 训练与运行验证在 lab4090 的 evorl 环境进行；不修改其他用户的进程。
- 保留向量奖励、偏好、HER、更新顺序和显式 PRNG 状态；整合代码不自动授权更改实验算法。
- 文档与文件重排检查引用、Hydra 配置组合及相关回归；无需重复创建状态表或审查层。
- PD-MORL 已冻结为 Git 标签 `pd-morl-baseline-10m-v1`；算法、配置及 artifact 指纹以 docs/PD_MORL_REPRODUCTION_PROTOCOL.md 的冻结条目为准。后续不覆盖或移动该标签；修订基线须经用户明确授权并建立新版本。
- 当前 PD-MORL 目录的 evorl/、scripts/、configs/ 保持冻结。PSL-MORL 在独立分支及 worktree 开发，共享框架的修改也放在该工作目录，不改动正在扫描的 PD-MORL 副本。
- lab4090 当前 evorl 环境及其可编辑安装映射用于 PD-MORL；PSL-MORL 如需安装或调整依赖，使用独立环境，不修改基线环境或将其 import evorl 重定向到其他目录。
