# PD-MORL 插值器：初始 L2、在线 L1

当前基线入口：[协议与运行链路](PD_MORL_REPRODUCTION_PROTOCOL.md)。本说明对应当前 JAX 实现；初始化和在线统一 L2 的旧实验见 [历史报告](../archive/docs/PD_MORL_JAX_L2_2M_REPORT.md)。

## 数学实现

对 key preferences `W` 和原始目标向量 `X`，初始化使用 p=2，在线重新拟合使用 p=1：

```text
Y = X / where(||X||_p > 0, ||X||_p, 1)
Phi[i,j] = -||W[i] - W[j]||_2
A = [[Phi, ones], [ones.T, 0]]
B = concat([Y, zeros])
C = solve(A, B)
I(w) = concat([-||w-W[i]||_2, 1]) @ C
```

RBF 核距离始终是欧氏距离；目标向量归一化的 L1/L2 不改变核距离。偏好向量使用原有单纯形逻辑，插值输出不再额外归一化、裁剪或投影。零目标向量保持为零。

生产拟合与前向计算由 `evorl/utils/pd_morl_interpolator.py` 在 JAX 中完成，固定 degree=0、linear kernel、smoothing=0；SciPy 用于测试、验证及独立微基准。

## 在线更新

`evaluate_keys_device` 返回形状为 `[3,3,2]` 的回报（repeat、key、objective）。先跨 repeats 求均值，再比较 `key @ candidate > key @ old`；只有严格改善才替换。比较使用原始目标回报，不先归一化。

每次触发都按在线 L1 重新拟合，包括没有任何 key 被替换的情况。因此第一次在线触发可能在原始 key 完全不变时改变插值器，这是初始 L2、在线 L1 协议的结果。

`raw_key_solutions` 保留原始目标值；`normalized_key_solutions` 用于拟合。当前 artifact 及 SHA 统一见主协议。

## 验证与历史记录

`tests/test_pd_morl_interpolator.py` 对 initial/L2 和 online/L1 分别进行 SciPy/JAX、JIT/vmap 和 float32/float64 对照，容差为 float32 `2e-5`、float64 `1e-10`；控制评估测试还检查原始回报替换规则。

`benchmark_pd_morl_jax_rbf.py` 测量当前 online/L1 路径，并另外给出 SciPy-L1/L2 参考。已存在的 `benchmark_pd_morl_jax_rbf_l2.json` 属于旧 L2 实验，保留其原始数值，不将旧耗时重新标注为当前 L1 测量结果。
