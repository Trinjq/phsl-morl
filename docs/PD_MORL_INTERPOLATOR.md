# PD-MORL 插值器：纯 JAX 与统一 L2

## 目标

本改动解决两个相互独立的问题：将 PD-MORL 的线性 RBF 初始化、在线 refit
和前向计算放到 JAX 设备侧，并把 Key objective returns 的初始与在线目标统一
为逐行 L2 归一化。偏好向量仍沿用原有 L1 单纯形逻辑；插值输出 `I(w)` 不再
追加任何归一化、裁剪或投影。

## 数学实现

对 Key preferences `W` 和 raw Key solutions `X`：

```text
Y = X / where(||X||_2 > 0, ||X||_2, 1)
Phi[i,j] = -||W[i] - W[j]||_2
A = [[Phi, ones], [ones.T, 0]]
B = concat([Y, zeros])
C = solve(A, B)
I(w) = concat([-||w-W[i]||_2, 1]) @ C
```

`PDMORLInterpolatorState` 保留原有字段和固定 shape：
`key_preferences`、`normalized_key_solutions`、`coefficients`、`shift`、
`scale`、`powers`、`epsilon`。degree=0 对应零次多项式，shift/scale/powers/
epsilon 分别为零、单位、零次和 1。

生产路径 `fit_interpolator_state`、`normalize_key_solutions` 和 `interpolate`
只使用 JAX；没有 SciPy、NumPy 往返、callback 或 CPU fallback。SciPy 只保留在
`tests/test_pd_morl_interpolator.py` 和独立 artifact 验证脚本中，作为数值参考。

## 在线控制

Key evaluation 返回固定形状 `[3, 3, 2]` 的 JAX array。`update_key_solutions_jax`
在 JIT 中先对三次 repeat 求均值，再按 raw returns 执行严格
`key @ candidate > key @ old`；相等不替换。无论是否替换，都会调用 JAX L2
refit。`raw_key_solutions` 保持未归一化，供下一次替换比较；Actor/Critic、HER、
reward、训练预算和 Key 触发时序不变。

## 验证

SciPy-L2 与 JAX-L2 在 3 个锚点、内部网格、JIT/vmap、float32/float64 和零向量
保护上对照；验收容差为 float32 `2e-5`、float64 `1e-10`。实验室窄回归覆盖
插值器、控制评估和 GPU-native evaluator；benchmark 使用 Artifact
`configs/artifacts/interp_objs_walker2d_brax.txt`，SHA-256 为
`989e74d631ba74572884c4f32b19e12c9317af847ffbc6c02bdde5ad39bfa209`。

旧行为的“初始化 L2、在线 L1”只作为 SciPy-L1 对照，不再是生产在线路径。
