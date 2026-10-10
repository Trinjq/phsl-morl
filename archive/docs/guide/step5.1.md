> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。


现在进入 Step 5.1。

当前已经有最小 preference-conditioned MO-TD3。

这一阶段不要增加任何新的训练机制，
只整理 MORL 数学原语，建立独立、纯 JAX 的数学工具层。

==================================================
一、先审查现有实现
==================

先检查当前代码中是否已经存在以下逻辑：

- scalarize
- vector norm
- cosine similarity
- angle

特别检查：

evorl/algorithms/mo_td3.py

如果已有 scalarize：

不要保留两份实现。

将它移动/统一到新的 MORL math module，
然后由 MO-TD3 import 使用。

要求原 Step4.4 行为保持不变。

==================================================
二、新建纯 JAX MORL math module
===============================

建立一个独立模块，例如：

evorl/utils/morl_math.py

或与现有项目结构最一致的位置。

至少实现：

1. scalarize(q_vec, w)

定义：

sum(q_vec * w, axis=-1)

要求支持：

q_vec [L], w [L] -> scalar
q_vec [B,L], w [B,L] -> [B]

如果 twin critic 调用需要 broadcast，
保持 objective axis 为最后一维。

---

2. safe_norm(x, axis=-1, keepdims=False, eps=1e-8)

定义为：

norm = sqrt(sum(x^2, axis=axis))
safe = maximum(norm, eps)

不要实现为：

sqrt(sum(x^2) + eps)

默认使用 L2 norm。

---

3. normalize_vector(x, axis=-1, eps=1e-8)

定义：

x / safe_norm(x, keepdims=True)

零向量必须保持为零向量，
不能产生 NaN/Inf。

---

4. cosine_similarity(x, y, axis=-1, eps=1e-8)

定义：

dot(x,y)
/
(
    max(||x||, eps)
    *
    max(||y||, eps)
)

目标是与 PyTorch F.cosine_similarity 的数值保护语义对齐。

要求：

- objective/vector axis 默认最后一维；
- 支持 batch；
- 支持 vmap；
- 不做额外 normalization state；
- 不修改输入向量；
- zero norm 时返回 finite value。

---

5. directional_angle(x, y, axis=-1, eps=1e-8)

定义：

cos = cosine_similarity(x,y)
cos = clip(cos, -1.0, 1.0)
angle = arccos(cos)

返回单位：

radian

不要转 degree。

注意：
本阶段只实现 generic geometry primitive。

后续 PD-MORL 会调用：

directional_angle(w_p, Q_vec)

但本阶段：

- 不实现 interpolator；
- 不实现 w_p；
- 不接 actor loss；
- 不接 critic loss。

==================================================
三、严格依赖要求
================

所有 runtime 数学只能使用：

jax
jax.numpy

禁止依赖：

- PyTorch
- NumPy runtime
- SciPy

测试中可以使用 Python 常量进行 expected value，
但核心实现不得依赖 NumPy/SciPy。

所有函数要求：

- jax.jit 可运行；
- jax.vmap 可运行；
- float32 正常；
- 不产生不必要的 host/device transfer。

==================================================
四、shape 规则
==============

所有向量运算默认：

axis = -1

至少支持：

[L]
[B,L]
[B,C,L]

其中：

L = objective dimension
C 可以表示 critic axis。

禁止把 batch axis 和 objective axis 混淆。

==================================================
五、数值边界定义
================

明确测试：

1. zero vector：

x = [0,0]

safe_norm(x) == eps

normalize_vector(x) == [0,0]

cosine_similarity(x,y) == 0

directional_angle(x,y) == pi/2

结果必须 finite。

2. same direction：

[1,0], [2,0]

cosine ≈ 1
angle ≈ 0

3. orthogonal：

[1,0], [0,1]

cosine ≈ 0
angle ≈ pi/2

4. opposite：

[1,0], [-1,0]

cosine ≈ -1
angle ≈ pi

5. near-boundary floating point：

确保 acos 输入经过 clip 后不产生 NaN。

==================================================
六、gradient 检查
=================

虽然本阶段不接 loss，
这些函数未来会进入 actor/critic gradient path。

因此增加最小 gradient smoke test：

对非退化、非完全共线向量执行：

jax.grad(directional_angle)

确认 gradient finite。

不要要求在 cosine = ±1 的精确边界处 gradient finite，
因为 acos 在数学上该处导数奇异。

不要为了通过这个测试擅自将 clip 改成：

[-1+eps, 1-eps]

如果后续训练阶段需要额外 gradient stabilization，
单独记录并在后续阶段处理。

==================================================
七、scalarize 重构要求
======================

如果 Step4.4 的 mo_td3.py 已有 scalarize：

将其统一到 MORL math module。

然后：

mo_td3.py import scalarize

必须保证：

- MO-TD3 数值结果不变；
- Step4.4 tests 全部继续通过；
- 不复制两份 scalarize。

==================================================
八、测试
========

创建：

tests/test_morl_math.py

至少包含：

- 手算 scalarize；
- batch scalarize；
- safe_norm；
- normalize；
- cosine same direction；
- cosine orthogonal；
- cosine opposite；
- zero norm；
- directional angle 0 / pi/2 / pi；
- batch input；
- [B,C,L] input；
- float32 dtype；
- jit；
- vmap；
- finite gradient smoke test。

同时重新运行：

tests/test_mo_td3.py

确认 Step4.4 未回归。

==================================================
九、不要实现
============

本阶段禁止加入：

- interpolator；
- projected preference w_p；
- HER；
- directional angle loss；
- actor angle term；
- critic angle term；
- evaluation；
- Pareto/HV/sparsity；
- PSL-MORL；
- hypernetwork。

这一阶段只建立数学原语。

==================================================
十、输出
========

创建：

docs/MORL_MATH.md

记录：

- 函数定义；
- shape 规则；
- eps 语义；
- zero-vector 行为；
- angle 单位；
- jit/vmap 结果；
- gradient 边界说明；
- scalarize 是否从 MO-TD3 重构；
- 测试结果。

完成后停止。
