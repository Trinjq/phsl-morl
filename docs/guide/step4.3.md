
现在开始 Step4.3：Official Preference Sampling。

Step4.2 已经完成 preference w 的生命周期和数据传递：

preference
→ rollout
→ SampleBatch
→ replay

本阶段只修改 preference 的采样方式。

不要修改其他已经通过测试的数据链。

==================================================
一、目标
========

将 Step4.2 中临时的：

u ~ Uniform(0,1)
w = [u, 1-u]

替换为 PD-MORL 官方 Walker2d 使用的离散 preference sampling。

当前：

num_objectives = 2
C_p = 10

==================================================
二、严格复现官方 preference grid
================================

根据 PD-MORL 官方 Walker2d 训练脚本生成：

step = 0.001

的二目标离散 preference grid。

共 1001 个 preference。

每个 preference 满足：

w[0] >= 0
w[1] >= 0
w[0] + w[1] == 1

必须严格按照官方源码的生成顺序构造。

不要自行改变排序。

==================================================
三、划分 10 个 subspaces
========================

严格复现：

np.array_split(w_batch, 10)

的结果。

应得到 10 个逻辑 preference subsets。

对于 1001 个 preference：

subspace 0:
101 个

subspace 1~9:
各 100 个

必须验证 JAX 实现与 NumPy 官方行为一致。

==================================================
四、采样规则
============

固定：

logical_worker_id = 0 ... 9

worker i：

只能从 subspace i 中随机选择一个 preference。

episode 开始时：

从对应 subspace 随机采样一个 w。

episode 内：

继续沿用 Step4.2 已经实现并验证的生命周期逻辑，
w 保持不变。

combined done 后：

仍然只更新对应 lane 的 w。

不要修改 Step4.2 的 episode lifecycle。

==================================================
五、最小修改原则
================

只修改 preference sampler 及其必要的 worker/subspace 索引。

不要修改：

- vector reward；
- SampleBatch；
- replay buffer；
- rollout preference 保存位置；
- observation；
- Actor；
- Critic；
- TD target；
- HER；
- interpolator；
- angle；
- evaluation；
- 多 GPU sharding。

本阶段不处理 1 GPU / 3 GPU placement。

只定义 10 个逻辑 worker。

单 GPU 测试即可。

==================================================
六、删除临时 sampler 的正式使用
===============================

Step4.2 的连续 uniform-simplex sampler：

u ~ Uniform(0,1)

不能再作为正式 Walker2d training sampler。

如果保留该函数用于测试，可以保留代码，
但必须明确标记：

TEMPORARY / TEST ONLY

正式 preference wrapper 必须使用官方离散 sampler。

==================================================
七、测试
========

Test 1：grid

验证：

w_batch.shape == (1001,2)

所有元素：

w >= 0

每行：

sum(w) == 1

并与官方 NumPy grid 数值和顺序一致。

Test 2：subspace

验证：

共 10 个 subspaces。

size：

[101,100,100,100,100,100,100,100,100,100]

并与：

np.array_split(w_batch,10)

结果逐项一致。

Test 3：worker sampling

worker i 采出的所有 w：

必须属于 subspace i。

禁止 worker 从其他 subspace 采样。

Test 4：episode lifecycle 回归

确认 Step4.2 行为没有改变：

- episode 内 w 不变；
- done 后只更新对应 lane；
- transition 保存旧 episode 的 w。

Test 5：replay 回归

确认：

preference 仍能完整经过
rollout → flatten → replay add/sample。

Test 6：JIT/GPU

正式 sampler 和 resample 路径必须：

- pure JAX；
- 可 jax.jit；
- 无 NumPy runtime RNG；
- 无 host callback；
- GPU backend 正常。

==================================================
八、输出文档
============

创建：

docs/MORL_OFFICIAL_PREFERENCE_SAMPLING.md

记录：

1. 官方 grid 生成方式；
2. 1001 个 preference；
3. 10 个 subspace；
4. 每个 subspace size；
5. worker → subspace 映射；
6. 修改文件；
7. 测试结果；
8. 临时 uniform sampler 是否仍保留以及用途。

==================================================
九、完成条件
============

满足以下条件才算：

STEP4.3 PASS

- 正式 sampler 不再使用连续 uniform simplex；
- 1001-point grid 与官方一致；
- np.array_split(...,10) 结果一致；
- worker 0..9 分别绑定 subspace 0..9；
- episode lifecycle 没有被破坏；
- preference replay 数据链没有被破坏；
- JIT 正常；
- GPU 正常；
- 没有修改 Actor/Critic；
- 没有实现 HER；
- 没有实现多 GPU placement。

完成后停止。

不要继续下一阶段。
