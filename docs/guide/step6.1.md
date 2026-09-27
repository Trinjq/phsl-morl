
现在进入 Step 6.1：

PD-MORL Control / Evaluation Plane Integration

本阶段定位：

不是继续增加 learner 算法，
而是补齐完整 PD-MORL 中尚未接入的
control-plane / evaluation-plane 行为。

当前已经冻结：

- MO-TD3 vector Actor/Critic；
- preference lifecycle；
- source-faithful HER；
- preference subspace / logical workers；
- vector Bellman target；
- whole-vector pessimistic critic selection；
- interpolator I(w)；
- directional-angle loss；
- actor / critic loss；
- Step6 PyTorch ↔ JAX Golden Tests；
- GPU matmul_precision = highest reproduction policy。

本阶段禁止修改上述数学行为。

==================================================
一、Step6.1 目标
================

实现并独立验证：

1. initial key-solution artifact loading；
2. online key-preference evaluation；
3. cumulative worker episode tracking；
4. key-update source timing；
5. candidate key-solution aggregation；
6. key-solution replacement；
7. online interpolator refit；
8. MORL vector-return evaluation；
9. training-time evaluation；
10. offline benchmark evaluation；
11. Pareto non-dominated filtering；
12. Hypervolume；
13. Sparsity。

所有这些行为必须位于：

host / control / evaluation boundary

而不是 learner JIT 内。

完成后仍不要做：

- 完整 Step7 end-to-end refactor；
- multi-GPU；
- PSL-MORL；
- Hypernetwork；
- 论文 1M-step reproduction。

==================================================
二、先做 SOURCE AUDIT
=====================

先审查并记录官方源码中的精确行为：

1. PD-MORL/train_Walker2d_MO_TD3_HER.py
2. PD-MORL/train_Walker2d_MO_TD3_HER_Key.py
3. lib/utilities/MORL_utils.py
4. lib/utilities/settings.py
5. PD-MORL/eval_benchmarks_MO_TD3_HER.py
6. 当前 Step5.4 interpolator implementation
7. 当前 Step5.5 alignment integration
8. 当前 EvoRL evaluator / workflow state

必须先回答：

- initial key solutions 从哪里加载；
- artifact shape；
- key preference 顺序；
- initial normalization；
- online normalization；
- eval_cnt_ep 初始值；
- eval_cnt 初始值；
- key update 精确条件；
- full evaluation 精确条件；
- candidate evaluation repeat 数；
- candidate vector 如何聚合；
- evaluation seed 公式；
- preference reset ordering；
- deterministic policy 语义；
- return 是否 discounted；
- training eval grid；
- final/offline eval grid；
- train-time aggregation；
- offline aggregation；
- Pareto duplicate 行为；
- HV sign/reference；
- sparsity N<=1 行为。

先输出：

STEP6.1 SOURCE AUDIT

任何没有源码证据的细节：

不要猜。

==================================================
三、Initial Key Solutions
=========================

Production workflow 不允许：

- zeros；
- random initialization；
- synthetic key solutions；
- 自动临时生成假 artifact。

必须通过显式：

interpolator_artifact

加载初始 key solutions。

来源应为：

PD-MORL key-preference pretraining pipeline
产生的 objective-solution artifact，

或用户明确提供并可追溯的：

interp_objs_walker2d.txt
.npz
等文件。

如果 artifact 不存在：

明确报错并停止。

不要 silent fallback。

---

Artifact provenance
-------------------

至少记录：

- file path；
- SHA256/hash；
- shape；
- dtype；
- key preference ordering；
- raw values。

如果用户当前本地
interp_objs_walker2d.txt
包含某组具体数值：

可以作为该实验 artifact 使用，

但：

不要把它们硬编码为
“通用官方 Walker 常量”。

---

Expected Walker shape
---------------------

L = 2
K_key = 3

key_preferences：

[0.0, 1.0]
[0.5, 0.5]
[1.0, 0.0]

key_solutions：

[3,2]

顺序必须逐行对应。

==================================================
四、Initial Interpolator Fit
============================

加载 raw key solutions 后：

使用 Step5.4 已冻结行为：

row-wise L2 normalization

然后：

SciPy
RBFInterpolator(
    key_preferences,
    normalized_key_solutions,
    kernel="linear"
)

host-side fit。

再提取：

fixed-shape JAX interpolator state。

不要修改：

kernel
normalization
degree/default semantics
wp post-processing。

==================================================
五、Worker Episode Counter
==========================

PD-MORL 官方触发逻辑依赖：

每个逻辑 worker 的累计完成 episode 数。

建立：

episode_count[K]

对于严格 baseline：

K = process_count = 10

第 k 个逻辑 lane：

episode_count[k]

表示：

该 worker 从训练开始到当前为止
累计完成的 episode 数。

注意：

episode termination 后：

该 lane 的 episode_count += 1

但：

worker step counter 不清零；
episode_count 也不清零。

---

重要：
------

episode_count
与：

learner update count
global env timestep
workflow iterations
replay size

完全不是同一个量。

不要互相替代。

==================================================
六、Key Update Trigger
======================

严格保留官方源码逻辑：

if (process_episode_array > eval_cnt_ep).all():
    eval_cnt_ep += 1
    ...

因此：

trigger condition =

all(
    episode_count[k] > eval_cnt_ep
    for all k
)

等价整数形式：

min(episode_count)

> =
> eval_cnt_ep + 1

不要误写成：

min(episode_count) >= eval_cnt_ep

---

Counter initialization
----------------------

eval_cnt_ep 的初始值：

必须从官方 source audit
逐字确认并复现。

不要根据推断硬编码。

---

触发后
------

必须先按源码顺序更新：

eval_cnt_ep += 1

并执行对应的：

key evaluation
replacement
refit

具体先后顺序
严格按 train_Walker2d_MO_TD3_HER.py。

==================================================
七、Key Trigger Boundary Test
=============================

必须增加：

test_key_update_off_by_one

例如在 source audit 确认：

eval_cnt_ep = 0

时，应验证：

episode_count =
[1,1,1,1,1,1,1,1,1,0]

→ NO TRIGGER

episode_count =
[1,1,1,1,1,1,1,1,1,1]

→ TRIGGER

trigger 后：
eval_cnt_ep = 1

episode_count =
[2,2,2,2,2,2,2,2,2,1]

→ NO TRIGGER

episode_count =
[2,2,2,2,2,2,2,2,2,2]

→ TRIGGER

如果官方初始 counter 不是 0：

测试值相应平移。

核心是验证：

strict `>` semantics。

==================================================
八、Online Key-Preference Evaluation
====================================

Key update 使用三个 Walker key preferences：

[0,1]
[0.5,0.5]
[1,0]

对于每个 key preference：

运行 deterministic actor：

a = actor(s, w_key)

禁止：

exploration noise。

每次 episode 累计：

undiscounted vector return

R_s ∈ R^L

---

Candidate solution
------------------

假设官方 eval_agent_interp
使用 S 次 repeats：

必须先从 source audit 确认 S。

candidate_solution：

R_candidate
===========

mean_s(
    R_s
)

即：

对 evaluation repeats 的
vector returns 逐目标取均值。

注意：

candidate 本身有 repeat averaging。

==================================================
九、Evaluation Seed
===================

官方 evaluation seed 公式：

seed =
eval_ep * 11

对于 repeat index：

eval_ep = 0...S-1

训练 evaluation
若 S=3：

[0,11,22]

offline benchmark
若 S=6：

[0,11,22,33,44,55]

Key evaluation 使用的 repeat 数：

必须从 eval_agent_interp
实际调用路径确认后
应用同一 seed 公式。

不要自己换成：

JAX split key sequence

然后声称完全 source-faithful。

JAX/Brax 中可以将这些整数 seed
映射为 PRNGKey：

FRAMEWORK-ADAPTATION

但逻辑 seed 值必须保持。

==================================================
十、Evaluation Reset Ordering
=============================

官方评估语义：

每个 repeat：

- 使用对应 seed 初始化；
- 然后按 preference 顺序
  逐个 episode reset/evaluate。

不同 preference
不保证拥有完全相同初态。

如果 JAX/Brax 为了 batch execution
同时评估多个 preferences：

必须明确分类为：

FRAMEWORK-ADAPTATION

并验证：

聚合结果语义合理。

严格 source-comparison mode
优先保留官方 logical ordering。

不要为了效率
静默改变 seed/reset 语义。

==================================================
十一、Key-Solution Replacement
==============================

对第 k 个 key：

old_solution[k]
candidate_solution[k]
w_key[k]

计算：

old_score =
dot(
    w_key[k],
    old_solution[k]
)

candidate_score =
dot(
    w_key[k],
    candidate_solution[k]
)

只有：

candidate_score > old_score

才替换：

old_solution[k] =
candidate_solution[k]

严格保持：

- strict `>`；
- no tolerance；
- no epsilon margin；
- no Pareto check；
- no averaging with old solution；
- no EMA；
- no weighted interpolation。

---

注意术语
--------

“no averaging”仅表示：

replacement 阶段
不做 old/new 移动平均。

candidate_solution 本身仍然是：

evaluation repeats 的 mean vector。

==================================================
十二、Online Interpolator Refit
===============================

完成全部 key comparisons 后：

无论：

有 replacement

还是：

没有任何 replacement

都必须重新构建 interpolator。

严格执行：

current raw key solutions
    ↓
row-wise L1 normalization
    ↓
SciPy RBFInterpolator(
    key_preferences,
    normalized_solutions,
    kernel="linear"
)
    ↓
extract fixed-shape JAX state

这与 initial fit 的：

L2 normalization

必须继续保持不同。

禁止统一修正。

==================================================
十三、Interpolator State Update
===============================

refit 后新的 JAX interpolator state：

只能更新 values。

要求：

- PyTree schema 不变；
- array shape 不变；
- knot count 不变；
- dtype policy 不变。

这样返回 training JIT 后：

不应因为 structure 变化
触发新的算法路径。

不得将：

SciPy Python object

放进：

AgentState
WorkflowState
training PyTree。

==================================================
十四、JIT Boundary
==================

training JIT 内只允许：

wp = I(w)

pure-JAX fixed-shape forward。

training JIT 内禁止：

- SciPy；
- RBFInterpolator creation；
- key evaluation；
- Python callback；
- host_callback；
- Pareto sorting；
- HV；
- sparsity；
- dynamic knot update。

以下全部放在 host/control/eval boundary：

- key evaluation；
- candidate aggregation；
- key replacement；
- L1 normalization；
- RBF refit；
- Pareto；
- HV；
- sparsity。

==================================================
十五、Full Training Evaluation Trigger
======================================

Full evaluation
与 key update
是两个独立 trigger。

官方训练时序已确认存在：

key-update trigger：
all episode counts
超过下一个整数 threshold。

full training evaluation：
all workers episode counts
超过与 100 * eval_cnt
相关的阈值。

但是：

Step6.1 实现前
必须从官方 train script 精确抄出：

- eval_cnt 初始值；
- 比较符号；
- 是否是 `>`；
- expression 精确形式；
- eval_cnt increment 顺序。

不要只写“每 100 episode”。

必须保存原始 counter semantics。

不得把两个 trigger 合并为：

evaluation_interval。

==================================================
十六、Training Evaluation Grid
==============================

Walker training intermediate evaluation：

preference step = 0.005

因此：

W_eval_train =
[
[0.000,1.000],
[0.005,0.995],
...
[0.995,0.005],
[1.000,0.000]
]

M = 201

training repeats：

3

对应 seeds：

[0,11,22]

actor：

deterministic。

return：

每 episode 未折扣 vector reward sum。

==================================================
十七、Final / Offline Evaluation Grid
=====================================

Walker final training evaluation：

step = 0.001

M = 1001

offline benchmark：

step = 0.001

M = 1001

offline default repeats：

6

seeds：

[0,11,22,33,44,55]

不要把：

201-point training evaluation

和：

1001-point final/offline evaluation

混在一起。

==================================================
十八、MORL Evaluator Output
===========================

建议建立清晰的：

MORLEvaluationResult

至少包括：

preferences [M,L]

returns_per_repeat [S,M,L]

mean_returns [M,L]

hv_per_repeat [S]       # train-time path when applicable
sparsity_per_repeat [S]

mean_hv
mean_sparsity

pareto_indices
pareto_returns

但最终字段
应尽量复用 EvoRL metrics/state 结构。

不要为了评估
修改 training sample format。

==================================================
十九、Training-Time Aggregation
===============================

严格复现官方 train-time evaluation：

对每个 repeat：

1. 完成所有 evaluation preferences；
2. 得到本 repeat 的 objective returns；
3. 构造对应 Pareto/指标；
4. 计算：
   HV_repeat
   sparsity_repeat

最后：

mean(HV_repeat)
mean(sparsity_repeat)

objective returns
也按官方聚合顺序处理。

不要先对全部 repeats 的 returns 求均值
再只算一次 HV，
除非源码就是这样。

==================================================
二十、Offline Benchmark Aggregation
===================================

offline path 与 train-time path 分开。

官方 benchmark：

保留每个 repeat 的：

HV
sparsity
objectives

脚本再报告：

mean / std

然后对：

mean objective returns

做 non-dominated filtering
得到最终 mean Pareto front。

不要把 offline aggregation
复用成 train-time aggregation。

==================================================
二十一、Pareto Non-Dominated Filtering
======================================

官方使用：

NonDominatedSorting().do(
    -returns,
    only_non_dominated_front=True
)

因为：

pymoo 使用 minimization convention，
而 objective returns 是 maximize。

等价数学定义：

point x 被支配
当且仅当存在 y：

y >= x
for every objective

并且至少一维：

y > x

---

Duplicates
----------

禁止提前：

np.unique
deduplicate

如果两个完全相同的点
都是非支配：

官方 pymoo 会保留两者。

EvoRL implementation
也必须保留 duplicate entries。

增加：

duplicate-point Golden Test。

==================================================
二十二、Pareto Tests
====================

固定人工测试：

A=[1,5]
B=[2,4]
C=[3,3]
D=[2,2]

预期：

A,B,C：
non-dominated

D：
dominated

再测试：

E=[2,4]
F=[2,4]

如果该点非支配：

E,F 均保留。

另覆盖：

- one point；
- all equal；
- all dominated except one；
- ties on one objective。

==================================================
二十三、Hypervolume
===================

PD-MORL continuous-control source：

reference point：

zero vector

官方 pymoo 适配：

returns → negative returns

用于 minimization HV。

第一版优先直接：

host-side pymoo

严格复现官方路径。

不要为了纯 JAX
自行重写 HV。

如果实现二目标 NumPy/JAX equivalent：

必须先建立：

pymoo reference Golden Test

比较：

fixed fronts
duplicate fronts
single point
boundary point

逐值一致后
才能替换。

==================================================
二十四、Sparsity
================

严格复现官方：

对于 Pareto front
N 个点、L 个 objectives：

每个 objective
独立排序，

计算：

sum_i
(P_ij - P_(i+1)j)^2

再：

对 objectives 求和

最后：

divide by N-1

不允许：

- sqrt；
- objective normalization；
- range normalization；
- weighting；
- scalarization。

源码边界行为：

N <= 1

返回：

sparsity = 0

即使论文可能表述成 N/A：

Step6.1 以源码 executable behavior 为准。

==================================================
二十五、Key Update 与 Full Evaluation 分离
==========================================

建议实现两个独立 control-plane component：

KeyInterpolatorUpdateController

MORLEvaluator

或等价设计。

KeyInterpolatorUpdateController：

- reads episode_count；
- checks eval_cnt_ep trigger；
- evaluates three key preferences；
- aggregates candidate returns；
- replaces key solutions；
- online L1 refit；
- updates interpolator state。

MORLEvaluator：

- checks full evaluation trigger；
- evaluates 201/1001 grid；
- computes vector returns；
- Pareto；
- HV；
- sparsity；
- logs metrics。

不要共享：

counter
trigger
grid

除非官方源码本来共享。

==================================================
二十六、State Design
====================

Control state 至少需要表示：

episode_count [K]

eval_cnt_ep

eval_cnt

raw key_solutions [K_key,L]

interpolator state

必要 evaluation counters

尽可能放在：

WorkflowState / extra state

但不要污染：

replay transition
Actor observation
Critic input。

所有 JAX-visible state
保持：

fixed shape。

==================================================
二十七、Tests：Key Update
=========================

至少实现：

test_initial_artifact_required

test_initial_artifact_shape

test_initial_artifact_preference_order

test_key_update_off_by_one

test_candidate_repeat_mean

test_candidate_better_replaces

test_candidate_equal_does_not_replace

test_candidate_worse_does_not_replace

test_replacement_is_direct_assignment

test_no_ema

test_online_refit_uses_l1

test_refit_even_without_replacement

test_interpolator_shape_stable_after_refit

test_training_jit_contains_no_scipy

==================================================
二十八、Tests：Evaluation
=========================

至少实现：

test_evaluation_seed_formula

test_training_eval_seed_list

test_offline_eval_seed_list

test_deterministic_actor_eval

test_vector_return_is_undiscounted

test_episode_limit_and_done

test_training_grid_201

test_final_grid_1001

test_training_aggregation_order

test_offline_aggregation_order

==================================================
二十九、Tests：Pareto / HV / Sparsity
=====================================

至少实现：

test_pareto_dominated_point

test_pareto_nondominated_points

test_pareto_duplicates_preserved

test_pareto_single_point

test_hv_against_pymoo

test_hv_zero_reference

test_hv_sign_conversion

test_sparsity_against_source_formula

test_sparsity_single_point_zero

test_sparsity_duplicates

==================================================
三十、Regression
================

Step6.1 完成后：

必须重新运行：

- Step5.1 MORL math；
- Step5.4 interpolator Golden Test；
- Step5.5 alignment loss；
- Step6 PyTorch ↔ JAX Golden Tests；
- Step5.2 HER；
- Step5.3 parallel exploration；
- MO-TD3 regression；
- scalar TD3 regression。

必须保持：

Step6 frozen tolerances

完全不变。

GPU precision policy：

matmul_precision = highest

继续生效。

==================================================
三十一、分类要求
================

文档中每个新模块标记：

SOURCE-FAITHFUL
FRAMEWORK-ADAPTATION
TEST FIXTURE
DEVIATION

典型分类：

SOURCE-FAITHFUL：

- key preference set；
- artifact-derived initial solutions；
- initial L2；
- online L1；
- strict replacement；
- repeat mean candidate；
- episode-count trigger；
- seed formula；
- evaluation grid；
- undiscounted returns；
- pymoo sign/reference；
- sparsity formula。

FRAMEWORK-ADAPTATION：

- CPU process episode counters
  → JAX/Brax logical lane counters；
- Python/SciPy control-plane
  → host boundary + fixed JAX state；
- serial evaluation
  → equivalent Brax batching
  only if logical seed semantics preserved。

TEST FIXTURE：

- synthetic key solutions
  only in isolated unit tests。

DEVIATION：

- zero/random initial key solutions；
- changed trigger frequency；
- fixed-timestep replacement；
- extra tolerance on replacement；
- deduplicating Pareto points；
- changed HV reference；
- unified L2/L1 normalization；
- SciPy inside JIT。

==================================================
三十二、Documentation
=====================

创建：

docs/PD_MORL_CONTROL_EVAL.md

至少记录：

1. Scope
2. Official source audit
3. Initial key-solution artifact
4. Artifact provenance
5. Key preferences
6. Worker episode counters
7. Key update trigger
8. Off-by-one semantics
9. Candidate evaluation
10. Repeat averaging
11. Seed formula
12. Replacement rule
13. Online L1 refit
14. Refit-without-replacement behavior
15. JIT boundary
16. Training evaluation trigger
17. 201-point grid
18. 1001-point grid
19. Training aggregation
20. Offline aggregation
21. Pareto filtering
22. Duplicate behavior
23. Hypervolume
24. Sparsity
25. Unit tests
26. Regression tests
27. SOURCE-FAITHFUL / FRAMEWORK-ADAPTATION / DEVIATION table
28. Remaining Step7 work

==================================================
三十三、PASS Gate
=================

只有以下全部满足才：

STEP6.1 PASS

1. initial key solutions 必须来自明确 artifact；
2. 不存在 zero/random production fallback；
3. artifact provenance 已记录；
4. episode_count[K] 正确累计；
5. key trigger 精确保持源码 strict `>`；
6. off-by-one test PASS；
7. candidate 是 repeat mean vector；
8. replacement 使用 strict scalar score improvement；
9. replacement 是直接覆盖，不做 EMA；
10. online L1 normalization 正确；
11. 无 replacement 时仍 refit；
12. interpolator fixed-shape state 更新正常；
13. SciPy 完全位于 training JIT 外；
14. evaluation seed = eval_ep * 11；
15. deterministic evaluation 正确；
16. undiscounted vector return 正确；
17. training grid = 201；
18. final/offline grid = 1001；
19. training aggregation 与源码一致；
20. offline aggregation 与源码一致；
21. Pareto dominated filtering 正确；
22. duplicate non-dominated points 不被私自删除；
23. HV 与官方 pymoo reference 对齐；
24. HV zero reference 正确；
25. sparsity 公式正确；
26. N<=1 sparsity == 0；
27. Step5.4 regression PASS；
28. Step5.5 regression PASS；
29. Step6 Golden Tests PASS；
30. frozen tolerance 未修改。

==================================================
三十四、停止条件
================

如果发现以下任一情况：

- 官方 source timing 与当前理解不同；
- eval_cnt / eval_cnt_ep 初始化不明确；
- candidate repeat 数无法确认；
- artifact provenance 不明确；
- pymoo 行为无法匹配；
- Step6 regression 失败；

立即停止。

输出：

STEP6.1 SOURCE / INTEGRATION MISMATCH REPORT

不要：

- 猜参数；
- 调大 tolerance；
- 修改 learner 数学；
- 继续 Step7；
- 进入 PSL-MORL。

全部 PASS 后输出：

STEP6.1 CONTROL / EVALUATION RESULT

然后停止。
