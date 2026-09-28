# PD-MORL 运行逻辑与 EvoRL 训练耗时问题

## 1. 问题结论

当前 EvoRL 实现没有丢弃训练 transition，也没有把原本的全局 1M 预算错误扩大。原始 PD-MORL 对 Walker 使用 10 个 worker，每个 worker 运行 1,000,000 步，因此总环境 transition 本来就是 10,000,000。

当前训练耗时高由两部分共同造成：

1. 原始算法本身每个主循环执行 10 次顺序 learner update，完整实验需要 9,998,470 次 critic optimizer update；
2. 原始算法具有高频 key evaluation 和周期性 full evaluation，而 EvoRL 当前用 Python 主机循环逐个调用单-lane JAX/Brax evaluator，产生大量小粒度 JIT 调用、设备同步和环境重置开销。

因此，**评估触发规则是 PD-MORL 原始逻辑；极高墙钟时间主要是这套逻辑在当前 JAX/Brax 执行方式下被放大。**

本文依据：

- PD-MORL 源码提交：`35aa1bccb31c149c3f3bb895c7c6ffa674e4b461`；
- PD-MORL Walker 训练入口：`E:/PD-MORL/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/PD-MORL/train_Walker2d_MO_TD3_HER.py`；
- PD-MORL 评估实现：`E:/PD-MORL/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/lib/utilities/MORL_utils.py`；
- EvoRL 对应实现：[mo_td3.py](../evorl/algorithms/mo_td3.py)、[pd_morl.py](../evorl/evaluators/pd_morl.py) 和 [offpolicy_utils.py](../evorl/algorithms/offpolicy_utils.py)；
- 正式配置：[pd_morl_walker_reproduction.yaml](../configs/experiment/pd_morl_walker_reproduction.yaml)。

## 2. 原始 PD-MORL 如何训练

### 2.1 采样结构

Walker 配置为：

```text
time_steps = 1,000,000
process_count = 10
batch_size = 256
policy_freq = 10
eval_freq = 100
eval_episodes = 3
max_episode_len = 500
```

源码创建 10 个 Python multiprocessing 子进程。每个子进程拥有独立的 MuJoCo/Gym 环境和固定 preference 子空间，并持续向容量为 1 的队列提交 transition。

主进程每轮从每个队列各取一条数据，因此一轮得到 10 条真实环境 transition：

```python
for ts in range(0, PROCESSES_COUNT * args.time_steps, PROCESSES_COUNT):
    for main_cnt in range(PROCESSES_COUNT):
        process_samples = train_queue_list[main_cnt].get()
        replay_buffer_main.populate(process_samples.ep_samples)
```

循环迭代次数是 1,000,000，而不是 10,000,000；但每轮收集 10 条 transition，所以最终总量为：

```text
1,000,000 rounds × 10 workers = 10,000,000 transitions
```

### 2.2 Learner 更新

replay 达到 `2 × batch_size × weight_num = 1,536` 条后，每轮执行：

```python
for i in range(PROCESSES_COUNT):
    batch = replay_buffer_main.sample(args.batch_size)
    agent_main.learn(batch, writer)
```

即每轮顺序执行 10 次 learner call。Actor 和 target 按 `policy_freq=10` 延迟更新，所以一轮大致对应：

```text
10 critic updates
1 actor update
1 target update
```

这些更新存在前后参数依赖，不能简单地当成一个大 batch 并行合并，否则训练轨迹会改变。

### 2.3 Key evaluation 与 interpolator 更新

源码维护每个 worker 的 episode count：

```python
if (process_episode_array > eval_cnt_ep).all():
    eval_cnt_ep += 1
    x_tmp = eval_agent_interp(..., eval_episodes=3)
    # 比较 key solutions，并重新拟合 RBFInterpolator
```

其语义是：当所有 10 个 worker 都超过下一个 episode 编号时，执行一次 key evaluation。`eval_cnt_ep` 每次只增加 1，因此如果预填充或快速终止让 episode count 超前，主循环会连续触发，直到计数追平。

Walker 的二维目标对应 3 个 key preferences。每次 key evaluation 串行执行：

```text
3 key preferences × 3 repeats = 9 evaluation episodes
```

结果用于比较并替换 key solutions，然后重新拟合 interpolator。这个控制面会改变后续 loss 使用的 projected preference，因此不能在不改变算法时间语义的前提下简单删除或异步延后。

### 2.4 训练期 full evaluation

源码还包含：

```python
if (process_episode_array > args.eval_freq * eval_cnt).all():
    eval_cnt += 1
    eval_agent(..., w_batch_eval, eval_episodes=3)
```

其中 `eval_freq=100`，`w_batch_eval` 的间隔为 `0.005`，因此每次执行：

```text
201 preferences × 3 repeats = 603 evaluation episodes
```

这些 episode 只用于 HV、sparsity、Pareto front、模型保存和进度记录，不写入 replay，也不计入 10M 训练 transition。

### 2.5 最终与离线评估

训练结束后，源码使用步长 `0.001` 的 1001 preferences 和 3 repeats：

```text
1001 × 3 = 3003 evaluation episodes
```

独立的 `eval_benchmarks_MO_TD3_HER.py` 默认使用 6 repeats：

```text
1001 × 6 = 6006 evaluation episodes
evaluation seeds = [0, 11, 22, 33, 44, 55]
```

### 2.6 原始评估也是串行逻辑

`MORL_utils.py` 中的 `eval_agent()`、`eval_agent_interp()` 和 `eval_agent_test()` 均按以下顺序执行：

```text
for repeat:
    for preference:
        reset environment
        while not terminal:
            deterministic action
            environment step
```

episode 在环境 terminal 时提前结束，不会强制跑满 500 个有效步。

## 3. EvoRL 当前如何运行

### 3.1 单进程、单 GPU、10 个逻辑 worker

正式运行只向 JAX 暴露一张 GPU。10 个 PD-MORL worker 映射为一个 Brax vector environment 的 10 个 lane；GPU 数量不会改变 `K=10`。

预算为：

```text
prefill = 1,530 global transitions = 153/worker
post-prefill rounds = 999,847
each round = 10 transitions

1,530 + 999,847 × 10 = 10,000,000 transitions
```

所以最终每个 worker 为：

```text
153 + 999,847 = 1,000,000 steps
```

### 3.2 Learner 调度

每轮的 10 次 critic update 在一个 JAX `lax.scan` 中顺序执行。Actor update mask 根据全局 learner call 编号和 `policy_freq=10` 选出一次 actor/target update。

完整预算为：

```text
critic optimizer steps = 999,847 × 10 = 9,998,470
actor optimizer steps = 999,847
target updates = 999,847
```

这部分计算量本身就很大：它不是 1M 次网络更新，而是接近 10M 次 critic 更新。

### 3.3 控制面触发

EvoRL 保留了源码条件：

```text
key evaluation:
    all(episode_count > eval_cnt_ep)

full evaluation:
    all(episode_count > 100 × eval_cnt)
```

触发检查位于每个 workflow round 之后。正式配置 `fold_iters=1`，因此 Python 主机循环、JAX 返回和控制面检查将发生约一百万次。

### 3.4 当前 evaluator

控制面环境明确构造成 `parallel=1`。`PDMORLEvaluator.evaluate()` 在 Python 中按 repeat、preference 顺序调用一个 JIT 编译的 `_evaluate_episode()`，并立即通过 `np.asarray(...)` 把结果同步回主机。

每个 evaluation episode 内部使用 `jax.lax.while_loop`：

- 环境 terminal 时提前终止；
- 最长运行 500 步；
- deterministic actor；
- undiscounted vector return。

因此，EvoRL 不会无条件浪费到每个 episode 500 步。真正的问题是每个 preference/repeat 都形成一次独立的 Python → JAX 调用、环境 reset 和设备同步，无法摊薄调度成本。

### 3.5 Replay 与 HER

训练 transition 写入一个全局 replay。HER 达到 `10000 × 10 = 100,000` 条 base entries 后激活，每条 base transition 产生 3 条 relabeled entries。

HER entry：

- 会参与 replay sampling；
- 不代表新的环境交互；
- 不计入 10M environment transitions。

## 4. 正式运行的实测证据

以下数据来自 2026-09-27 21:49 CST 的 seed-1 正式运行快照：

```text
workflow iteration = 12,000
worker steps = 12,153 × 10
valid training transitions = 121,530
key evaluations = 527
full evaluations = 5
loss/gradient/Q/wp diagnostics = finite
HER = active
```

截至该点，仅控制面就发起：

```text
key evaluation episode calls
= 527 × 3 preferences × 3 repeats
= 4,743

full evaluation episode calls
= 5 × 201 preferences × 3 repeats
= 3,015

total serial evaluation episode calls
= 7,758
```

这 7,758 个 episode 调用不计入 121,530 条训练 transition。它们不是“被丢弃的训练步”，而是额外的控制与度量工作。

每个 episode 的真实长度会因 Walker 提前终止而变化，因此不能把 7,758 全部机械乘以 500 当成真实环境步。即便 episode 很短，7,758 次独立 JIT 调用、reset 和同步仍然具有显著墙钟开销。

## 5. 为什么 EvoRL 版本特别慢

### 5.1 第一原因：近 10M 次 critic optimizer update

完整训练要求 9,998,470 次 critic update。400×400 双 critic、batch 256、梯度计算和 Adam update 都在一张 GPU 上按依赖顺序完成。

单纯把“1M workflow rounds”理解成“1M 次训练”会低估约 10 倍。

### 5.2 第二原因：key evaluation 触发极其频繁

早期 Walker 经常在远小于 500 步时跌倒。正式快照中，worker 在约 12,153 步内已经分别完成约 528–559 个 episode，平均 episode 长度只有约 22 步。

由于 key evaluation 的门槛按 episode 编号递增，12,000 轮内已经触发 527 次，而不是每 500 步才触发一次。

这是原始 PD-MORL 的控制规则，但在早期短 episode 阶段非常昂贵。

### 5.3 第三原因：单-lane、逐 episode 的 JAX 调用

原始源码同样串行遍历 evaluation preferences，但它在一个普通 Python/PyTorch/Gym 控制流内执行。EvoRL 当前每个 preference/repeat 都单独：

1. 从 Python 调用 JIT episode；
2. reset 一个 Brax 环境；
3. 执行动态 `while_loop`；
4. 将返回值同步为 NumPy；
5. 再开始下一个 preference。

JAX 更擅长较大的批量计算，而不是成千上万次小粒度、需要主机同步的调用。当前实现没有把 preferences 和 repeats 组织成可摊薄调度成本的批量 evaluator。

### 5.4 第四原因：full evaluation 造成明显停顿

每次 full evaluation 串行执行 603 个 episode call。运行日志中约每 2,000 多个 workflow rounds 出现一次明显的数分钟训练停顿，与生成的 Pareto snapshot 对齐。

这些评估对复现 HV/sparsity progression 有用，但不推进训练计数。

### 5.5 第五原因：每轮都经过主机边界

`fold_iters=1` 意味着约 999,847 次：

```text
JAX training round
→ 返回 Python
→ 检查 episode counters
→ 检查 key/full evaluation trigger
→ 检查 checkpoint/logging
→ 下一轮
```

这是保留精确控制触发时机的直接实现，但降低了 JAX 大块编译执行的优势。

### 5.6 不是主要原因的项目

- GPU 没有空转：抽样时训练进程约占 98% SM；
- 每 1,000 轮记录一次诊断，不是主要瓶颈；
- checkpoint 间隔为 100,000 轮，不解释持续低吞吐；
- HER 没有制造额外环境步；
- 没有 transition 被静默丢弃；
- 其他 GPU 进程可能造成波动，但当前训练自身已接近占满 GPU，是次要因素。

## 6. 原始实现与 EvoRL 的关键差异

| 维度 | 原始 PD-MORL | EvoRL 当前实现 | 影响 |
|---|---|---|---|
| 采样 | 10 个 Gym/MuJoCo 子进程 | 10 个 Brax vector lanes | 语义对应 |
| Learner | 主进程顺序调用 10 次 PyTorch learn | 单 GPU `lax.scan` 顺序更新 10 次 | 更新次数对应 |
| 原始 Walker 配置 | `cuda=False` | JAX GPU | 框架不同 |
| 控制触发 | 主进程逐轮检查 | 每个 workflow round 返回主机检查 | 语义对应，主机边界昂贵 |
| Key evaluation | 3×3，Python/Gym 串行 | 3×3，单-lane JIT 串行 | JAX 调用/同步开销放大 |
| Full evaluation | 201×3，Python/Gym 串行 | 201×3，单-lane JIT 串行 | 每次出现数分钟停顿 |
| 终评 | 1001×3 | 1001×3 | 对应 |
| 独立报告 | 1001×6 | 1001×6 | 对应 |
| 物理环境 | MuJoCo Walker2d-v2 | Brax Walker adapter | episode 分布和耗时不同 |

## 7. 对当前运行的判断

当前运行不是死锁，也没有数值错误。它正在按冻结协议推进：

- worker counters 单调增长；
- key/full evaluation 按源码条件执行；
- HER 已在阈值后激活；
- loss、gradient norm、Q、target 和 interpolator 输出保持 finite；
- GPU 持续计算。

但当前执行路径的墙钟效率不适合快速迭代。按运行早期吞吐外推，完整 seed 需要数十小时，且评估频率会随 episode 长度变化，不能给出稳定的线性 ETA。

## 8. 可选修复方向

### 8.1 保持算法语义的优先方向

1. 在固定 checkpoint 上实现并验证批量 evaluation：同时处理多个 preferences/repeats，但保持相同 reset seeds、deterministic actions、终止语义和输出顺序；
2. 减少 Python → JAX episode 调用次数和逐 episode `np.asarray` 同步；
3. 对 key evaluation、full evaluation 和 learner round 分别做墙钟 profiling，建立可复现的基线；
4. 在证明返回值逐项一致后，才替换正式 evaluator。

批量化存在一个实际权衡：不同 preference 的 episode 长度不同。简单 `vmap` 可能一直执行到批次中最长 episode，需要用 done mask，并比较“减少调度次数”是否足以抵消额外的 masked environment steps。

### 8.2 会改变冻结协议的加速方式

以下方法更快，但不能用于当前 source-faithful 结果，除非明确建立新的实验分类：

- 降低 key evaluation 频率；
- 减少 full evaluation preferences 或 repeats；
- 训练期间关闭 full evaluation；
- 把评估异步化，因为 key result 会改变 interpolator 生效时间；
- 使用多 GPU learner 或 evaluation；
- 合并或跳过顺序 optimizer updates。

## 9. 当前处置边界

正在运行的 seed-1 使用已冻结协议。本文只记录问题和根因，不修改运行中的进程、超参数、评估触发或 learner math。

如果目标是取得当前协议下的一份可审计结果，应让现有运行继续。如果目标是缩短后续 seed 的运行时间，应先在独立分支上完成 evaluator 批量化与逐项等价验证，再决定是否重新分类并启动后续实验。
