# PD-MORL parallel preference exploration

## Already implemented in Step 4.3

Step 4.3 already provides the official preference grid, dynamic K-way
subdivision, logical-worker/subspace binding, and episode-fixed preference
lifecycle with per-done-lane resampling. Step 5.3 does not reimplement or
change those rules.

## PyTorch execution model

The official Walker script creates `K=process_count` child processes. Each
owns one environment and independent state/RNG streams, but receives the same
actor and critic objects after `share_memory()`. Each child queue submits one
single-transition `Experience` plus its worker ID and counters.

One main-loop round reads exactly one item from each of the K queues, inserts
the K base transitions sequentially into one main replay pool, and lets the
replay perform add-time HER. Once the source's learner-start check passes, the
main process calls `agent_main.learn()` exactly K times. Each call performs one
critic update; the persistent `total_it` triggers actor and target updates
when `total_it % policy_freq == 0`.

## JAX/EvoRL replacement

| PyTorch source | JAX/EvoRL |
| --- | --- |
| K child processes | K logical workers / Brax lanes |
| one env per process | one batched Brax environment with `B=K` |
| shared-memory actor/critic | one shared `AgentState` applied to all lanes |
| K queue messages | one `[K,...]` transition batch |
| main-process aggregation | flattening of the batched transition |
| one main replay | the existing workflow replay state |

The strict baseline sets `num_envs == process_count` and `rollout_length=1`.
The existing `B > K` modulo mapping remains available only as a
**FRAMEWORK-ADAPTATION / PERFORMANCE MODE**; it is not the baseline here.

`process_count` and physical `device_count` are independent. This step uses a
single GPU for all K logical workers and adds no multi-GPU placement or replay
sharding architecture.

## Collection/update ratio

For every source-faithful round:

```text
K base environment transitions -> K critic learner updates
```

The MO workflow's update scan has length K. Its actor mask uses global learner
IDs `round_index*K + [1..K]`, preserving the source's persistent
`total_it % policy_freq` behavior across collection rounds. With Walker
defaults, `K=10` and `policy_freq=10`, each round performs 10 critic updates
and one delayed actor/target update. Here `round_index` is
`state.metrics.iterations`, which advances only after a real learner round.
The 153 prefill collection rounds leave it at zero, so the first K learner
calls map to `total_it=1..K`; collection before learner start cannot trigger a
delayed update.

## Three independent warm-up thresholds

| Purpose | Walker threshold | JAX state/condition |
| --- | ---: | --- |
| Learner start | replay `>= 1536` | `learner_start_replay_entries=1536` |
| Random behavior policy | per-worker steps `< 10000` | `worker_steps[K] < start_timesteps` |
| HER activation | replay `> 10000 * process_count` | `her_start_timesteps=10000` |

The official source checks the main replay entry count after inserting one
transition from every worker. It skips learning while
`len(replay_buffer_main.buffer) < 2 * batch_size * weight_num`; Walker's
threshold is therefore `2 * 256 * 3 = 1536` entries. The MO-only config stores
this as `learner_start_replay_entries=1536`. EvoRL's existing
`learning_start_timesteps` field controls setup prefill rather than a main-loop
condition, so the MO prefill stops at 1530. The first `B=10` collection round
then crosses from 1530 to 1540 entries and immediately runs the learner,
matching the source's boundary check. Scalar TD3 retains its original startup
configuration.

The MO agent stores one cumulative environment-step counter per logical
worker. Setup initializes all counters to 153 after the 1530-entry prefill.
Each subsequent `rollout_length=1` round increments every counter once;
episode termination and reset do not reset them. A JAX batched mask selects a
uniform action within the Brax action bounds while a worker counter is below
10000. At and after 10000, that lane uses the shared actor plus the official
exploration noise. The transition keeps the current episode preference in
both branches. Learner updates therefore begin at replay size 1540 while all
workers continue random-action collection until their own step 10000.

## HER integration

The `[K,...]` base batch passes once through the Step 5.2 add-time HER path and
then enters the same replay state. Before HER activation it adds K originals;
when fully active it adds `K * (1 + N_w)` entries. Walker defaults therefore
add at most `10 * 4 = 40` entries per collection round. Relabels still change
only preference. HER keeps the separate official warm-up base
`her_start_timesteps=10000`, so its activation threshold remains
`10000 * process_count`; lowering learner start does not activate HER early.

## Classification

**SOURCE-FAITHFUL:** K logical workers, one shared policy, one logical replay,
K-transition collection rounds, K critic updates, and persistent delayed
actor/target scheduling.

**FRAMEWORK-ADAPTATION:** Python processes become Brax batch lanes, queues
become an array batch, and multiprocessing RNG streams become JAX PRNG.
The source workers can overlap their next action with main-process learning;
the JAX replacement uses deterministic synchronous collect-then-update
rounds while preserving the logical K-transition/K-update schedule.

No Step 5.3 algorithm-semantic deviation is known after separating learner
start, per-worker random behavior, and HER activation.

## Verification

- Default single-GPU Walker learner boundary: setup ended at replay/sample count
  1530 with `can_sample=False`; the first collection round ended at 1540 with
  `can_sample=True` and completed finite critic updates while worker counters
  advanced from 153 to 154 and remained in random-action mode.
- Mixed worker counters around 10000 select random actions only for lanes below
  the threshold; the JIT test also verifies action bounds and preference
  preservation.
- K=4/8/10 orchestration, shared policy, global replay, HER, warm-up crossing,
  learner boundary, per-worker random-action masks, delayed-update masks, and
  JIT: `10 passed` on GPU.
- Step 4.3-Step 5.3 targeted regression: `36 passed, 1 deselected` on GPU;
  the deselected standalone Walker test is covered by the real workflow run
  above.
- Real single-GPU Walker workflow with `B=K=10` ran three consecutive rounds:
  replay size `40 -> 80 -> 120`, sampled timesteps `10 -> 20 -> 30`, with
  finite losses throughout. Actor, critic, and target parameters changed.
  This was an intentionally accelerated smoke configuration, not the Walker
  default: `random_timesteps=0`, `learning_start_timesteps=0`,
  `process_count=10`, `num_relabel_preferences=3`, `rollout_length=1`,
  `batch_size=8`, and initial replay size 0. At the time of that run HER still
  reused `learning_start_timesteps`, so setting it to 0 also preactivated HER;
  the equivalent current override is `her_start_timesteps=0`. The first round
  therefore produced `10 * (1 + 3) = 40` entries.
- A K=4 Walker run kept the actor unchanged after rounds 1 and 2 (8 critic
  updates), then updated it during round 3 at global critic update 10.
- The unchanged scalar TD3 scheduling branch completed a separate Walker GPU
  workflow step with finite actor and critic losses.
