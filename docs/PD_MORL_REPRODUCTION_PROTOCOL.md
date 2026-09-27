# PD-MORL Step9.0 Reproduction Protocol

Status: **STEP9.0 PD-MORL REPRODUCTION PROTOCOL PASS**

This is an accounting and protocol audit only. No 1M-step run or six-seed
experiment was started. The next permitted experiment is one source-equivalent
Walker seed, after an explicit user request.

Step9.1 corrects the earlier Step9.0 draft, which added the 1,530-transition
prefill on top of the official budget. The official `N=1,000,000` already
includes those first 153 steps in every worker.

## 1. Official source budget semantics

The executable baseline is
`PD-MORL/train_Walker2d_MO_TD3_HER.py`, with Walker settings from
`PD-MORL/lib/utilities/settings.py`. The source starts `Cp=10` child workers.
Each child calls `enumerate(exp_source)` and has no local `N` stop. The parent
loop is `range(0, Cp * N, Cp)`, receives one queue item from every child per
round, and performs one learner call per received item. Therefore the source
meaning is **N rounds per worker**, not one global pool of N transitions.

| Counter | Initial value | Increment location | Increment | Stop / meaning |
| --- | ---: | --- | ---: | --- |
| child `time_step` | 0 | `enumerate(exp_source)` | 1 per child transition | queue metadata; not the parent stop counter |
| parent `ts` | 0 | `for ts in range(0, Cp*N, Cp)` | `Cp` per round | loop end at `Cp*N`; source progress label |
| `process_step_array` | 0 per worker | one queue item per worker | `time_step + 1` | episode/key/full evaluation bookkeeping |
| learner calls | 0 | parent inner `for i in range(Cp)` | `Cp` per round | one update for each collected transition |
| source budget | — | parent range | `N` rounds | `N=1,000,000` per worker |

Thus Walker has `N*Cp = 10,000,000` valid environment transitions. HER
relabels are replay entries and are not environment transitions.

## 2. Official paper budget

Appendix B.3 Table 5 reports Walker main training with `N=1,000,000`, batch
256, `gamma=.995`, `tau=.005`, replay capacity `2,000,000`, `Cp=10`, three
HER preferences, learning rates `3e-4`, and one hidden layer of width 400 in
the table. The executable source instead defines actor `19 -> 400 -> 400 -> 6`
and twin critics `25 -> 400 -> 400 -> 2`; executable source takes precedence.
Other settings are policy delay 10, exploration noise .1, target noise .2,
target-noise clip .5, and actor-loss coefficient 10. The paper's six-run
average is a repetition protocol, not six source workers in one run.

## 3. EvoRL counter semantics

The formal config fixes `rollout_length=1`, `num_envs=process_count=10`, and
`num_updates_per_iter=1`.

| Counter | Shape / initial value | Per workflow round | HER included? | Stop / trigger use |
| --- | --- | ---: | --- | --- |
| `config.total_timesteps` | scalar, 10,000,000 in formal config | — | no | `learn()` stop budget; includes prefill exactly once |
| `metrics.sampled_timesteps` | scalar, 0 then 1,530 prefill | +10 | no | `learn()` remaining-budget calculation and logging |
| `metrics.iterations` | scalar, 0 | +1 | no | current workflow round and checkpoint step |
| `extra_state.worker_steps` | `uint32[10]`, 0 then 153 after prefill | +1 per lane | no | random-action warm-up and source progress logging |
| `extra_state.total_it` | scalar `uint32(0)` | +10 | no | source-equivalent learner-slot accounting only |
| `extra_state.episode_count` | `uint32[10]`, 0 plus completed episodes | completed episodes per lane | no | key/full evaluation triggers |
| critic optimizer steps | derived scalar, starts after prefill | +10 | no | one update per post-prefill collected worker transition |
| actor optimizer steps | derived scalar | +1 for K=10, delay=10 | no | delayed policy update |
| target updates | derived scalar | +1 for K=10, delay=10 | no | paired with actor update |
| replay base inserts | replay state | +10 | no | physical source transitions stored |
| replay HER inserts | replay state | activated after threshold | yes, only here | replay expansion, never environment count |
| replay size | scalar buffer size | base + active HER, capped at 2M | yes | learner sampling availability |

The current `total_it` field is an accounting field; it does not drive a
schedule. It is incremented in `MOTD3Workflow.step()` by `process_count`.

## 4. Source-to-EvoRL mapping

Let `K=10`, `P=153` prefill steps per worker, and `R=N-P=999,847` be the
number of post-prefill EvoRL workflow rounds:

```text
final_worker_steps        = P + R = N
valid_env_transitions     = (P + R) * K = N * K
critic_optimizer_steps    = R * K
actor_optimizer_steps     = floor(R*K / policy_delay)
target_update_steps       = actor_optimizer_steps
```

The corrected formal budget therefore has

```text
valid_env_transitions  = 10,000,000
critic updates         = 9,998,470
actor updates          = 999,847
target updates         = 999,847
```

EvoRL prefill is `random_timesteps=1,530`, exactly 153 transitions per lane
and 1,530 valid transitions globally. It is part of the official 10,000,000
transition budget. Consequently `total_timesteps=10,000,000`, and the main
workflow contributes the remaining `999,847 * 10 = 9,998,470` transitions.

## 5. Meaning of the Step8 smoke numbers

The Step8 reports (`critic=133,120`, `actor=13,312`, `target=13,312`) count
optimizer applications, not minibatch rows or loss terms. They correspond to
13,312 workflow rounds with K=10 source learner slots per round:

```text
critic = 13,312 * 10 = 133,120
actor  = floor(133,120 / 10) = 13,312
```

`total_it` has the same cumulative increment as the critic count in this
configuration. The reported replay size (`238,640`) is physical replay
entries (base transitions plus active HER relabels), not environment steps.

## 6. HER accounting

Every collected transition gets one base replay entry. After
`buffer_size + 1 > her_start_timesteps * K`, the current JAX implementation
adds three relabeled entries for that transition. With capacity 2M, replay
size is capped, while environment and learner counters remain uncapped. This
matches the source threshold and keeps HER out of `sampled_timesteps`.

## 7. Learner update ratio

After the 1,530-entry prefill, the first workflow round inserts ten entries,
crosses the 1,536-entry learner threshold, and makes K learner calls. The
current parallel TD3 path then scans exactly K critic updates in every
post-prefill workflow round and applies the delayed actor/target update at
source-equivalent slot numbers. Therefore, over the optimizer-active portion,

```text
environment transitions : critic optimizer steps = 1 : 1
```

The formal config fixes `num_updates_per_iter=1`; no schedule change was made.

## 8. Warm-up accounting

The source uses per-worker `time_step < start_timesteps` for random actions;
Walker therefore has a 10,000-step random-action phase in each logical lane.
Learner startup is separately gated by replay size
(`2*batch_size*weight_num = 1,536`) after the framework prefill. HER
activation uses the global replay-size threshold
`start_timesteps * Cp = 100,000`. EvoRL maps these to `worker_steps`,
`start_timesteps=10,000`, `random_timesteps/learning_start_timesteps=1,530`,
`learner_start_replay_entries=1,536`, and `her_start_timesteps=10,000` with
`process_count=10`. The 1,530 value is the framework's initial replay-fill
budget, not a replacement for the source random-action threshold. No algorithm
schedule was changed by this audit.

## 9. Evaluation timing

Key replacement and full training evaluation remain episode-count driven:
key evaluation is triggered when every logical lane advances past the current
key episode counter; full evaluation is triggered at the corresponding 100
episode cadence. They are not converted to fixed timestep intervals. The
exact number during a 1M-worker-step run is data-dependent because episode
completion counts depend on the environment trajectory.

Training evaluation uses 201 preferences with three repeats. The training
final hook uses 1001 preferences with three repeats. The separate explicit
`evaluate_offline()` paper-report path uses 1001 preferences with six repeats.
For every training, final, or offline repeat, sparsity first selects that
repeat's maximization Pareto front and then applies the adjacent-gap formula;
only the resulting per-repeat sparsities are averaged. The previous evaluator
incorrectly applied the formula to all returns.

## 10. Key pretraining accounting

The loaded `configs/artifacts/interp_objs_walker2d.txt` is the traced key
artifact used to initialize the main run. The paper's Walker key-solution
pretraining budget (`N=2,000,000`, separate settings in Table 6) is not part
of the Table 5 `N=1,000,000` main MO-TD3-HER budget. They are separate costs;
this formal config does not silently charge pretraining to main training.

## 11. Hyperparameter parity table

| Item | Official Walker | Current formal config | Classification |
| --- | --- | --- | --- |
| main N | 1,000,000 per worker | `source_worker_steps=1,000,000` | MATCH |
| batch | 256 | 256 | MATCH |
| gamma / tau | .995 / .005 | .995 / .005 | MATCH |
| replay capacity | 2,000,000 | 2,000,000 | MATCH |
| workers K | 10 | 10 | MATCH |
| HER relabels | 3 | 3 | MATCH |
| actor / critic LR | 3e-4 / 3e-4 | 3e-4 / 3e-4 | MATCH |
| hidden architecture | executable actor/critics have two 400-wide layers | `[400,400]` | MATCH; paper/source discrepancy recorded |
| policy delay | 10 | 10 | MATCH |
| exploration / target noise | .1 / .2, clip .5 | .1 / .2, clip .5 | MATCH |
| actor-loss coefficient | 10 | 10 | MATCH |
| gradient clipping | PyTorch total L2 norm, max 100, actor and critic | Optax global norm, max 100, same update order | MATCH; framework implementation mapping |
| episode limit | 500 | 500 in formal config | MATCH at limit |
| reward / done / reset | MuJoCo Walker path | Brax adapter | ENVIRONMENT DEVIATION (below) |
| keys / interpolator | source key workflow | traced key artifact + interpolator state | FRAMEWORK-ADAPTATION |
| evaluation grids | training 201/3, final 1001/3, offline 1001/6 | same | MATCH |

The paper's Table 5 wording and executable network differ. Under the project
rule that executable source wins, `[400,400]` is source-faithful. Mapping
PyTorch `clip_grad_norm_` to Optax global-norm clipping changes framework
syntax, not the algorithm parameter or clipping mathematics.

## 12. Environment comparability

| Property | Original source | EvoRL formal run |
| --- | --- | --- |
| observation | 17 (`qpos[1:]` + clipped `qvel`) | 17-dimensional Brax Walker adapter |
| action | 6 | 6 |
| objective 1 | forward velocity + 1 | `x_velocity + 1` |
| objective 2 | `5 - sum(action**2)` | `5 - sum(action**2)` |
| termination | Walker height/angle bounds | Brax termination plus truncation/autoreset wrapper |
| episode limit | Gym/MuJoCo 500 | formal config 500 (default Brax is 1000) |
| reset | small random perturbation | Brax reset/backend implementation |
| physics | MuJoCo | Brax backend |

The result name for a future run must therefore be **source-faithful PD-MORL
reproduction with Brax environment adaptation**, not an exact MuJoCo paper
reproduction.

## 13. Training seed audit

The executable source default is seed 1 and derives child seeds as
`p_id * args.seed`. The paper specifies six runs but the source, README, and
local paper do not provide six numeric training seeds. Therefore:

**OFFICIAL TRAINING SEEDS NOT SPECIFIED.**

Any later `0..5` (or other set) are experiment seeds, not official seed values.

## 14. Evaluation protocol

Training: 201 preferences and three repeats. Training final hook: 1001
preferences and three repeats. Explicit offline/paper-report evaluation: 1001
preferences and six repeats with the frozen evaluation seed list
`[0,11,22,33,44,55]`. Evaluation randomness is not derived from the training
seed by a new protocol. Hypervolume uses frozen reference point `[0,0]`;
Sparsity follows the official source order independently for every repeat:
Pareto filter per repeat -> adjacent gap formula -> mean across repeats.
Previous evaluator implementation incorrectly applied sparsity to all evaluation
returns without Pareto filtering; this has been corrected and verified via Golden Tests.
The mean-return Pareto front remains a separate display artifact. No tolerance
was changed.

## 15. Paper reference metrics

The paper reports Walker reference values over six runs: HV
`5.41 ± 0.004 × 10^6`, sparsity `0.03 ± 0.005 × 10^4`, reference point
`(0,0)`. Because the formal run uses Brax, these are **reference-only** and
are not a pass/fail threshold.

## 16. Dry-run counter test

Command executed:

```text
python scripts/test_pd_morl_counters.py
```

It uses K=10, batch 256, `[400,400]`, policy delay 10, and three HER
relabels, while shortening only the budget to 100 worker steps. It passed with:

```json
{
  "workflow_rounds": 97,
  "final_worker_steps": 100,
  "valid_env_transitions": 1000,
  "prefill_env_transitions": 30,
  "replay_base_inserts": 1000,
  "replay_her_inserts": 2400,
  "replay_total_entries_written": 3400,
  "critic_optimizer_steps": 970,
  "actor_optimizer_steps": 97,
  "target_update_steps": 97
}
```

## 17. Formal reproduction config

`configs/experiment/pd_morl_walker_reproduction.yaml` is the source-equivalent
budget contract. Hydra resolution was validated with:

```text
python scripts/train.py --config-name experiment/pd_morl_walker_reproduction --cfg job
```

Startup logging emits `SOURCE_N`, `K`, `PREFILL_GLOBAL_TRANSITIONS`,
`PREFILL_STEPS_PER_WORKER`, `POST_PREFILL_WORKFLOW_ROUNDS`, final worker and
transition expectations, and expected critic/actor/target optimizer counts.
It rejects a `total_timesteps` value that adds prefill twice. Run completion
asserts exact worker steps, sampled transitions, and the prefill plus
post-prefill identity. Full training evaluations and the training-final evaluation persist their
returns, HV/sparsity per repeat, mean metrics, and Pareto progression in
`pd_morl_evaluations.jsonl`; checkpoints retain model/optimizer/replay and
interpolator state. The config is intentionally not launched in Step9.0.

The formal environment `/home/qiuquanj/miniforge3/envs/evorl` uses Python
3.11.16 and `orbax-checkpoint` 0.12.4. Its project `save`/`load` checkpoint
roundtrip passed. The separate Windows interpreter's Orbax import failure is
therefore an **EXTERNAL ENVIRONMENT ISSUE — NON-BLOCKING**.

## 18. Expected runtime and resources (ESTIMATE)

Step8 production-shape smoke is the only local timing anchor. The formal run
has approximately 10,000,000 valid transitions, so wall time is roughly the
smoke wall time multiplied by its measured transition ratio; evaluation and
checkpoint I/O can dominate near evaluation triggers. Replay capacity 2M plus
checkpoints requires several GB of disk in a typical float32 Walker run. The
actual runtime, peak memory, checkpoint size, and evaluation cost must be
measured by the first explicitly authorized single-seed run; these are
estimates, not results.

## 19. Remaining deviations

1. Physics is Brax rather than the source MuJoCo backend.
2. Official numeric training seeds are unspecified; future seeds must be
   labeled experiment seeds.
3. The paper table describes one 400-wide hidden layer, while executable
   source and EvoRL use `[400,400]`; executable source is the baseline.
4. Worker randomness is JAX lane-split in EvoRL rather than the source's
   process-local Python/NumPy streams; this is a framework RNG adaptation.

These are documented scope boundaries. No algorithm schedule, Step6
tolerance, data-parallel learner, or production run was changed or started.

## Gate result

All 21 Step9.0 gates are explicit: source N and stop semantics, transition and
learner accounting, total-it meaning, HER separation, warm-up, evaluation
triggers, key-pretraining separation, hyperparameters, environment deviation,
seed search, six-run protocol, dry-run arithmetic, formal config, and unchanged
algorithm/tolerance behavior.

**STEP9.0 PD-MORL REPRODUCTION PROTOCOL PASS**

Suggested next command (do not execute automatically): one full Walker seed
using the formal config, after user authorization.
