# EvoRL PD-MORL Baseline

## 1. Scope

This is the single-GPU EvoRL integration of the frozen PD-MORL baseline. It
connects the Step 4--6.1 learner, control, and evaluation components without
adding a new algorithm. Multi-GPU execution, PSL-MORL, and paper-scale
reproduction are outside this step.

## 2. STEP7 INTEGRATION AUDIT

The entry audit was completed before integration edits.

| Item | Entry status | Finding and Step 7 action |
|---|---|---|
| MO-TD3 Agent | ALREADY CONNECTED | Vector actor/critics and frozen losses were live; exposed canonically as `PDMORLAgent` with a compatibility alias. |
| MO-TD3 Workflow | NEEDS INTEGRATION | Learner path existed; exposed as `PDMORLWorkflow` and completed host metrics/state plumbing. |
| Preference wrapper | ALREADY CONNECTED | `EpisodePreferenceWrapper` fixes one preference per episode. DO NOT CHANGE. |
| Preference subspace | ALREADY CONNECTED | The official 1001-point grid is split across logical workers. DO NOT CHANGE. |
| HER | ALREADY CONNECTED | Source insertion order and activation semantics are used. A prefill ordering bug was fixed so episode counting sees `dones` before replay cleanup. |
| Interpolator state | ALREADY CONNECTED | Fixed-shape arrays are in `AgentState`; no SciPy object enters JIT. DO NOT CHANGE. |
| PD-MORL alignment loss | ALREADY CONNECTED | Frozen Step 5.5 functions are called directly. DO NOT CHANGE. |
| Replay | ALREADY CONNECTED | One native EvoRL global `ReplayBuffer` is shared by all logical lanes. DO NOT CHANGE. |
| `episode_count` | ALREADY CONNECTED | Fixed `[K]`, cumulative, and includes prefill. |
| Key update controller | IMPLEMENTED BUT NOT CONNECTED | Step 6.1 functions existed inline; `KeyInterpolatorUpdateController` now owns the existing evaluate/refit call. |
| `PDMORLEvaluator` | ALREADY CONNECTED | Host-side key, train-grid, final, and offline evaluation reuse it. DO NOT CHANGE. |
| Pareto/HV/sparsity | ALREADY CONNECTED | Step 6.1 implementations are reused outside learner JIT. DO NOT CHANGE. |
| Config | NEEDS INTEGRATION | Renamed public fields by responsibility and retained frozen production values. |
| Logging | NEEDS INTEGRATION | MORL diagnostics were incomplete; metrics now use centralized `morl/train`, `morl/control`, and `morl/eval` names. |
| Checkpoint/restore | IMPLEMENTED BUT NOT CONNECTED | Native state checkpointing already covered the PyTree; an online-refit round-trip test now verifies restored `I(w)` and all counters/replay. |

## 3. Final architecture

`PDMORLAgent` owns only policy/value forward passes and frozen learner losses.
`PDMORLWorkflow` owns environment interaction, logical lanes, replay/HER,
learner scheduling, counters, host control, evaluation, logging, and state
checkpointing. `KeyInterpolatorUpdateController` and `PDMORLEvaluator` remain
host-side components.

## 4. Reused EvoRL components

The integration directly reuses:

- `TD3Workflow.step` for sampling, optimizer application, delayed actor update,
  and target updates;
- `OffPolicyWorkflowTemplate` for prefill, multi-step JIT, recorder calls, and
  checkpoint saves;
- `rollout`, `ReplayBuffer`, `agent_gradient_update`,
  `soft_target_update`, `WorkflowMetric`, and Orbax checkpoint utilities;
- Brax creation/wrappers and the existing recorder chain.

The only shared-framework additions are an optional gradient-norm auxiliary
metric and a recorder formatting hook. Scalar TD3 retains its default behavior.

## 5. PD-MORL-specific components

PD-MORL overrides the preference-conditioned actor and vector critics, vector
target/loss functions, HER replay write, per-worker episode/step counters,
parallel source update schedule, and host control/evaluation hooks. New helpers
are limited to `KeyInterpolatorUpdateController` and diagnostic fields.

## 6. Agent responsibilities

`PDMORLAgent` holds actor, critics, targets, interpolator arrays, raw key
solutions, counters, and fixed logical-worker state. It computes actions,
whole-vector pessimistic targets, `wp = I(w)`, critic loss, and delayed actor
loss. It does not perform Pareto filtering, hypervolume, SciPy refits, or
evaluation.

## 7. Workflow responsibilities

`PDMORLWorkflow` creates one Brax batch with `B=K`, maintains the global replay,
runs source-faithful HER, delegates updates to TD3, advances `total_it` by `K`
per learner round, calls host control/evaluation after JIT, records metrics, and
saves the complete workflow state.

## 8. Full training dataflow

Logical lane -> episode preference -> actor -> noisy/warmup action -> Brax
vector reward -> transition with `(s,a,r_vec,s_next,w,done)` -> HER -> global
replay -> minibatch -> target actor and twin vector Q -> scalarized whole-vector
selection -> Bellman target -> `I(w)` -> frozen critic loss/update ->
`policy_freq` gate -> frozen actor loss/update -> soft target update.

## 9. Control-plane dataflow

Fixed state contains `episode_count[K]`, `eval_cnt_ep`, `eval_cnt`, raw key
solutions, and interpolator arrays. On the strict key trigger, the counter is
incremented first, the three key preferences are evaluated, repeat means are
compared by strict scalar improvement, and the existing online L1/SciPy linear
RBF refit replaces only fixed-shape interpolator arrays. Refit occurs even when
no key solution is replaced.

## 10. Evaluation-plane dataflow

The full trigger calls the current actor through `PDMORLEvaluator`, using the
201-point production training grid and three repeats. It records the current
Pareto count, mean HV, and mean sparsity. Final evaluation uses the 1001-point
grid with three repeats; explicit offline evaluation uses six repeats.

## 11. Config

Algorithm fields are `num_objectives`, `process_count`,
`num_relabel_preferences`, `actor_loss_coeff`, `random_action_warmup`,
`learner_start_threshold`, `replay_capacity`, `policy_freq`, `gamma`, `tau`,
`exploration_noise`, `target_policy_noise`, `noise_clip`, and
`gradient_clip_norm`. Control fields are `interpolator_artifact`,
`key_update_enabled`, `eval_freq`, `eval_episodes`,
`training_eval_preference_step`, and `offline_eval_preference_step`. Framework
precision is explicitly `matmul_precision: highest`.

The linear kernel, initial L2 normalization, online L1 normalization, absent
post-normalization, and critic angle coefficient 1 are not public tuning
fields. Walker `actor_loss_coeff` is 10.

## 12. Workflow state

The fixed-shape PyTree contains actor/critic/target parameters, both optimizer
states, replay, PRNG/environment state, preferences, worker steps,
`episode_count[K]`, `eval_cnt_ep`, `eval_cnt`, `total_it`, key/refit counters,
raw key solutions `[3,2]`, interpolator arrays, and workflow counters. SciPy
objects and dynamic Pareto arrays remain on the host.

## 13. Checkpoint / restore

Native Orbax serialization saves the complete state when
`save_replay_buffer: true`. The integration test performs an online refit,
saves, restores, and checks exact `I(w)`, learner/control counters, replay size,
parameters, targets, optimizer states, PRNG, environment, and logical-worker
state through the restored PyTree. Restore does not reload the initial artifact.

## 14. Toy smoke

`tests/test_pd_morl_integration.py` contains a two-objective continuous
`TEST FIXTURE`. On laboratory GPU it passed setup, prefill, rollout, vector
reward, preference propagation, HER/global replay, learner and target updates,
control evaluation/refit, metrics, sensitivity, and checkpoint/restore.

## 15. Walker smoke

The real Brax Walker smoke runs on one RTX 4090 with `B=K=10`, GPU float32, and
`matmul_precision=highest`. `TEST OVERRIDE` shortens the episode, warmup,
trigger threshold, network, and evaluation grid only inside the test. It passed
real rollout/replay/HER, critic updates, delayed actor/target updates, counters,
both control triggers, online refit, evaluation, Pareto/HV/sparsity, preference
sensitivity, finite diagnostics, and checkpoint restore. The isolated smoke
completed in 455.72 seconds.

The shortened network above records the historical Step 7 run: that test used
one 16-unit hidden layer. The obsolete reduced-network override was removed
after Step 7.1. Current PD-MORL Walker smoke and integration tests obtain both
`[400,400]` hidden-layer lists from `configs/agent/mo-td3.yaml`; network-size
overrides are no longer permitted on a real-Walker test path.

## 16. Preference sensitivity

For one checkpoint and controlled observations, deterministic actions were
evaluated for `[0,1]`, `[0.5,0.5]`, and `[1,0]`. Actions differed measurably;
the same three preferences also produced finite vector trajectory returns.
This confirms conditioning survived integration and is not a paper-performance
claim or an expected objective ordering.

## 17. Stability diagnostics

The workflow records critic total/Smooth-L1/angle components, actor
scalarized/angle/total components, raw actor and critic gradient norms, Q1/Q2
min/mean/max, `wp` min/mean/max, replay size, HER state, episode count range,
subspace count, key/refit counts, HV, sparsity, and Pareto count. Smoke tests
require every numerical diagnostic to be finite and verify both loss identities.

## 18. Evaluation integration

No evaluator was reimplemented. The workflow invokes the existing
`PDMORLEvaluator`, `update_key_solutions`, Pareto, pymoo HV, and sparsity paths
using the current actor and interpolator state. Host results enter the recorder
under `morl/eval/*`.

## 19. Regression

Laboratory validation used `CUDA_VISIBLE_DEVICES=0` and `pytest --device gpu`.
The complete Step 4--7 frozen suite, including scalar TD3 and both real-Walker
tests, finished with `125 passed` in 1353.79 seconds. Step 6 tolerances were
unchanged.

## 20. Source-faithful / adaptation table

| Classification | Behavior |
|---|---|
| SOURCE-FAITHFUL | Network/loss mathematics, whole-vector target selection, HER, `K` update schedule, strict triggers/replacement, L2/L1 normalization, grids/repeats, Pareto/HV/sparsity |
| FRAMEWORK-ADAPTATION | PyTorch processes become JAX logical lanes; host hook exports SciPy-fit arrays; GPU float32 uses highest matmul precision; native EvoRL replay/optimizer/checkpoint/logger are reused |
| TEST FIXTURE | Two-objective toy environment and fixture key values |
| TEST OVERRIDE | Historical Step 7 used a shortened network; current tests only shorten Walker episode/warmup/threshold/evaluation cost |
| DEVIATION | None introduced in the algorithm or production control semantics |

## 21. Remaining known deviations

The official experiments ran PyTorch on CPU, while this baseline runs JAX on a
single GPU with the already frozen highest-precision adaptation. Brax reset
keys retain the documented deterministic folding adaptation. No additional
algorithm deviation is present.

## 22. Remaining work before paper reproduction

Paper reproduction still requires the full one-million-step training budget,
six-run mean/std reporting, final paper HV/sparsity, and figure reproduction.
Multi-GPU and PSL-MORL remain separate later stages and are not part of this
baseline integration.

## Status

`STEP7 END-TO-END EVORL PD-MORL PASS`

## Production-Shape Single-GPU Smoke

Step 7.1 adds a separate `gpu/slow/integration` test at
`tests/test_pd_morl_production_shape_smoke.py`. Ordinary CPU test runs skip it.
The gate ran on one laboratory NVIDIA RTX 4090 as `cuda:0` with JAX backend
`gpu`, float32 parameters, and the project-configured
`matmul_precision=highest`.

The test derives its configuration from the production Walker YAML and asserts
that all prohibited overrides remain equal to production values:

| Item | Runtime value |
|---|---:|
| objectives | 2 |
| logical workers / Brax batch | `K=10`, `B=10` |
| learner batch | 256 |
| actor hidden layers | `[400,400]` |
| both critic hidden layers | `[400,400]` |
| replay capacity | 2,000,000 |
| actor loss coefficient / policy frequency | `10 / 10` |
| gamma / tau | `0.995 / 0.005` |
| exploration / target noise / noise clip | `0.1 / 0.2 / 0.5` |
| gradient clip norm / HER relabels | `100 / 3` |

The runtime parameter and learner-shape assertions produced:

| Tensor/layer | Shape |
|---|---:|
| state batch | `[256,17]` |
| preference | `[256,2]` |
| action | `[256,6]` |
| critic input | `[256,25]` |
| twin Q / each Q | `[256,2,2]` / `[256,2]` |
| projected preference `wp` | `[256,2]` |
| TD target | `[256,2]` |
| actor layer 1 / layer 2 / output | `[19,400]` / `[400,400]` / `[400,6]` |
| Q1 layer 1 / layer 2 / output | `[25,400]` / `[400,400]` / `[400,2]` |
| Q2 layer 1 / layer 2 / output | `[25,400]` / `[400,400]` / `[400,2]` |

Q1 and Q2 were explicitly confirmed to have independent parameters. The only
`TEST OVERRIDE` values were episode length, prefill/learner-start timing,
random-action and HER activation timing, key/full trigger timing, evaluation
grid/repeats, and checkpoint output location. No network, learner batch,
replay-capacity, algorithm, or numerical-policy value was shortened.

The production-shape test exercised critic-only and delayed actor paths
separately: the critic changed while actor parameters remained identical on a
non-policy step, the actor changed at the `total_it % policy_freq == 0` gate,
and both targets changed only with the delayed update. The connected workflow
then completed rollout, physical HER insertion, global replay sampling, three
production-shape learner rounds, key evaluation, online L1 refit, full
evaluation, and checkpoint/restore.

All critic component losses, actor component losses, raw gradient norms,
Q1/Q2 min/mean/max, `wp` min/mean/max, TD-target min/mean/max, replay/counter
metrics, HV, sparsity, and Pareto count were finite. The observed TD-target
min/mean/max was `0.46567816 / 2.8466501 / 5.147126`. The key-update and refit
counts were both one, HER was active, and the evaluation produced a non-empty
Pareto set. Controlled deterministic actions for `[0,1]`, `[0.5,0.5]`, and
`[1,0]` differed measurably.

After online refit, checkpoint/restore exactly preserved actor, critics,
targets, optimizer states, PRNG state, `total_it`, episode/control counters,
raw key solutions, worker state, replay size, interpolator arrays, and `I(w)`.
The initial artifact was not reloaded.

First JIT compilation completed in 52.55 seconds and a steady update completed
in 0.0247 seconds. JAX made one expected input-sharding specialization when
initial setup arrays became committed; the cache remained stable after online
refit. The control hook now places refitted arrays on the previous state's
sharding, avoiding an otherwise unnecessary learner recompile without changing
any numerical value. Peak sampled GPU memory was 2640 MiB; no OOM occurred.
The smoke passed in 421.73 seconds.

The post-smoke frozen GPU regression completed with `85 passed` in 579.62
seconds, covering Step 5.4, Step 5.5, Step 6 Golden Tests, Step 6.1,
Step 7 integration, real Walker, and scalar TD3. Frozen tolerances and the
production Walker config were unchanged.

`STEP7.1 PRODUCTION-SHAPE SINGLE-GPU SMOKE PASS`

## Reduced-Network Override Removal

After Step 7.1, the obsolete `[16]` actor/critic override was removed from the
real-Walker Step 7 integration test. Both real-Walker tests now read network
sizes from `configs/agent/mo-td3.yaml` and assert the built actor, Q1, and Q2
shapes are the production `[400,400]` architecture. Toy and isolated unit tests
retain their small fixture networks because they are not Walker paths.

On the laboratory GPU, the production-shape smoke passed (`1 passed` in
723.04 seconds), followed by Step 7 integration, Step 6 Golden Tests, Step 6.1,
and scalar TD3 regression (`73 passed` in 773.65 seconds). Frozen tolerances
were unchanged. No PD-MORL real-Walker small-network path remains.
