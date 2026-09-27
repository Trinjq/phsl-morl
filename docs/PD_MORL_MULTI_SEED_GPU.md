# PD-MORL independent-seed GPU runs

The Step8 data-parallel learner is archived on
`archive/pd-morl-step8-data-parallel`. It is not part of the production
PD-MORL path: one process must see exactly one GPU and runs must not share
parameters, optimizer state, replay, PRNG state, interpolator state, or
checkpoints.

## Frozen Step7.2 baseline

Each run is the single-GPU EvoRL-PD-MORL baseline with the existing
source-faithful settings: Walker, K=10 logical preference workers, batch 256,
one replay, HER=3, actor and twin critics `[400, 400]`, actor loss coefficient
10, policy frequency 10, gamma 0.995, tau 0.005, target noise 0.2, noise clip
0.5, gradient clip 100, and highest matmul precision. The evaluation seed
protocol remains the evaluator's frozen `evaluation_seeds()` rule; it is not
derived from the training seed.

`configs/config.yaml:seed` is the explicit training seed. The normal
`scripts/train.py` entrypoint rejects a PD-MORL process that sees more than
one GPU and logs `distributed_data_parallel=false`.

## Independent-run launcher

Use explicit experiment seeds; they are not official paper seeds unless
defined elsewhere:

```bash
python scripts/run_pd_morl_seeds.py --seeds 0 1 2 --gpus 0 1 2
```

The launcher creates independent directories:

```text
outputs/pd_morl/walker/seed_<seed>/
```

Each child receives its own `CUDA_VISIBLE_DEVICES`, `seed`, Hydra output
directory, model, replay, optimizer, controller, and checkpoint. More seeds
than GPUs are queued and scheduled as a GPU becomes free. No child process
manages another child's JAX devices.

For a short smoke, keep the production network, K, and batch unchanged while
overriding only duration/evaluation cadence, for example:

```bash
python scripts/run_pd_morl_seeds.py --seeds 0 1 2 --gpus 0 1 2 \
  total_timesteps=10 learning_start_timesteps=1536 \
  learner_start_replay_entries=1536 recorders=[log]
```

Aggregate only completed final-evaluation metrics, never trajectories or
replay:

```bash
python scripts/aggregate_pd_morl_runs.py outputs/pd_morl
```

## Classification and stopping boundary

The short three-process Walker smoke was run on three RTX 4090 devices with
seeds `0,1,2`. `nvidia-smi` showed one distinct physical GPU UUID per child;
inside each child JAX reported only logical `cuda:0`, as expected after
`CUDA_VISIBLE_DEVICES` isolation. The smoke used the production network,
K=10, batch 256, and only shortened the duration. It did not start a
data-parallel learner or a formal 1M-step run.

A two-process same-seed initialization smoke (`seed=7` on GPUs 0 and 1)
produced the identical initialization fingerprint
`82a6d54885...ca163b` in both isolated run directories.

SOURCE-FAITHFUL: the algorithm, K=10, batch 256, replay/HER, losses, control,
evaluation, checkpoint semantics, and single-GPU training behavior.

FRAMEWORK-ADAPTATION: independent OS-process assignment of complete runs to
different GPUs.

EXPERIMENT ORCHESTRATION: seed scheduling, isolated run directories, and
offline aggregation.

DEVIATION: none. The archived Step8 data-parallel learner is not re-enabled.
No 1M-step production runs are started by this change.

## Final 3-GPU independent-seed learner smoke

The final concurrent smoke used the existing independent-run launcher on GPUs
0, 1, and 2, with seeds `0,1,2`. Each child saw exactly one JAX GPU and ran
Walker with actor/twin-critic `[400,400]`, K=10, batch 256, the existing replay
and HER path, and the existing delayed actor/target-update math. Only the run
length and evaluation triggers were shortened; the smoke did not start a
data-parallel learner or a 1M-step experiment.

Persistent output root:
`/home/qiuquanj/projects/evorl/outputs/pd_morl_learner_smoke7/`.

| seed | physical GPU UUID | JAX GPUs | critic updates | actor updates | target updates | HER | replay size | finite loss | checkpoint |
|---:|---|---:|---:|---:|---:|---|---:|---|---|
| 0 | `GPU-f3c66d4d-8c06-eeb7-6523-6513d4363087` | 1 | 133120 | 13312 | 13312 | true | 238640 | true | `walker/seed_0/checkpoints/512` |
| 1 | `GPU-df61bf6b-9281-e8ee-e006-fcd247e644b5` | 1 | 133120 | 13312 | 13312 | true | 238640 | true | `walker/seed_1/checkpoints/512` |
| 2 | `GPU-3e377ed9-38f1-5f1f-1c5f-c6312f44f6ed` | 1 | 133120 | 13312 | 13312 | true | 238640 | true | `walker/seed_2/checkpoints/512` |

The launcher summary is `independent_seed_smoke_summary.json` and reports
distinct model, optimizer, replay, and checkpoint directories for all three
children. The interpolator fingerprint is identical across children because
the shortened smoke intentionally disabled key replacement; each child still
owns an independent interpolator state in its own state/checkpoint directory.

Final status: **STEP8 INDEPENDENT-SEED MULTI-GPU RUNNER PASS**.

Stop boundary reached: no data-parallel learner was restored, no tolerance was
changed, and no 1M-step formal experiment was started.
