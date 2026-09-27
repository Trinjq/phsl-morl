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

SOURCE-FAITHFUL: the algorithm, K=10, batch 256, replay/HER, losses, control,
evaluation, checkpoint semantics, and single-GPU training behavior.

FRAMEWORK-ADAPTATION: independent OS-process assignment of complete runs to
different GPUs.

EXPERIMENT ORCHESTRATION: seed scheduling, isolated run directories, and
offline aggregation.

DEVIATION: none. The archived Step8 data-parallel learner is not re-enabled.
No 1M-step production runs are started by this change.
