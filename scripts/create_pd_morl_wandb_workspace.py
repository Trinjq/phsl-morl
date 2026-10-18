"""Save W&B sweep charts; sampling sweeps can reuse a previous baseline group."""

import argparse
import json
from pathlib import Path


def main():
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group")
    parser.add_argument("--project", default="pd-morl")
    parser.add_argument("--entity")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline-group", help="reuse its rollout=4, critic=320 runs")
    mode.add_argument(
        "--seeds", nargs="+", type=int, help="full-budget seed comparison"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="show completed seeds before the whole seed scan finishes",
    )
    args = parser.parse_args()
    if args.allow_partial and args.seeds is None:
        parser.error("--allow-partial requires --seeds")
    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    runs = list(api.runs(f"{entity}/{args.project}", filters={"group": args.group}))
    if args.allow_partial:
        runs = [
            run
            for run in runs
            if run.state == "finished" and run.summary.get("import_complete")
        ]
    filters = [expr.Metric("Group") == args.group]
    if args.allow_partial:
        filters = expr.And(
            expr.Metric("Group") == args.group, expr.Metric("State") == "finished"
        )
    group_key = "critic_updates_per_rollout"
    group_label = "critic update count"
    colors = {80: "#0072B2", 160: "#E69F00", 320: "#009E73"}
    workspace_name = "PD-MORL 2M critic sweep"
    if args.seeds is not None:
        actual_seeds = {r.config["seed"] for r in runs}
        matching = (
            actual_seeds <= set(args.seeds)
            if args.allow_partial
            else actual_seeds == set(args.seeds)
        )
        if len(set(args.seeds)) != len(args.seeds) or not matching:
            raise ValueError("seed comparison must match the requested distinct seeds")
        group_key = "seed"
        group_label = "training seed"
        palette = [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7",
            "#56B4E9",
            "#000000",
        ]
        colors = {
            seed: palette[i % len(palette)] for i, seed in enumerate(sorted(args.seeds))
        }
        workspace_name = "PD-MORL full-budget seed comparison"
    if args.baseline_group:
        if len(runs) != 6:
            raise ValueError("expected six new sampling runs")
        baselines = list(
            api.runs(
                f"{entity}/{args.project}",
                filters={
                    "group": args.baseline_group,
                    "config.critic_updates_per_rollout": 320,
                    "config.rollout_length": 4,
                },
            )
        )
        if len(baselines) != 3:
            raise ValueError("expected three existing rollout=4, critic=320 baselines")
        runs.extend(baselines)
        if {(r.config["rollout_length"], r.config["seed"]) for r in runs} != {
            (length, seed) for length in (2, 4, 8) for seed in (0, 1, 2)
        }:
            raise ValueError("sampling comparison requires three lengths x three seeds")
        for run in runs:
            config = run.config
            if (
                config["num_envs"] != 160
                or config["replay_batch_size"] != 512
                or config["actor_update_interval"] != 10
                or config["total_timesteps"] != 2_000_960
                or config["critic_updates_per_rollout"] != 80 * config["rollout_length"]
            ):
                raise ValueError(
                    "sampling runs must have equal update/transition budgets"
                )
        if len({r.config["key_artifact_sha256"] for r in runs}) != 1:
            raise ValueError("sampling runs must share one key artifact")
        filters = expr.Or(
            expr.Metric("Group") == args.group,
            expr.And(
                expr.Metric("Group") == args.baseline_group,
                expr.Config("critic_updates_per_rollout") == 320,
                expr.Config("rollout_length") == 4,
            ),
        )
        group_key = "rollout_length"
        group_label = "rollout length"
        colors = {2: "#0072B2", 4: "#E69F00", 8: "#009E73"}
        workspace_name = "PD-MORL 2M sampling cadence"
    expected_runs = len(args.seeds) if args.seeds is not None else 9
    if args.allow_partial:
        expected_runs = len({run.config["seed"] for run in runs})
    if (
        not runs
        or len(runs) != expected_runs
        or any(run.state != "finished" for run in runs)
    ):
        raise ValueError(f"expected {expected_runs} finished runs")
    groupby = wr.Config(group_key)
    curve_groupby = (
        wr.Config("critic_updates_per_rollout") if args.seeds is not None else groupby
    )
    total = max(run.config["total_timesteps"] for run in runs)

    def line(title, metrics, grouped=False):
        return wr.LinePlot(
            title=title,
            x="env_steps",
            y=metrics,
            title_x="Base environment transitions",
            max_runs_to_show=len(runs),
            smoothing_type="none",
            range_x=(0, total),
            point_visualization_method="sampling",
            groupby=curve_groupby if grouped else None,
            groupby_aggfunc="mean" if grouped else None,
            groupby_rangefunc="stddev" if grouped else None,
        )

    sections = [
        ws.Section(
            name="Evaluation: 201 preferences x 3 repeats",
            is_open=True,
            panels=[
                line(f"HV: all {len(runs)} runs", ["eval/source_hv"]),
                line("HV: mean and variation across seeds", ["eval/source_hv"], True),
                line(
                    "Sparsity: mean and variation across seeds",
                    ["eval/source_sparsity"],
                    True,
                ),
                line("Pareto point count", ["eval/source_pareto_point_count"]),
            ],
        ),
        ws.Section(
            name="Training",
            is_open=True,
            panels=[
                line("Critic loss", ["train/critic_loss"]),
                line("Actor loss", ["train/actor_loss"]),
                line("Actor directional angle", ["train/actor_angle"]),
                line("Critic gradient norm", ["train/critic_raw_grad_norm"]),
            ],
        ),
        ws.Section(
            name="Key anchor evaluations",
            is_open=True,
            panels=[
                line(
                    f"Key {label}: candidate vs stored score before update",
                    [f"key/{i}/candidate_score", f"key/{i}/stored_score"],
                )
                for i, label in enumerate(("[0,1]", "[0.5,0.5]", "[1,0]"))
            ],
        ),
        ws.Section(
            name="Final evaluation: 1001 preferences x 3 repeats",
            is_open=True,
            panels=[
                wr.BarPlot(
                    title=f"Final HV by {group_label}",
                    metrics=["final/source_hv"],
                    groupby=groupby,
                    groupby_aggfunc="mean",
                    groupby_rangefunc="stddev",
                ),
                wr.BarPlot(
                    title=f"Final sparsity by {group_label}",
                    metrics=["final/source_sparsity"],
                    groupby=groupby,
                    groupby_aggfunc="mean",
                    groupby_rangefunc="stddev",
                ),
                wr.MediaBrowser(
                    title="Saved HV figures",
                    media_keys=["plots/hv_curve"],
                    num_columns=3,
                    mode="grid",
                    grid_x_axis="run",
                ),
            ],
        ),
    ]
    output = Path(__file__).resolve().parents[1] / "outputs/wandb_imports" / args.group
    output.mkdir(parents=True, exist_ok=True)
    pointer = output / "visualization_workspace.json"
    if pointer.exists():
        workspace = ws.Workspace.from_url(json.loads(pointer.read_text())["url"])
        workspace.sections = sections
    else:
        workspace = ws.Workspace(
            entity=entity,
            project=args.project,
            name=workspace_name,
            sections=sections,
            auto_generate_panels=False,
        )
    workspace.runset_settings = ws.RunsetSettings(
        filters=filters,
        run_settings={
            run.id: ws.RunSettings(color=colors[run.config[group_key]]) for run in runs
        },
    )
    workspace.save()
    verified = ws.Workspace.from_url(workspace.url)
    assert sum(len(section.panels) for section in verified.sections) == 14
    assert args.group in str(verified.runset_settings.filters)
    if args.baseline_group:
        assert args.baseline_group in str(verified.runset_settings.filters)
    result = dict(url=workspace.url, group=args.group, runs=len(runs), panels=14)
    if args.seeds is not None:
        result["pending_seeds"] = sorted(
            set(args.seeds) - {run.config["seed"] for run in runs}
        )
    pointer.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
