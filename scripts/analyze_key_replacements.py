import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("seed_*/key_replacement_diagnostics.csv")):
        frame = pd.read_csv(path)
        frame["source_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"no seed_*/key_replacement_diagnostics.csv under {root}"
        )
    data = pd.concat(frames, ignore_index=True)
    numeric = [
        "seed",
        "key_index",
        "actual_env_transitions",
        "stored_score",
        "candidate_score",
        "score_gap",
        "relative_gap_pct",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric)
    data["replaced"] = data["replaced"].astype(str).str.lower().eq("true")
    data["repeat_std"] = data["candidate_repeat_scores_3"].map(
        lambda value: float(np.std(json.loads(value), ddof=1))
    )
    return data.sort_values(["seed", "actual_env_transitions", "key_index"])


def _seed_axes(data: pd.DataFrame, ylabel: str):
    seeds = sorted(data.seed.unique())
    figure, axes = plt.subplots(
        len(seeds), 1, figsize=(8, 3.2 * len(seeds)), squeeze=False
    )
    for axis, seed in zip(axes[:, 0], seeds):
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Actual environment transitions")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    return figure, axes[:, 0], seeds


def _save_gap_plot(data: pd.DataFrame, root: Path):
    figure, axes, seeds = _seed_axes(data, "Relative gap (%)")
    for axis, seed in zip(axes, seeds):
        part = data[data.seed == seed]
        for key, rows in part.groupby("key_index"):
            axis.plot(
                rows.actual_env_transitions,
                rows.relative_gap_pct,
                marker="o",
                markersize=2,
                label=f"Key {key}",
            )
            hits = rows[rows.replaced]
            axis.scatter(
                hits.actual_env_transitions,
                hits.relative_gap_pct,
                marker="*",
                s=80,
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(root / "key_score_gap_by_seed.png", dpi=180)
    plt.close(figure)


def _save_score_plot(data: pd.DataFrame, root: Path):
    figure, axes, seeds = _seed_axes(data, "Scalarized score")
    for axis, seed in zip(axes, seeds):
        for key, rows in data[data.seed == seed].groupby("key_index"):
            x = rows.actual_env_transitions
            axis.step(x, rows.stored_score, where="post", label=f"Key {key} stored")
            axis.plot(x, rows.candidate_score, alpha=0.8, label=f"Key {key} candidate")
        axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "key_scores_by_seed.png", dpi=180)
    plt.close(figure)


def _save_event_plot(data: pd.DataFrame, root: Path):
    figure, axis = plt.subplots(figsize=(8, 4))
    for seed, rows in data.groupby("seed"):
        events = rows.groupby("actual_env_transitions").replaced.sum().cumsum()
        axis.step(events.index, events.values, where="post", label=f"Seed {seed}")
    axis.set(xlabel="Actual environment transitions", ylabel="Cumulative replaced keys")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "key_replacement_events.png", dpi=180)
    plt.close(figure)


def _save_hv_plot(data: pd.DataFrame, root: Path):
    seeds = sorted(data.seed.unique())
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for seed in seeds:
        rows = data[data.seed == seed]
        mean_gap = rows.groupby("actual_env_transitions").relative_gap_pct.mean()
        axes[0].plot(mean_gap.index, mean_gap.values, label=f"Seed {seed}")
        source = Path(rows.source_dir.iloc[0]) / "hv_convergence.csv"
        if source.exists():
            hv = pd.read_csv(source)
            hv = hv[(hv.kind == "convergence") & (hv.preference_count == 51)]
            axes[1].plot(hv.actual_env_transitions, hv.source_hv, label=f"Seed {seed}")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Mean relative key gap (%)")
    axes[1].set_ylabel("HV (51 preferences × 3)")
    axes[1].set_xlabel("Actual environment transitions")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(root / "key_gap_hv_comparison.png", dpi=180)
    plt.close(figure)


def _summaries(data: pd.DataFrame, root: Path):
    rows = []
    for (seed, key), group in data.groupby(["seed", "key_index"]):
        group = group.sort_values("actual_env_transitions")
        rows.append(
            {
                "seed": int(seed),
                "key_index": int(key),
                "evaluation_count": len(group),
                "replacement_count": int(group.replaced.sum()),
                "positive_gap_count": int((group.score_gap > 0).sum()),
                "min_relative_gap_pct": float(group.relative_gap_pct.min()),
                "median_relative_gap_pct": float(group.relative_gap_pct.median()),
                "start_relative_gap_pct": float(group.relative_gap_pct.iloc[0]),
                "final_relative_gap_pct": float(group.relative_gap_pct.iloc[-1]),
                "candidate_score_change": float(
                    group.candidate_score.iloc[-1] - group.candidate_score.iloc[0]
                ),
                "candidate_repeat_score_std_median": float(group.repeat_std.median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "key_replacement_summary.csv", index=False)
    payload = {
        "seed_key": rows,
        "seed_count": int(data.seed.nunique()),
        "key_eval_events": int(data.groupby(["seed", "event_id"]).ngroups),
        "replacement_events": int(
            data.groupby(["seed", "event_id"]).replaced.any().sum()
        ),
        "replaced_key_entries": int(data.replaced.sum()),
    }
    (root / "key_gap_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("reports/key_diagnostics"))
    args = parser.parse_args()
    data = _load_runs(args.root)
    _save_gap_plot(data, args.root)
    _save_score_plot(data, args.root)
    _save_event_plot(data, args.root)
    _save_hv_plot(data, args.root)
    _summaries(data, args.root)


if __name__ == "__main__":
    main()
