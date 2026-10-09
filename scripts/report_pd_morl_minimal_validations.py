"""Generate the two requested reports and schedule-ablation figures."""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

A_COLOR="#0F4D92"
B_COLOR="#B64342"
SEED_COLORS=("#0F4D92","#42949E","#9A4D8E")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values=[float(x) for x in values]
    return {"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"min":min(values),"max":max(values)}


def fmt(value):
    return f"{value:,.3f}"


def configure_plotting():
    plt.rcParams.update({
        "font.family":["Arial","Helvetica","DejaVu Sans","sans-serif"],
        "font.size":12,
        "axes.spines.right":False,
        "axes.spines.top":False,
        "axes.linewidth":1.5,
        "legend.frameon":False,
        "svg.fonttype":"none",
    })


def save_figure(fig,path):
    fig.tight_layout(pad=2)
    fig.savefig(path.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def key_report(repo,existing_path):
    runs=[load(repo/f"outputs/key_source_like_w05/seed_{seed}/result.json") for seed in (1,2,3)]
    old=load(existing_path)
    old_balanced=next(row for row in old["aggregates"] if row["preference"]==[0.5,0.5])
    new={
        "r1":stats(row["vector_return"][0] for row in runs),
        "r2":stats(row["vector_return"][1] for row in runs),
        "scalarized_return":stats(row["scalarized_return"] for row in runs),
        "direction_r1":stats(row["normalized_direction"][0] for row in runs),
        "direction_r2":stats(row["normalized_direction"][1] for row in runs),
    }
    old_dir=np.array([old_balanced["normalized_direction_r1"]["mean"],old_balanced["normalized_direction_r2"]["mean"]])
    new_dir=np.array([new["direction_r1"]["mean"],new["direction_r2"]["mean"]])
    angle=math.degrees(math.acos(np.clip(old_dir@new_dir/(np.linalg.norm(old_dir)*np.linalg.norm(new_dir)),-1,1)))
    scalar_change=100*(new["scalarized_return"]["mean"]/old_balanced["scalarized_return"]["mean"]-1)
    variance_ratio=new["scalarized_return"]["sample_std"]/old_balanced["scalarized_return"]["sample_std"]
    lines=[
        "# Key [0.5, 0.5] Source-Like Sanity Report","",
        "## Worthwhile question","",
        "This sanity check tests whether restoring the official key-training protocol materially changes the balanced Brax Walker2d key solution without replacing the frozen Key Artifact.","",
        "## Protocol","",
        "Only preference noise, learner start after the first 256 replay entries (the first vectorized chunk satisfying replay size >=200), evaluation every 100 completed episodes with 10 episodes, and strict noiseless evaluation at [0.5, 0.5] differ from the prior key-training script. Random actions continue through 25,088 transitions (the nearest complete 256-transition chunk to 25,000). No HER, RBF, or directional-angle loss is used.","",
        "## Seed results","",
        "| Seed | Best R1 | Best R2 | 0.5 R1 + 0.5 R2 | Direction J/||J||2 | Best step | Evaluations | Runtime (min) |","|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(f"| {row['seed']} | {fmt(row['vector_return'][0])} | {fmt(row['vector_return'][1])} | {fmt(row['scalarized_return'])} | [{row['normalized_direction'][0]:.4f}, {row['normalized_direction'][1]:.4f}] | {row['best_step']:,} | {len(row['evaluation_history'])} | {row['runtime_seconds']/60:.2f} |")
    lines += ["","## Comparison with the existing 30-seed balanced-policy results","",
        "| Quantity | New 3 seeds mean ± sample SD | Existing 30 seeds mean ± sample SD |","|---|---:|---:|",
        f"| R1 | {fmt(new['r1']['mean'])} ± {fmt(new['r1']['sample_std'])} | {fmt(old_balanced['r1']['mean'])} ± {fmt(old_balanced['r1']['sample_std'])} |",
        f"| R2 | {fmt(new['r2']['mean'])} ± {fmt(new['r2']['sample_std'])} | {fmt(old_balanced['r2']['mean'])} ± {fmt(old_balanced['r2']['sample_std'])} |",
        f"| Scalarized return | {fmt(new['scalarized_return']['mean'])} ± {fmt(new['scalarized_return']['sample_std'])} | {fmt(old_balanced['scalarized_return']['mean'])} ± {fmt(old_balanced['scalarized_return']['sample_std'])} |",
        f"| Direction R1 | {new['direction_r1']['mean']:.4f} ± {new['direction_r1']['sample_std']:.4f} | {old_balanced['normalized_direction_r1']['mean']:.4f} ± {old_balanced['normalized_direction_r1']['sample_std']:.4f} |",
        f"| Direction R2 | {new['direction_r2']['mean']:.4f} ± {new['direction_r2']['sample_std']:.4f} | {old_balanced['normalized_direction_r2']['mean']:.4f} ± {old_balanced['normalized_direction_r2']['sample_std']:.4f} |","",
        "## Findings","",
        f"A. The mean scalarized return changed by {scalar_change:+.1f}% relative to the existing 30-seed mean.",
        f"B. The angle between the two mean normalized directions is {angle:.2f} degrees.",
        f"C. The 3-seed scalarized-return sample SD is {variance_ratio:.2f}× the existing 30-seed SD. This is descriptive only because n=3 and n=30 are unequal.","",
        "The frozen Key Artifact was not modified, and these results do not authorize retraining all 30 key seeds.",
    ]
    (repo/"docs/KEY_SOURCE_LIKE_SANITY_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def front_distance(fronts):
    pooled=np.vstack(fronts)
    span=np.maximum(pooled.max(0)-pooled.min(0),1e-12)
    norm=[(front-pooled.min(0))/span for front in fronts]
    distances=[]
    for i in range(len(norm)):
        for j in range(i+1,len(norm)):
            d=np.linalg.norm(norm[i][:,None,:]-norm[j][None,:,:],axis=-1)
            distances.append((d.min(1).mean()+d.min(0).mean())/2)
    return statistics.mean(distances)


def schedule_report(repo):
    groups={}
    for name in ("source_like","gpu_native"):
        rows=[]
        for seed in (1,2,3):
            run=repo/f"outputs/pd_morl_schedule_stability/{name}/seed_{seed}"
            result=load(run/"final_training_eval/results.json")
            counters=load(run/"counters_summary.json")
            idx=next(i for i,p in enumerate(result["preferences"]) if p==[0.5,0.5])
            front=np.asarray(result["pareto_returns"],dtype=float)
            rows.append({
                "seed":seed,"hv":result["source_hv"],"sparsity":result["source_sparsity"],
                "count":result["source_pareto_point_count"],"front":front,
                "r1_range":[float(front[:,0].min()),float(front[:,0].max())],
                "r2_range":[float(front[:,1].min()),float(front[:,1].max())],
                "w05":result["mean_returns"][idx],"critic":counters["critic_optimizer_step_count"],
                "actor":counters["actor_optimizer_step_count"],"runtime":result["wall_clock_seconds"],
            })
        groups[name]=rows
    figures=repo/"outputs/pd_morl_schedule_stability/figures"
    figures.mkdir(parents=True,exist_ok=True)
    configure_plotting()
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,seed in zip(axes,(1,2,3)):
        for name,color,label in (("source_like",A_COLOR,"Source-like"),("gpu_native",B_COLOR,"GPU-native")):
            front=groups[name][seed-1]["front"]
            ax.plot(front[:,0],front[:,1],"o-",ms=2,lw=1.5,color=color,label=label)
        ax.set(title=f"Seed {seed}",xlabel="R1",ylabel="R2")
    axes[0].legend()
    save_figure(fig,figures/"pareto_front_per_seed")
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,(name,title) in zip(axes,(("source_like","Source-like"),("gpu_native","GPU-native"))):
        for row,color in zip(groups[name],SEED_COLORS):
            ax.plot(row["front"][:,0],row["front"][:,1],"o-",ms=2,lw=1.3,color=color,label=f"Seed {row['seed']}")
        ax.set(title=title,xlabel="R1",ylabel="R2")
        ax.legend()
    save_figure(fig,figures/"pareto_front_three_seed_overlay")
    for metric,label,filename in (("hv","HV","hv_comparison"),("sparsity","Sparsity","sparsity_comparison"),("count","Pareto point count","pareto_point_count_comparison")):
        fig,ax=plt.subplots(figsize=(6.5,4.5))
        x=np.arange(3); width=.36
        a=[row[metric] for row in groups["source_like"]]; b=[row[metric] for row in groups["gpu_native"]]
        ax.bar(x-width/2,a,width,color=A_COLOR,edgecolor="black",label="Source-like")
        ax.bar(x+width/2,b,width,color=B_COLOR,edgecolor="black",hatch="//",label="GPU-native")
        ax.set(xticks=x,xticklabels=["Seed 1","Seed 2","Seed 3"],ylabel=label)
        ax.legend()
        save_figure(fig,figures/filename)
    a_disp=front_distance([row["front"] for row in groups["source_like"]])
    b_disp=front_distance([row["front"] for row in groups["gpu_native"]])
    a_sp=stats(row["sparsity"] for row in groups["source_like"]); b_sp=stats(row["sparsity"] for row in groups["gpu_native"])
    stability=a_disp<=.8*b_disp
    sparsity=a_sp["mean"]<=.8*b_sp["mean"]
    lines=["# PD-MORL Schedule Stability Report","","## Worthwhile question","","This paired ablation tests whether the training schedule, with the algorithm, artifact, losses, reward, network, optimizer, transition budget, seeds, and evaluation protocol fixed, is a major source of Pareto-front instability.","","## Smoke validation","",f"See `outputs/pd_morl_schedule_stability/smoke/validation.json`. Both runs used the GPU backend, finite losses, active HER, key evaluation/RBF refits, and consistent counters; the A run recorded 10 critic updates per 10-transition rollout.","","## Final paired results","","| Schedule | Seed | HV | Sparsity | Points | R1 range | R2 range | w=[0.5,0.5] return | Critic / actor updates | Runtime (h) |","|---|---:|---:|---:|---:|---|---|---|---|---:|"]
    for name,label in (("source_like","A: Source-like"),("gpu_native","B: GPU-native")):
        for row in groups[name]:
            lines.append(f"| {label} | {row['seed']} | {fmt(row['hv'])} | {fmt(row['sparsity'])} | {row['count']} | [{fmt(row['r1_range'][0])}, {fmt(row['r1_range'][1])}] | [{fmt(row['r2_range'][0])}, {fmt(row['r2_range'][1])}] | [{fmt(row['w05'][0])}, {fmt(row['w05'][1])}] | {row['critic']:,} / {row['actor']:,} | {row['runtime']/3600:.2f} |")
    lines += ["","## Cross-seed comparison","",f"The normalized symmetric nearest-front dispersion is {a_disp:.4f} for A and {b_disp:.4f} for B (A/B={a_disp/b_disp:.3f}; lower is more stable).",f"Mean sparsity is {fmt(a_sp['mean'])} for A and {fmt(b_sp['mean'])} for B (A/B={a_sp['mean']/b_sp['mean']:.3f}; lower is denser).","",f"1. Source-like schedule clearly lowers cross-seed front dispersion: **{'yes' if stability else 'no'}**.",f"2. Source-like schedule clearly lowers mean Sparsity: **{'yes' if sparsity else 'no'}**. Front-break evidence is interpreted jointly with the point-count plot and overlays.",f"3. The GPU-native schedule is a plausible major instability source: **{'yes' if stability and sparsity else 'inconclusive' if stability or sparsity else 'not supported'}**.","","“Clearly” is the preregistered descriptive threshold A/B <= 0.8. With only three paired seeds, these are diagnostic rather than population-level significance claims.","","## Figures","",*[f"- `{path.name}`" for path in sorted(figures.glob("*.png"))]]
    (repo/"docs/PD_MORL_SCHEDULE_STABILITY_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--existing-key-summary",type=Path,required=True)
    args=parser.parse_args()
    key_report(args.repo,args.existing_key_summary)
    schedule_report(args.repo)


if __name__=="__main__":
    main()
