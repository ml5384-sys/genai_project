"""Render report figures from eval/results/*.json -> docs/figures/*.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "eval/results", ROOT / "docs/figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def hbar(ax, labels, values, colors, fmt="{:.3f}", err=None, xmax=None):
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=colors, height=0.62, xerr=err, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    xmax = xmax or max(values) * 1.18
    ax.set_xlim(0, xmax)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012 + (err[list(values).index(v)] if err is not None else 0), yi, fmt.format(v),
                va="center", fontsize=10, color=INK)


def fig_ranking():
    r = json.loads((RES / "ranking.json").read_text())
    m = r["methods"]
    order = ["Random", "TF-IDF (keyword)", "Embedding doc-cosine", "Skill Jaccard (exact)",
             "Coverage only (required+preferred)", "JobFit (full)", "Hybrid: JobFit + doc-cosine (z-avg)"]
    names = [o for o in order if o in m]
    vals = [m[n]["nDCG@10"] for n in names]
    errs = [m[n]["nDCG@10_ci"] for n in names]
    cols = [BLUE if n.startswith("JobFit") else (AQUA if n.startswith("Hybrid") else GRAY) for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    hbar(ax, names, vals, cols, err=errs, xmax=1.0)
    ax.set_xlabel("nDCG@10 (weak labels: resume category ↔ posting function; 95% CI)")
    ax.set_title(f"Ranking quality over {r['n_jobs']:,} postings, {r['n_resumes']} resumes", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "ranking_ndcg.png", dpi=200)

    abl = [n for n in m if n.startswith("  -")]
    full = m["JobFit (full)"]["nDCG@10"]
    deltas = [m[n]["nDCG@10"] - full for n in abl]
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    y = range(len(abl))[::-1]
    ax.barh(list(y), deltas, color=[ORANGE if d < 0 else AQUA for d in deltas], height=0.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([n.strip("- ").strip() for n in abl])
    ax.grid(axis="y", visible=False)
    lim = max(abs(d) for d in deltas) * 1.5 + 0.005
    ax.set_xlim(-lim, lim)
    for yi, d in zip(y, deltas):
        ax.text(d + (0.002 if d >= 0 else -0.002), yi, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right", fontsize=10)
    ax.set_xlabel(f"Δ nDCG@10 when the component is removed (full model = {full:.3f})")
    ax.set_title("Ablation: contribution of each component", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=200)


def fig_paraphrase():
    p = json.loads((RES / "paraphrase.json").read_text())
    mini = p["models"]["sentence-transformers/all-MiniLM-L6-v2"]["with descriptor"]
    groups = ["Keyword (taxonomy)", "Semantic only", "JobFit: keyword ∪ semantic"]
    rec = [p["keyword"]["recall"], mini["semantic_recall"], mini["combined_recall"]]
    fpr = [p["keyword"]["false_positive_rate"], mini["semantic_fpr"], mini["combined_fpr"]]
    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    x = range(len(groups))
    w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], rec, w, color=BLUE, label="Recall on paraphrased evidence")
    ax.bar([i + w / 2 + 0.01 for i in x], fpr, w, color=ORANGE, label="False-positive rate on hard negatives")
    for i in x:
        ax.text(i - w / 2 - 0.01, rec[i] + 0.02, f"{rec[i]:.0%}", ha="center", fontsize=10)
        ax.text(i + w / 2 + 0.01, fpr[i] + 0.02, f"{fpr[i]:.0%}", ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.set_title(f"Requirement matching on {p['n_cases']} hand-written paraphrase cases", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "paraphrase.png", dpi=200)


def fig_funnel():
    labels = ["LinkedIn postings (2023–24)", "Entry / Associate / Internship", "Target functions\n(FIN, ACCT, IT, CNSL, ANLS)", "Sampled job database"]
    vals = [123849, 47983, 12065, 8000]
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    hbar(ax, labels, vals, [GRAY, GRAY, GRAY, BLUE], fmt="{:,.0f}", xmax=150000)
    ax.set_xlabel("postings")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_title("Job database construction", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "data_funnel.png", dpi=200)


def fig_per_category():
    r = json.loads((RES / "ranking.json").read_text())
    pc = r["per_category_full"]
    names = list(pc)
    vals = [pc[n]["P@10"] for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    hbar(ax, [f"{n.title()} (n={pc[n]['n']})" for n in names], vals, [BLUE] * len(names), fmt="{:.0%}", xmax=1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Precision@10 — share of top-10 postings in a relevant function")
    ax.set_title("JobFit by resume category", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "per_category.png", dpi=200)


if __name__ == "__main__":
    fig_funnel()
    for f in (fig_ranking, fig_paraphrase, fig_per_category):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    print(sorted(p.name for p in FIG.glob("*.png")))
