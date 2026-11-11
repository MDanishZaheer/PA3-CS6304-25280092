# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compact 300 dpi PNG figures using Times New Roman at 10 pt."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


FONT = Path(__file__).resolve().parents[1] / "assets" / "times.ttf"
COLORS = {"first": "#555555", "self_consistency": "#1f77b4", "certainty_rwc": "#2ca02c",
          "visconf_rwc": "#2ca02c", "agreement": "#9467bd", "quality": "#d62728",
          "always32": "#555555", "single": "#1f77b4", "parallel": "#ff7f0e", "refinement": "#2ca02c"}


def _style():
    if not FONT.is_file():
        raise FileNotFoundError(f"Task 3 figure font is missing: {FONT}")
    font_manager.fontManager.addfont(str(FONT))
    name = font_manager.FontProperties(fname=str(FONT)).get_name()
    if name != "Times New Roman":
        raise ValueError(f"Expected Times New Roman; found {name}.")
    plt.rcParams.update({"font.family": name, "font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 10, "figure.titlesize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def sampling_curves(summaries, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    for ax, (dataset, frame) in zip(axes, summaries.items()):
        for method, label in (("first", "FirstAns"), ("self_consistency", "Self-consistency"),
                              ("certainty_rwc" if dataset == "math" else "visconf_rwc", "Quality + RWC")):
            rows = frame.loc[frame["method"] == method].sort_values("budget")
            ax.plot(rows["budget"], 100 * rows["correct"], marker="o", ms=3,
                    color=COLORS[method], label=label)
        oracle = frame.loc[frame["method"] == "self_consistency"].sort_values("budget")
        ax.plot(oracle["budget"], 100 * oracle["oracle"], marker="s", ms=3,
                color="black", linestyle="--", label="Oracle coverage")
        ax.set(xlabel="Candidate budget N", title=dataset.upper(), xscale="log")
        ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out accuracy / coverage (%)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)


def failure_decomposition(summaries, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    for ax, (dataset, frame) in zip(axes, summaries.items()):
        rows = frame.loc[frame["method"] == "self_consistency"].sort_values("budget")
        x = np.arange(len(rows))
        no_correct = 100 * rows["no_correct_candidate"].to_numpy()
        missed = 100 * rows["selection_miss"].to_numpy()
        ax.bar(x, no_correct, label="No correct candidate", color="#9c755f")
        ax.bar(x, missed, bottom=no_correct, label="Correct candidate missed", color="#d62728")
        ax.set(xticks=x, xticklabels=rows["budget"], xlabel="Candidate budget N", title=dataset.upper())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Held-out questions (%)")
    axes[0].set_ylim(0, max(100 * frame.loc[frame["method"] == "self_consistency",
                                        ["no_correct_candidate", "selection_miss"]].sum(axis=1).max()
                            for frame in summaries.values()) * 1.08)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)


def signal_associations(summaries, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    for ax, (dataset, frame) in zip(axes, summaries.items()):
        rows = frame.loc[frame["score"].ne("confidence_vision_token_correlation")]
        values = 100 * rows["top_bottom_difference"].to_numpy()
        ax.bar(np.arange(len(rows)), values, color="#1f77b4")
        ax.axhline(0, color="black", lw=0.7)
        ax.set(xticks=np.arange(len(rows)), xticklabels=rows["score"], title=dataset.upper())
        ax.tick_params(axis="x", labelrotation=50)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Top - bottom quartile accuracy (pp)")
    return _save(fig, path)


def visual_trajectories(trajectories, path):
    _style()
    if len(trajectories) != 2:
        raise ValueError("Expected two fixed MathVista evaluation trajectories.")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    for ax, record in zip(axes, trajectories):
        x = np.arange(len(record["confidence"]))
        ax.plot(x, record["confidence"], lw=0.8, label="Calibrated confidence")
        ax.plot(x, record["visual_engagement"], lw=0.8, label="Visual engagement")
        ax.set(xlabel="Generated token index", title=f"Question {record['sample_id']}", ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Token score")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)


def live_comparison(summary, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    fig.subplots_adjust(wspace=0.42, bottom=0.28)
    x = np.arange(len(summary))
    colors = [COLORS[name] for name in summary["strategy"]]
    axes[0].bar(x, 100 * summary["accuracy"], color=colors)
    axes[0].set(ylabel="Accuracy (%)", xticks=x, xticklabels=summary["strategy"])
    axes[1].bar(x, summary["output_tokens"], color=colors)
    axes[1].set(ylabel="Output tokens / question", xticks=x, xticklabels=summary["strategy"])
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.2)
    return _save(fig, path)


def adaptive_comparison(summaries, distributions, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5))
    fig.subplots_adjust(wspace=0.38)
    markers = {"agreement": "o", "quality": "^", "always32": "s"}
    for dataset, frame in summaries.items():
        color = "#1f77b4" if dataset == "math" else "#ff7f0e"
        for row in frame.itertuples():
            axes[0].scatter(row.token_cost, 100 * row.correct, marker=markers[row.policy],
                            color=color, s=42, zorder=3)
        subset = distributions[dataset].loc[distributions[dataset]["policy"] == "quality"]
        axes[1].plot(subset["count"], 100 * subset["fraction"], marker="o",
                     color=color, label=dataset.upper())
    axes[0].set(xlabel="Mean generated tokens / question", ylabel="Held-out accuracy (%)")
    axes[0].legend(handles=[Line2D([], [], linestyle="", marker=marker, color="black", label=label)
                            for label, marker in (("Agreement", "o"), ("Quality-aware", "^"),
                                                  ("Fixed 32", "s"))],
                   loc="center", frameon=False)
    axes[1].set(xlabel="Quality-aware stop count", ylabel="Questions (%)", xticks=[4, 8, 16, 32])
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    return _save(fig, path)


def cost_frontier(fixed, adaptive, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, dataset in zip(axes, ("math", "mathvista")):
        rows = fixed[dataset]
        for method in ("first", "self_consistency", "certainty_rwc" if dataset == "math" else "visconf_rwc"):
            subset = rows.loc[rows["method"] == method].sort_values("budget")
            ax.plot(subset["token_cost"], 100 * subset["correct"], marker="o", ms=3,
                    label="Quality + RWC" if method.endswith("_rwc") else method.replace("_", " "),
                    color=COLORS[method])
        for row in adaptive[dataset].itertuples():
            ax.scatter(row.token_cost, 100 * row.correct, marker="x", s=42,
                       color=COLORS[row.policy], label=row.policy)
        ax.set(xlabel="Mean generated tokens / question", title=dataset.upper())
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out accuracy (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)
