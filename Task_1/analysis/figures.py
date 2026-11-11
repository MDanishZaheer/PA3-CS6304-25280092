# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Create report-sized 300 dpi PNG plots with Times New Roman 10 pt text."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


LABELS = {"baseline": "Base", "full_ft_mid": "FT 625", "full_ft": "FT",
          "full_ft_l2sp": "L2-SP", "lora_r4": "LoRA 4", "lora_r16": "LoRA 16",
          "oplora_r16": "OPLoRA 16"}
COLORS = {"baseline": "#555555", "full_ft_mid": "#9c755f", "full_ft": "#1f77b4",
          "full_ft_l2sp": "#ff7f0e", "lora_r4": "#2ca02c", "lora_r16": "#17becf",
          "oplora_r16": "#9467bd"}
FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "times.ttf"


def _style():
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Upload the Task_1/assets/times.ttf font before making report figures: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    if family != "Times New Roman":
        raise RuntimeError(f"Expected Times New Roman font, found {family}.")
    plt.rcParams.update({"font.family": family, "font.size": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def specialization_retention(rows, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    for row in rows:
        condition = row["condition"]
        ax.scatter(row["gsm8k_em"] * 100, row["retention_mean"] * 100,
                   c=COLORS[condition], s=34, zorder=3, label=LABELS[condition])
    ax.set(xlabel="GSM8K exact match (%)", ylabel="Mean non-math accuracy (%)")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)


def benchmark_changes(rows, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=False)
    methods = [row for row in rows if row["condition"] != "baseline"]
    x = np.arange(len(methods))
    labels = [LABELS[row["condition"]] for row in methods]
    for ax, metric, label in ((axes[0], "gsm8k_delta_pp", "GSM8K change (pp)"),
                              (axes[1], "retention_delta_pp", "Non-math mean change (pp)")):
        values = [row[metric] for row in methods]
        ax.bar(x, values, color=[COLORS[row["condition"]] for row in methods], width=0.7)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel(label)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def loss_curves(histories, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    for method, rows in histories.items():
        ax.plot([int(row["update"]) for row in rows], [float(row["sft_loss"]) for row in rows],
                label=LABELS[method], color=COLORS[method], lw=1)
    ax.set(xlabel="Optimizer update", ylabel="Response-token training loss")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.2)
    return _save(fig, path)


def diagnostic_plot(rows, path):
    _style()
    methods = list(dict.fromkeys(row["condition"] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    x = np.arange(len(methods))
    for ax, metric, label in ((axes[0], "rho", "Protected overlap (rho)"),
                              (axes[1], "eta", "Relative displacement (eta)"),
                              (axes[2], "response_ratio", "Response ratio R(z)")):
        values = [np.mean([row[metric] for row in rows if row["condition"] == method])
                  for method in methods]
        ax.bar(x, values, color=[COLORS[method] for method in methods], width=0.7)
        ax.set_ylabel(label)
        ax.set_xticks(x, [LABELS[method] for method in methods], rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def cost_plot(rows, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    for row in rows:
        method = row["method"]
        ax.scatter(row["total_hours"], row["gsm8k_em"] * 100,
                   s=max(25, row["trainable_millions"] * 0.2), c=COLORS[method], label=LABELS[method])
    ax.set(xlabel="Training and preprocessing time (h)", ylabel="GSM8K exact match (%)")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return _save(fig, path)


def qualitative_plot(rows, path):
    _style()
    methods = list(dict.fromkeys(row["condition"] for row in rows))
    formats = ("plain", "json", "template")
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    x = np.arange(len(methods))
    width = 0.26
    for index, format_name in enumerate(formats):
        values = [next((row["format_compliance_rate"] for row in rows
                        if row["condition"] == method and row["format"] == format_name), np.nan)
                  for method in methods]
        ax.bar(x + (index - 1) * width, np.array(values) * 100, width, label=format_name)
    ax.set_xticks(x, [LABELS[method] for method in methods], rotation=35, ha="right")
    ax.set(ylabel="Format compliance (%)", ylim=(0, 105))
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)


def qualitative_lengths(rows, path):
    _style()
    methods = list(dict.fromkeys(row["condition"] for row in rows))
    formats = ("plain", "json", "template")
    fig, ax = plt.subplots(figsize=(6.9, 2.7))
    x = np.arange(len(methods))
    width = 0.26
    for index, format_name in enumerate(formats):
        values = [next(row["mean_response_tokens"] for row in rows
                       if row["condition"] == method and row["format"] == format_name)
                  for method in methods]
        ax.bar(x + (index - 1) * width, values, width, label=format_name)
    ax.set_xticks(x, [LABELS[method] for method in methods], rotation=35, ha="right")
    ax.set_ylabel("Mean generated tokens")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)
