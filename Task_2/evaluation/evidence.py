# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Build saved Task 2 tables, NeurIPS-sized PNG figures, and aligned examples."""

import json
from collections import defaultdict
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

from common.shared import read_json, write_json
from data.download import download_image


def _figure_style(paths):
    font_path = paths["root"] / "Task_2" / "assets" / "times.ttf"
    if not font_path.is_file():
        raise FileNotFoundError(f"Upload the Times New Roman font with Task 2: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.labelsize": 10,
                         "axes.titlesize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "figure.dpi": 120, "savefig.dpi": 300})


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _development_table(paths):
    rows = []
    for path in sorted((paths["results"] / "metrics").glob("development_*.json")):
        item = read_json(path)
        rows.append({"update": item["update"], "suite": "caption", "caption_reference_loss": item["caption_reference_loss"]})
        for suite, scores in item["suites"].items():
            rows.append({"update": item["update"], "suite": suite, **scores})
    if not rows:
        raise RuntimeError("No Task 2 development evaluations are saved.")
    table = pd.DataFrame(rows).sort_values(["update", "suite"])
    table.to_csv(paths["results"] / "metrics" / "development_table.csv", index=False)
    return table


def _final_tables(paths):
    summary = read_json(paths["results"] / "metrics" / "final_evaluation.json")
    rows, types, advantages = [], [], []
    for suite, suite_data in summary["suites"].items():
        for condition, scores in suite_data["conditions"].items():
            if condition.startswith("correct_advantage_"):
                advantages.append({"suite": suite, "comparison": condition,
                                   "mean": scores["mean"], "ci95_low": scores["ci95"][0],
                                   "ci95_high": scores["ci95"][1]})
                continue
            rows.append({"suite": suite, "condition": condition,
                         **{name: values["mean"] for name, values in scores.items()},
                         "accuracy_ci_low": scores["accuracy"]["ci95"][0],
                         "accuracy_ci_high": scores["accuracy"]["ci95"][1]})
        for condition, breakdown in suite_data["question_type_metrics"].items():
            types.extend({"suite": suite, "condition": condition, "question_type": kind, **values}
                         for kind, values in breakdown.items())
    final = pd.DataFrame(rows)
    type_table = pd.DataFrame(types)
    final.to_csv(paths["results"] / "metrics" / "final_table.csv", index=False)
    type_table.to_csv(paths["results"] / "metrics" / "question_type_table.csv", index=False)
    pd.DataFrame(advantages).to_csv(paths["results"] / "metrics" / "correct_image_advantages.csv", index=False)
    return summary, final, type_table


def _resources(paths):
    rows = []
    feature_path = paths["results"] / "metrics" / "feature_cache.json"
    if feature_path.is_file():
        item = read_json(feature_path)
        rows.append({"stage": "feature_cache", "seconds": item.get("seconds_all_sessions", item.get("seconds_this_session")),
                     "peak_gpu_gib": item.get("peak_gpu_gib_all_sessions", item.get("peak_gpu_gib_this_session")),
                     "trainable_parameters": 0, "stage_export_mib": None})
    for stage in ("caption", "instruction", "natural"):
        path = paths["results"] / "metrics" / f"{stage}_resources.json"
        if path.is_file():
            item = read_json(path)
            stage_export = paths["results"] / "checkpoints" / f"{stage}_weights.pt"
            rows.append({"stage": stage, "seconds": item.get("seconds_all_sessions", item.get("seconds_this_session")),
                         "peak_gpu_gib": item.get("peak_gpu_gib_all_sessions", item.get("peak_gpu_gib_this_session")),
                         "trainable_parameters": item["trainable_parameters"],
                         "stage_export_mib": item.get("stage_export_mib", stage_export.stat().st_size / 2 ** 20
                                                      if stage_export.is_file() else None)})
    table = pd.DataFrame(rows)
    table.to_csv(paths["results"] / "metrics" / "resources_table.csv", index=False)
    return table


def _plot_learning(development, paths):
    history = pd.read_csv(paths["results"] / "histories" / "training.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.2), constrained_layout=True)
    train_axis, caption_axis, answer_axis, swap_axis = axes.flat
    train_axis.plot(history["update"], history["sft_loss"], linewidth=0.8, alpha=0.5, label="SFT loss")
    train_axis.plot(history["update"], history["sft_loss"].rolling(50, min_periods=1).mean(),
                 linewidth=1.3, label="50-update mean")
    train_axis.set(ylabel="Example-weighted SFT loss", xlabel="Optimizer update")
    train_axis.legend(frameon=False)
    captions = development[development["suite"] == "caption"]
    caption_axis.plot(captions["update"], captions["caption_reference_loss"], marker="s")
    caption_axis.set(ylabel="Caption reference loss", xlabel="Optimizer update")
    for suite, label in (("grounding_dev", "Grounding"), ("natural_dev", "Natural VQA")):
        part = development[development["suite"] == suite]
        answer_axis.plot(part["update"], part["correct_nll"], marker="o", label=label)
        swap_axis.plot(part["update"], part["image_swap_penalty"], marker="o", label=label)
    answer_axis.set(ylabel="Answer reference NLL", xlabel="Optimizer update")
    swap_axis.set(ylabel="Swap minus correct NLL", xlabel="Optimizer update")
    for axis in axes.flat:
        for boundary in (2250, 3750):
            axis.axvline(boundary, color="0.5", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    answer_axis.legend(frameon=False)
    swap_axis.legend(frameon=False)
    _save(fig, paths["results"] / "figures" / "rq1_curriculum.png")


def _plot_controls(final, paths):
    conditions = ("correct", "complementary", "mismatched", "no_image")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.5), sharey=True, constrained_layout=True)
    for axis, suite in zip(axes, ("grounding_test", "natural_test")):
        values = [float(final[(final.suite == suite) & (final.condition == condition)]["accuracy"].iloc[0])
                  for condition in conditions]
        axis.bar(range(4), values, color=["#315f93", "#ac7145", "#888888", "#b0ad9b"])
        axis.set_xticks(range(4), ["Correct", "Partner", "Mismatch", "No image"], rotation=35)
        axis.set(title=suite.replace("_test", "").replace("_", " ").title(), ylabel="Accuracy")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    _save(fig, paths["results"] / "figures" / "rq2_image_controls.png")


def _plot_pair_diagnostics(final, paths):
    conditions = ("correct", "complementary", "mismatched", "no_image")
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.5), sharey=True, constrained_layout=True)
    for axis, suite in zip(axes, ("grounding_test", "natural_test")):
        part = final.set_index(["suite", "condition"])
        pair_accuracy = [part.loc[(suite, condition), "pair_accuracy"] for condition in conditions]
        identical = [part.loc[(suite, condition), "identical_answer_rate"] for condition in conditions]
        x = range(4)
        axis.plot(x, pair_accuracy, marker="o", label="Both answers correct")
        axis.plot(x, identical, marker="s", label="Identical answers")
        axis.set_xticks(list(x), ["Correct", "Partner", "Mismatch", "No image"], rotation=35)
        axis.set(title=suite.replace("_test", "").replace("_", " ").title(), ylabel="Pair fraction")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    _save(fig, paths["results"] / "figures" / "rq3_pair_diagnostics.png")


def _aligned_examples(paths, limit=3):
    saved_path = paths["results"] / "examples" / "qualitative.json"
    if saved_path.is_file():
        saved = read_json(saved_path)
        if all((paths["results"] / "figures" / f"qualitative_{suite}_{item['pair_id']}.png").is_file()
               for suite, items in saved.items() for item in items):
            return saved
    all_examples = {}
    for suite in ("grounding_test", "natural_test"):
        path = paths["results"] / "predictions" / f"{suite}.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        groups = defaultdict(lambda: defaultdict(list))
        for record in records:
            groups[record["pair_id"]][record["condition"]].append(record)
        chosen = []
        for pair_id in sorted(groups):
            conditions = groups[pair_id]
            correct = sorted(conditions["correct"], key=lambda row: row["position"])
            swapped = sorted(conditions["complementary"], key=lambda row: row["position"])
            if len(correct) != 2 or len(swapped) != 2:
                continue
            useful = (correct[0]["canonical_correct"] == correct[1]["canonical_correct"] == 1.0
                      and any(a["normalized_prediction"] != b["normalized_prediction"]
                              for a, b in zip(correct, swapped)))
            chosen.append((0 if useful else 1, pair_id, correct, swapped))
        chosen.sort(key=lambda value: (value[0], value[1]))
        selected = []
        for priority, pair_id, correct, swapped in chosen[:limit]:
            entry = {"suite": suite, "pair_id": pair_id,
                     "selection_rule": "first correct-both pair with image-sensitive outputs, then pair ID fallback"
                     if priority == 0 else "pair ID fallback", "correct": correct, "swapped": swapped}
            selected.append(entry)
            fig, axes = plt.subplots(1, 2, figsize=(6.7, 5.2))
            for index, axis in enumerate(axes):
                row, altered = correct[index], swapped[index]
                image_path = download_image(row["example"]["image_key"], paths["data"])
                with Image.open(image_path) as image:
                    axis.imshow(image.convert("RGB"))
                axis.axis("off")
                axis.set_title(f"Image {index + 1}: {row['example']['image_key']}")
                question = row["example"]["question"]
                text = (f"Q: {fill(question, width=38)}\n"
                        f"Reference: {fill(row['example']['target'], width=38)}\n"
                        f"Correct image: {fill(row['prediction'], width=38)}\n"
                        f"Partner image: {fill(altered['prediction'], width=38)}")
                axis.text(0, -0.07, text, transform=axis.transAxes, va="top", wrap=True)
            _save(fig, paths["results"] / "figures" / f"qualitative_{suite}_{pair_id}.png")
        all_examples[suite] = selected
    write_json(paths["results"] / "examples" / "qualitative.json", all_examples)
    return all_examples


def build_evidence(paths):
    """Regenerate every report table and PNG from saved training/evaluation files."""
    _figure_style(paths)
    development = _development_table(paths)
    summary, final, question_types = _final_tables(paths)
    resources = _resources(paths)
    _plot_learning(development, paths)
    _plot_controls(final, paths)
    _plot_pair_diagnostics(final, paths)
    examples = _aligned_examples(paths)
    return {"development": development, "final": final, "types": question_types,
            "resources": resources, "summary": summary, "examples": examples}
