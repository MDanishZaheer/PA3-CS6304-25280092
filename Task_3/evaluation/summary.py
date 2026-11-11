# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Question-level aggregation, paired intervals, and compact report tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.shared import save_frame, save_json


def _paired_interval(values, seed=6304, resamples=1000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indexes].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_fixed(frame, config, dataset_name, results_root):
    """Average orders within each question before averaging questions."""
    heldout = frame.loc[frame["split"] == "evaluation"].copy()
    heldout["no_correct_candidate"] = ~heldout["oracle"]
    heldout["selection_miss"] = heldout["oracle"] & ~heldout["correct"]
    by_question = heldout.groupby(["sample_id", "budget", "method"], as_index=False)[
        ["correct", "oracle", "distinct_answers", "token_cost", "no_correct_candidate", "selection_miss"]].mean()
    summary = by_question.groupby(["budget", "method"], as_index=False)[
        ["correct", "oracle", "distinct_answers", "token_cost", "no_correct_candidate", "selection_miss"]].mean()
    summary.insert(0, "dataset_name", dataset_name)
    summary["questions"] = by_question["sample_id"].nunique()
    save_frame(summary, Path(results_root) / "metrics" / f"{dataset_name}_fixed_summary.parquet")
    destination = Path(results_root) / "tables" / f"{dataset_name}_fixed.csv"
    summary.to_csv(destination, index=False)
    comparisons = []
    for budget in config["experiment"]["budgets"]:
        subset = by_question.loc[by_question["budget"] == budget]
        pivot = subset.pivot(index="sample_id", columns="method", values="correct")
        for method in pivot.columns:
            if method == "first":
                continue
            for reference in ("first", "self_consistency"):
                if method == reference:
                    continue
                delta = pivot[method].to_numpy() - pivot[reference].to_numpy()
                lower, upper = _paired_interval(delta, resamples=config["experiment"]["bootstrap_resamples"])
                comparisons.append({"dataset_name": dataset_name, "budget": budget,
                                    "method": method, "reference": reference,
                                    "difference_pp": 100 * float(np.mean(delta)),
                                    "ci_low_pp": 100 * lower, "ci_high_pp": 100 * upper})
    intervals = pd.DataFrame(comparisons)
    intervals.to_csv(Path(results_root) / "tables" / f"{dataset_name}_paired_intervals.csv", index=False)
    return summary, intervals


def summarize_signals(frame, dataset_name, results_root):
    summary = frame.groupby("score", as_index=False)[
        ["correctness_correlation", "length_correlation", "top_quartile_accuracy",
         "bottom_quartile_accuracy", "top_bottom_difference"]].mean()
    summary.insert(0, "dataset_name", dataset_name)
    summary.to_csv(Path(results_root) / "tables" / f"{dataset_name}_signals.csv", index=False)
    return summary


def summarize_live(frame, results_root):
    if frame["sample_id"].nunique() != 50 or len(frame) != 150:
        raise ValueError("Live evidence needs all three strategies on the fixed 50 questions.")
    summary = frame.groupby("strategy", as_index=False).agg(
        accuracy=("correct", "mean"), input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"), runtime_seconds=("runtime_seconds", "mean"),
        extraction_rate=("extracted", "mean"), truncation_rate=("truncated_calls", lambda x: (x > 0).mean()),
        peak_gpu_gb=("peak_gpu_gb", "max"))
    summary["questions"] = 50
    summary.to_csv(Path(results_root) / "tables" / "live_strategies.csv", index=False)
    pivot = frame.pivot(index="sample_id", columns="strategy", values="correct")
    comparisons = []
    for method, reference in (("parallel", "single"), ("refinement", "single"),
                              ("refinement", "parallel")):
        delta = pivot[method].to_numpy(dtype=float) - pivot[reference].to_numpy(dtype=float)
        lower, upper = _paired_interval(delta)
        comparisons.append({"method": method, "reference": reference,
                            "difference_pp": 100 * float(np.mean(delta)),
                            "ci_low_pp": 100 * lower, "ci_high_pp": 100 * upper})
    pd.DataFrame(comparisons).to_csv(Path(results_root) / "tables" /
                                     "live_paired_intervals.csv", index=False)
    # The requested revision transitions use the shared first parallel answer, not parallel consensus.
    from models.live import _file_stem
    from common.shared import read_json
    from data.answers import correctness
    initial = {}
    for question_id in pivot.index:
        path = Path(results_root) / "progress" / "live_calls" / f"{_file_stem(question_id)}_parallel_0.json"
        call = read_json(path)
        initial[question_id] = correctness([call["extracted_answer"]], call["gold"], "math", {})[0]
    initial = pd.Series(initial)
    transitions = {"wrong_to_correct": int((~initial & pivot["refinement"]).sum()),
                   "correct_to_wrong": int((initial & ~pivot["refinement"]).sum()),
                   "initial_correct": int(initial.sum()),
                   "revised_correct": int(pivot["refinement"].sum())}
    save_json(Path(results_root) / "metrics" / "revision_transitions.json", transitions)
    return summary, transitions


def summarize_adaptive(frame, thresholds, dataset_name, results_root):
    by_question = frame.groupby(["sample_id", "policy"], as_index=False)[
        ["correct", "token_cost", "count"]].mean()
    summary = by_question.groupby("policy", as_index=False)[["correct", "token_cost", "count"]].mean()
    summary.insert(0, "dataset_name", dataset_name)
    summary["questions"] = by_question["sample_id"].nunique()
    summary.to_csv(Path(results_root) / "tables" / f"{dataset_name}_adaptive.csv", index=False)
    pivot = by_question.pivot(index="sample_id", columns="policy", values="correct")
    comparisons = []
    for method in ("agreement", "quality"):
        delta = pivot[method].to_numpy() - pivot["always32"].to_numpy()
        lower, upper = _paired_interval(delta)
        comparisons.append({"dataset_name": dataset_name, "method": method,
                            "reference": "always32", "difference_pp": 100 * float(np.mean(delta)),
                            "ci_low_pp": 100 * lower, "ci_high_pp": 100 * upper})
    pd.DataFrame(comparisons).to_csv(Path(results_root) / "tables" /
                                     f"{dataset_name}_adaptive_paired_intervals.csv", index=False)
    distribution = frame.groupby(["policy", "count"], as_index=False).size()
    distribution["fraction"] = distribution["size"] / distribution.groupby("policy")["size"].transform("sum")
    distribution.insert(0, "dataset_name", dataset_name)
    distribution.to_csv(Path(results_root) / "tables" / f"{dataset_name}_stops.csv", index=False)
    fixed32 = frame.loc[frame["policy"] == "always32", ["sample_id", "order", "correct"]]
    for policy in ("agreement", "quality"):
        policy_frame = frame.loc[frame["policy"] == policy]
        joined = policy_frame.merge(fixed32, on=["sample_id", "order"], suffixes=("", "_fixed32"))
        failures = joined.loc[joined["count"].lt(32) & ~joined["correct"] & joined["correct_fixed32"]]
        failures.to_csv(Path(results_root) / "tables" / f"{dataset_name}_{policy}_premature_stops.csv", index=False)
    save_json(Path(results_root) / "metrics" / f"{dataset_name}_chosen_thresholds.json", thresholds)
    return summary, distribution


def summarize_stopping_context(frame, labeled, dataset_name, results_root):
    """Use full-pool correctness only after held-out policy outputs are fixed."""
    full_pool = labeled.groupby("sample_id", as_index=False)["correct"].mean().rename(
        columns={"correct": "full_pool_correct_fraction"})
    retrospective = frame.merge(full_pool, on="sample_id", validate="many_to_one")
    retrospective["full_pool_bin"] = pd.cut(
        retrospective["full_pool_correct_fraction"], [-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"])
    summary = retrospective.groupby(["policy", "count", "full_pool_bin"],
                                    observed=True, as_index=False).agg(
        questions=("sample_id", "nunique"), accuracy=("correct", "mean"),
        mean_agreement=("agreement", "mean"),
        mean_full_pool_correct_fraction=("full_pool_correct_fraction", "mean"))
    summary.insert(0, "dataset_name", dataset_name)
    summary.to_csv(Path(results_root) / "tables" / f"{dataset_name}_stopping_context.csv", index=False)
    return summary
