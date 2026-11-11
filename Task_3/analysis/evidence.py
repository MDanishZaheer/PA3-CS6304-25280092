# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Rebuild RQ1-RQ5 tables and figures strictly from saved Kaggle outputs."""

from pathlib import Path

import pandas as pd

from analysis import figures
from common.shared import read_json, require_result, save_json


def _csv(results_root, filename):
    return pd.read_csv(require_result(Path(results_root) / "tables" / filename))


def _figure(path, draw, regenerate):
    return draw() if regenerate else require_result(path)


def rq1(results_root, regenerate_figures=True):
    fixed = {name: _csv(results_root, f"{name}_fixed.csv") for name in ("math", "mathvista")}
    intervals = {name: _csv(results_root, f"{name}_paired_intervals.csv")
                 for name in ("math", "mathvista")}
    sampling_path = Path(results_root) / "figures" / "rq1_sampling.png"
    failures_path = Path(results_root) / "figures" / "rq1_failures.png"
    paths = [_figure(sampling_path, lambda: figures.sampling_curves(fixed, sampling_path), regenerate_figures),
             _figure(failures_path, lambda: figures.failure_decomposition(fixed, failures_path), regenerate_figures)]
    compact = {name: frame.loc[frame["method"].isin(["first", "self_consistency",
                     "certainty_rwc" if name == "math" else "visconf_rwc"])]
               for name, frame in fixed.items()}
    methods_at_32 = {name: frame.loc[frame["budget"] == 32].sort_values("correct", ascending=False)
                     for name, frame in fixed.items()}
    export_quality = {name: read_json(require_result(Path(results_root) / "metrics" /
                       f"{name}_export_quality.json")) for name in ("math", "mathvista")}
    return {"tables": compact, "methods_at_32": methods_at_32, "paired_intervals": intervals,
            "export_quality": export_quality, "figures": paths}


def rq2(results_root, regenerate_figures=True):
    signals = {name: _csv(results_root, f"{name}_signals.csv") for name in ("math", "mathvista")}
    fixed = _csv(results_root, "mathvista_fixed.csv")
    fusion = fixed.loc[(fixed["budget"] == 32) &
                       fixed["method"].isin(["vision_best", "vision_rwc", "visconf_best",
                                               "visconf_rwc", "tokenwise_best", "tokenwise_rwc"])]
    fusion_tokens = pd.read_parquet(require_result(Path(results_root) / "metrics" / "mathvista_fusion.parquet"))
    trajectory = read_json(require_result(Path(results_root) / "metrics" / "mathvista_trajectories.json"))
    signals_path = Path(results_root) / "figures" / "rq2_signals.png"
    trajectories_path = Path(results_root) / "figures" / "rq2_trajectories.png"
    paths = [_figure(signals_path, lambda: figures.signal_associations(signals, signals_path), regenerate_figures),
             _figure(trajectories_path, lambda: figures.visual_trajectories(trajectory, trajectories_path), regenerate_figures)]
    fusion_summary = fusion_tokens[["token_correlation", "fusion_covariance"]].mean().to_frame("mean")
    return {"signal_table": signals, "fusion_table": fusion,
            "fusion_covariance": fusion_summary, "figures": paths}


def rq3(results_root, regenerate_figures=True):
    live = _csv(results_root, "live_strategies.csv")
    if not (len(live) == 3 and live["questions"].eq(50).all()):
        raise ValueError("Saved live table does not contain all 50 questions and three strategies.")
    transitions = read_json(require_result(Path(results_root) / "metrics" / "revision_transitions.json"))
    paired = _csv(results_root, "live_paired_intervals.csv")
    figure_path = Path(results_root) / "figures" / "rq3_live.png"
    path = _figure(figure_path, lambda: figures.live_comparison(live, figure_path), regenerate_figures)
    return {"live_table": live, "paired_intervals": paired,
            "revision_transitions": transitions, "figures": [path]}


def rq4(results_root, regenerate_figures=True):
    adaptive = {name: _csv(results_root, f"{name}_adaptive.csv") for name in ("math", "mathvista")}
    stops = {name: _csv(results_root, f"{name}_stops.csv") for name in ("math", "mathvista")}
    context = {name: _csv(results_root, f"{name}_stopping_context.csv")
               for name in ("math", "mathvista")}
    thresholds = {name: read_json(require_result(Path(results_root) / "metrics" /
                         f"{name}_chosen_thresholds.json")) for name in ("math", "mathvista")}
    paired = {name: _csv(results_root, f"{name}_adaptive_paired_intervals.csv")
              for name in ("math", "mathvista")}
    figure_path = Path(results_root) / "figures" / "rq4_adaptive.png"
    path = _figure(figure_path, lambda: figures.adaptive_comparison(adaptive, stops, figure_path), regenerate_figures)
    return {"thresholds": thresholds, "adaptive_table": adaptive,
            "paired_intervals": paired, "stopping_distributions": stops,
            "stopping_context": context, "figures": [path]}


def rq5(results_root, regenerate_figures=True):
    fixed = {name: _csv(results_root, f"{name}_fixed.csv") for name in ("math", "mathvista")}
    adaptive = {name: _csv(results_root, f"{name}_adaptive.csv") for name in ("math", "mathvista")}
    examples = read_json(require_result(Path(results_root) / "metrics" / "qualitative_examples.json"))
    figure_path = Path(results_root) / "figures" / "rq5_cost.png"
    path = _figure(figure_path, lambda: figures.cost_frontier(fixed, adaptive, figure_path), regenerate_figures)
    example_rows = []
    for name, example in examples.items():
        if "selections" in example:
            for selected in example["selections"]:
                example_rows.append({"case": name, "sample_id": example["sample_id"],
                                     "question": example["question"], "gold": example["gold"],
                                     "method": selected["method"], "answer": selected["selected_answer"],
                                     "correct": selected["correct"], "response_excerpt": str(selected["response"])[:240]})
        elif "early_answer" in example:
            for method, answer, response in (("early stop", example["early_answer"], example["early_response"]),
                                             ("fixed 32", example["fixed32_answer"], example["fixed32_response"])):
                example_rows.append({"case": name, "sample_id": example["sample_id"],
                                     "question": example["question"], "gold": example["gold"],
                                     "method": method, "answer": answer, "response_excerpt": str(response)[:240]})
        else:
            for method, answer, response in (("initial", example["initial_answer"], example["initial_response"]),
                                             ("revision", example["revised_answer"], example["revision_response"])):
                example_rows.append({"case": name, "sample_id": example["sample_id"],
                                     "question": example["question"], "gold": example["gold"],
                                     "method": method, "answer": answer, "response_excerpt": str(response)[:240]})
    return {"cost_tables": adaptive, "examples": examples,
            "example_table": pd.DataFrame(example_rows), "figures": [path]}


def select_examples(labeled_by_dataset, fixed_by_dataset, adaptive_by_dataset, results_root):
    """Choose first eligible held-out example by sample ID, never by hand."""
    examples = {}
    for dataset in ("math", "mathvista"):
        labeled = labeled_by_dataset[dataset]
        def response_for(sample_id, rollout_id):
            match = labeled.loc[labeled["sample_id"].eq(sample_id) &
                                labeled["rollout_id"].eq(str(rollout_id))]
            if len(match) != 1:
                raise ValueError(f"Missing candidate response for {dataset}/{sample_id}/{rollout_id}.")
            return match.iloc[0]["response"]

        fixed = fixed_by_dataset[dataset]
        quality = "certainty_rwc" if dataset == "math" else "visconf_rwc"
        subset = fixed.loc[(fixed["split"] == "evaluation") & (fixed["order"] == 0) &
                           (fixed["budget"] == 32) & fixed["method"].isin(["self_consistency", quality])]
        wide = subset.pivot(index="sample_id", columns="method", values="correct")
        differing = sorted(wide.index[wide["self_consistency"].ne(wide[quality])])
        if differing:
            sample_id = differing[0]
            records = subset.loc[subset["sample_id"] == sample_id].to_dict("records")
            for record in records:
                record["response"] = response_for(sample_id, record["selected_rollout_id"])
            question = labeled_by_dataset[dataset].loc[
                labeled_by_dataset[dataset]["sample_id"] == sample_id].iloc[0]
            examples[f"{dataset}_selection_disagreement"] = {
                "rule": "Lexicographically first held-out question, order 0, N=32, with a correctness disagreement",
                "sample_id": sample_id, "question": question["question"], "gold": question["gold"],
                "selections": records}
        adaptive = adaptive_by_dataset[dataset]
        fixed32 = adaptive.loc[adaptive["policy"] == "always32", ["sample_id", "order", "correct",
                                                                     "selected_answer", "selected_rollout_id"]]
        premature = adaptive.loc[(adaptive["policy"] == "quality") & adaptive["count"].lt(32)]
        premature = premature.merge(fixed32, on=["sample_id", "order"], suffixes=("", "_fixed32"))
        premature = premature.loc[~premature["correct"] & premature["correct_fixed32"]]
        if len(premature):
            row = premature.sort_values(["sample_id", "order"]).iloc[0]
            question = labeled_by_dataset[dataset].loc[
                labeled_by_dataset[dataset]["sample_id"] == row["sample_id"]].iloc[0]
            examples[f"{dataset}_premature_stop"] = {
                "rule": "Lexicographically first held-out quality-aware early error recovered at fixed 32",
                "sample_id": row["sample_id"], "order": int(row["order"]),
                "question": question["question"], "gold": question["gold"],
                "early_answer": row["selected_answer"], "fixed32_answer": row["selected_answer_fixed32"],
                "early_response": response_for(row["sample_id"], row["selected_rollout_id"]),
                "fixed32_response": response_for(row["sample_id"], row["selected_rollout_id_fixed32"])}
    from models.live import _file_stem
    from data.answers import correctness
    live_path = Path(results_root) / "predictions" / "live_strategies.parquet"
    if live_path.is_file():
        live = pd.read_parquet(live_path)
        revised = live.loc[live["strategy"] == "refinement"].sort_values("sample_id")
        changed_answer = None
        for row in revised.itertuples():
            stem = _file_stem(row.sample_id)
            initial = read_json(Path(results_root) / "progress" / "live_calls" / f"{stem}_parallel_0.json")
            revision = read_json(Path(results_root) / "progress" / "live_calls" / f"{stem}_revision.json")
            if changed_answer is None and initial["extracted_answer"] != revision["extracted_answer"]:
                changed_answer = {"rule": "First sample ID whose extracted answer changed in revision",
                    "sample_id": row.sample_id, "question": initial["question"], "gold": row.gold,
                    "initial_answer": initial["extracted_answer"],
                    "revised_answer": revision["extracted_answer"],
                    "initial_response": initial["response"], "revision_response": revision["response"]}
            correct_initial = correctness([initial["extracted_answer"]], row.gold, "math", {})[0]
            if correct_initial != row.correct:
                examples["revision_transition"] = {"rule": "First sample ID with a correctness transition",
                    "sample_id": row.sample_id, "question": initial["question"], "gold": row.gold,
                    "initial_answer": initial["extracted_answer"], "revised_answer": revision["extracted_answer"],
                    "initial_response": initial["response"], "revision_response": revision["response"]}
                break
        if "revision_transition" not in examples and changed_answer is not None:
            examples["revision_changed_answer"] = changed_answer
    save_json(Path(results_root) / "metrics" / "qualitative_examples.json", examples)
    return examples
