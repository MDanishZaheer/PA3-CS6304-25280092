# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Development-only stopping-rule selection and held-out adaptive simulation."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.shared import read_json, save_frame, save_json
from data.answers import NO_ANSWER
from methods.selection import candidate_scores, revealed_only, select


def _state(candidates, dataset_name):
    visible = revealed_only(candidates)
    groups = [item["group"] for item in visible]
    sc_index, counts = select(visible, "self_consistency")
    quality = "visconf" if dataset_name == "mathvista" else "certainty"
    scores = candidate_scores(visible, dataset_name == "mathvista")
    rwc_index, votes = select(visible, f"{quality}_rwc", scores)
    leader = groups[sc_index]
    sorted_votes = sorted(votes.values(), reverse=True)
    margin = (sorted_votes[0] - (sorted_votes[1] if len(sorted_votes) > 1 else 0.0)) / sum(votes.values())
    return {"count": len(candidates), "correct": bool(candidates[sc_index]["correct"]),
            "selected_answer": candidates[sc_index]["answer"],
            "selected_rollout_id": candidates[sc_index]["rollout_id"],
            "leader": leader, "unique_leader": list(counts.values()).count(max(counts.values())) == 1,
            "agreement": counts[leader] / len(candidates),
            "rwc_agrees": groups[rwc_index] == leader, "rwc_margin": float(margin),
            "token_cost": int(sum(item["token_count"] for item in candidates)),
            "oracle": any(item["correct"] for item in candidates)}


def build_states(labeled, splits, config, dataset_name, results_root):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_adaptive_states.parquet"
    if destination.is_file():
        return pd.read_parquet(destination)
    indexed = {question_id: {row["rollout_id"]: row for row in subset.to_dict("records")}
               for question_id, subset in labeled.groupby("sample_id", sort=False)}
    rows = []
    for split_name in ("development", "evaluation"):
        for position, question_id in enumerate(splits[split_name], 1):
            pool = indexed[question_id]
            for order_number, order in enumerate(splits["orders"][question_id]):
                for count in (*config["adaptive"]["checkpoints"], 32):
                    state = _state([pool[key] for key in order[:count]], dataset_name)
                    rows.append({"dataset_name": dataset_name, "split": split_name,
                                 "sample_id": question_id, "order": order_number, **state})
            if position % 100 == 0:
                print(f"{dataset_name} adaptive states: {split_name} {position} questions")
    states = pd.DataFrame(rows)
    save_frame(states, destination)
    return states


def _simulate(states, family, tau=None, gamma=None):
    rows = []
    for (_, _, question_id, order), group in states.groupby(
            ["dataset_name", "split", "sample_id", "order"], sort=False):
        group = group.sort_values("count")
        if family == "always32":
            chosen = group.iloc[-1]
        else:
            eligible = group[group["count"].isin([4, 8, 16]) & group["unique_leader"] &
                             group["leader"].ne(NO_ANSWER) & group["agreement"].ge(tau)]
            if family == "quality":
                eligible = eligible[eligible["rwc_agrees"] & eligible["rwc_margin"].ge(gamma)]
            chosen = eligible.iloc[0] if len(eligible) else group.iloc[-1]
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def select_thresholds(states, config, dataset_name, results_root):
    destination = Path(results_root) / "metrics" / f"{dataset_name}_adaptive_thresholds.json"
    if destination.is_file():
        return read_json(destination)
    dev = states.loc[states["split"] == "development"]
    baseline = _simulate(dev, "always32")
    baseline_accuracy = float(baseline["correct"].mean())
    minimum_accuracy = baseline_accuracy - config["adaptive"]["allowed_dev_drop_pp"] / 100
    chosen = {"baseline_accuracy": baseline_accuracy, "minimum_accuracy": minimum_accuracy}
    for family in ("agreement", "quality"):
        options = [{"family": "always32", "tau": None, "gamma": None,
                    "accuracy": baseline_accuracy, "mean_tokens": float(baseline["token_cost"].mean())}]
        for tau in config["adaptive"]["agreement_thresholds"]:
            margins = [None] if family == "agreement" else config["adaptive"]["quality_margins"]
            for gamma in margins:
                result = _simulate(dev, family, tau, gamma)
                options.append({"family": family, "tau": tau, "gamma": gamma,
                                "accuracy": float(result["correct"].mean()),
                                "mean_tokens": float(result["token_cost"].mean())})
        feasible = [option for option in options if option["accuracy"] >= minimum_accuracy - 1e-12]
        selected = min(feasible, key=lambda option: (option["mean_tokens"], -option["accuracy"],
                                                       option["tau"] if option["tau"] is not None else np.inf,
                                                       option["gamma"] if option["gamma"] is not None else np.inf))
        chosen[family] = selected
    save_json(destination, chosen)
    return chosen


def evaluate_adaptive(states, thresholds, dataset_name, results_root):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_adaptive.parquet"
    if destination.is_file():
        return pd.read_parquet(destination)
    evaluations = states.loc[states["split"] == "evaluation"]
    pieces = []
    for label in ("always32", "agreement", "quality"):
        setting = {"family": "always32", "tau": None, "gamma": None} if label == "always32" else thresholds[label]
        result = _simulate(evaluations, **{key: setting[key] for key in ("family", "tau", "gamma")})
        result["policy"] = label
        pieces.append(result)
    final = pd.concat(pieces, ignore_index=True)
    save_frame(final, destination)
    return final
