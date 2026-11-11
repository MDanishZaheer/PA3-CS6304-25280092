# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Candidate quality, independent selection, and answer-weighted consensus."""

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

VISIBLE_FIELDS = ("rollout_id", "answer", "group", "token_count", "logp", "kl_u_p",
                  "image_attention", "prompt_attention", "generated_attention")


def revealed_only(candidates):
    """Remove references and correctness labels before a selection method sees candidates."""
    return [{key: item[key] for key in VISIBLE_FIELDS if key in item} for item in candidates]


def candidate_scores(candidates, vision):
    """Calibrate VisConf on only the currently revealed candidate prefix."""
    token_kl = [np.asarray(item["kl_u_p"], dtype=np.float64) for item in candidates]
    if vision:
        nonempty = [values for values in token_kl if len(values)]
        if not nonempty:
            raise ValueError("All candidates have zero measured tokens.")
        pooled = np.concatenate(nonempty)
        low, high = np.percentile(pooled, [5, 95], method="linear")
        denominator = max(float(high - low), 1e-8)
    scores = {name: [] for name in ("logp", "certainty", "vision", "visconf", "tokenwise")}
    for item, kl in zip(candidates, token_kl):
        logp = np.asarray(item["logp"], dtype=np.float64)
        if len(logp) != len(kl):
            raise ValueError("logp and kl_u_p token arrays are not aligned.")
        scores["logp"].append(float(np.mean(logp)) if len(logp) else -np.inf)
        scores["certainty"].append(float(np.mean(kl)) if len(kl) else -np.inf)
        if vision:
            attention = [np.asarray(item[key], dtype=np.float64) for key in
                         ("image_attention", "prompt_attention", "generated_attention")]
            if any(len(values) != len(kl) for values in attention):
                raise ValueError("Visual attention arrays do not match the generated tokens.")
            total = sum(attention)
            visual = np.divide(attention[0], total, out=np.zeros_like(total), where=total > 0)
            confidence = np.clip((kl - low) / denominator, 0.0, 1.0)
            scores["vision"].append(float(np.mean(visual)) if len(visual) else -np.inf)
            scores["visconf"].append(float(np.mean(visual) * np.mean(confidence)) if len(kl) else -np.inf)
            scores["tokenwise"].append(float(np.mean(visual * confidence)) if len(kl) else -np.inf)
    return scores


def earliest_group_winner(groups, weights):
    votes = defaultdict(float)
    for group, weight in zip(groups, weights):
        votes[group] += float(weight)
    highest = max(votes.values())
    winner = next(group for group in groups if np.isclose(votes[group], highest, rtol=0, atol=1e-12))
    index = groups.index(winner)
    return index, votes


def select(candidates, method, scores=None):
    """Select a candidate; all ties favor its first supporting rollout in this order."""
    if len(candidates) == 1 or method == "first":
        return 0, None
    if method == "shortest":
        lengths = [item["token_count"] for item in candidates]
        return int(np.argmin(lengths)), None
    groups = [item["group"] for item in candidates]
    if method == "self_consistency":
        return earliest_group_winner(groups, [1.0] * len(groups))
    score_name, aggregation = method.rsplit("_", 1)
    values = np.asarray(scores[score_name], dtype=float)
    values = np.nan_to_num(values, nan=-np.inf)
    if aggregation == "best":
        return int(np.argmax(values)), None
    if aggregation != "rwc":
        raise ValueError(f"Unknown selection method {method}.")
    ranks = rankdata(-values, method="average")
    weights = (len(candidates) - ranks) / (len(candidates) - 1)
    return earliest_group_winner(groups, weights)


def methods_for(dataset_name, include_ablations=False):
    methods = ["first", "self_consistency", "logp_best", "certainty_best", "certainty_rwc"]
    if dataset_name == "mathvista":
        methods.extend(["vision_best", "vision_rwc", "visconf_best", "visconf_rwc"])
        if include_ablations:
            methods.extend(["tokenwise_best", "tokenwise_rwc"])
    if include_ablations:
        methods.append("shortest")
    return methods


def prefix_summary(candidates, vision, include_ablations=False):
    visible = revealed_only(candidates)
    scores = candidate_scores(visible, vision)
    methods = methods_for("mathvista" if vision else "math", include_ablations)
    total_tokens = int(sum(item["token_count"] for item in candidates))
    oracle = any(item["correct"] for item in candidates)
    distinct = len(set(item["group"] for item in candidates))
    rows = []
    for method in methods:
        index, votes = select(visible, method, scores)
        selected = candidates[index]
        rows.append({"method": method, "correct": bool(selected["correct"]),
                     "oracle": bool(oracle), "distinct_answers": distinct,
                     "selected_rollout_id": selected["rollout_id"],
                     "selected_answer": selected["answer"], "selected_group": selected["group"],
                     "selected_index": index, "token_cost": int(selected["token_count"]) if method == "first" else total_tokens,
                     "votes": dict(votes) if votes is not None else None})
    return rows, scores
