# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Within-question and within-rollout signal diagnostics at N=32."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.shared import save_frame, save_json
from methods.selection import candidate_scores


def _correlation(left, right):
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def analyze_signals(labeled, dataset_name, results_root, evaluation_ids, trajectory_ids=()):
    destination = Path(results_root) / "metrics" / f"{dataset_name}_signals.parquet"
    if destination.is_file() and (dataset_name != "mathvista" or
                                  (Path(results_root) / "metrics" / "mathvista_fusion.parquet").is_file()):
        return pd.read_parquet(destination)
    vision = dataset_name == "mathvista"
    selected = labeled.loc[labeled["sample_id"].isin(evaluation_ids)]
    if selected["sample_id"].nunique() != len(evaluation_ids):
        raise ValueError(f"{dataset_name}: signal diagnostics require all held-out questions.")
    rows = []
    fusion_rows = []
    trajectories = []
    for question_id, subset in selected.groupby("sample_id", sort=False):
        candidates = subset.sort_values("rollout_id").to_dict("records")
        scores = candidate_scores(candidates, vision)
        correct = np.asarray([item["correct"] for item in candidates], dtype=float)
        lengths = np.asarray([item["token_count"] for item in candidates], dtype=float)
        for score_name in (["length", "logp", "certainty"] +
                           (["vision", "visconf", "tokenwise"] if vision else [])):
            values = lengths if score_name == "length" else np.asarray(scores[score_name], dtype=float)
            order = np.argsort(values, kind="stable")
            rows.append({"sample_id": question_id, "score": score_name,
                         "correctness_correlation": _correlation(values, correct),
                         "length_correlation": _correlation(values, lengths),
                         "top_quartile_accuracy": float(np.mean(correct[order[-8:]])),
                         "bottom_quartile_accuracy": float(np.mean(correct[order[:8]])),
                         "top_bottom_difference": float(np.mean(correct[order[-8:]]) - np.mean(correct[order[:8]]))})
        if vision:
            token_correlations = []
            pooled = np.concatenate([np.asarray(item["kl_u_p"], dtype=float) for item in candidates
                                     if len(item["kl_u_p"])])
            low, high = np.percentile(pooled, [5, 95], method="linear")
            for candidate_index, (item, visconf, tokenwise) in enumerate(zip(
                    candidates, scores["visconf"], scores["tokenwise"])):
                kl = np.asarray(item["kl_u_p"], dtype=float)
                visual_parts = [np.asarray(item[key], dtype=float) for key in
                                ("image_attention", "prompt_attention", "generated_attention")]
                denominator = sum(visual_parts)
                visual = np.divide(visual_parts[0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
                confidence = np.clip((kl - low) / max(float(high - low), 1e-8), 0, 1)
                correlation = (float(np.corrcoef(confidence, visual)[0, 1])
                               if len(kl) > 2 and np.std(confidence) > 0 and np.std(visual) > 0 else float("nan"))
                token_correlations.append(correlation)
                fusion_rows.append({"sample_id": question_id, "rollout_id": item["rollout_id"],
                                    "token_correlation": correlation,
                                    "fusion_covariance": float(tokenwise - visconf),
                                    "visconf": float(visconf), "tokenwise": float(tokenwise),
                                    "correct": bool(item["correct"])})
                if question_id in trajectory_ids and candidate_index == 0:
                    trajectories.append({"sample_id": question_id, "rollout_id": item["rollout_id"],
                                         "confidence": confidence.tolist(), "visual_engagement": visual.tolist(),
                                         "rule": "First rollout ID in a preselected evaluation question"})
            rows.append({"sample_id": question_id, "score": "confidence_vision_token_correlation",
                         "correctness_correlation": float(np.nanmean(token_correlations)) if np.isfinite(token_correlations).any() else float("nan"),
                         "length_correlation": float("nan"), "top_quartile_accuracy": float("nan"),
                         "bottom_quartile_accuracy": float("nan"), "top_bottom_difference": float("nan")})
    result = pd.DataFrame(rows)
    save_frame(result, destination)
    if vision:
        save_frame(pd.DataFrame(fusion_rows), Path(results_root) / "metrics" / "mathvista_fusion.parquet")
        save_json(Path(results_root) / "metrics" / "mathvista_trajectories.json", trajectories)
    return result
