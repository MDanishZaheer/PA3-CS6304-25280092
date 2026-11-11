# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Evaluate nested fixed-budget prefixes of the supplied rollout pools."""

import json
from pathlib import Path

import pandas as pd

from common.shared import save_frame, save_json
from data.answers import (_math_equivalent, _parsed_math, answer_groups,
                          comparison_counts, correctness,
                          GROUP_COMPARISON_TIMEOUT, GOLD_COMPARISON_TIMEOUT,
                          RETRY_COMPARISON_TIMEOUT)
from methods.selection import prefix_summary


def label_candidates(frame, dataset_name, results_root):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_labeled.parquet"
    if destination.is_file():
        labeled = pd.read_parquet(destination)
        if len(labeled) != len(frame):
            raise ValueError("Saved labeled candidate count differs from the supplied export.")
        return labeled
    question_ids = sorted(frame["sample_id"].unique())
    chunk_root = Path(results_root) / "progress" / f"{dataset_name}_label_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    parts = []
    totals = {f"{stage}_{key}": 0 for stage in ("group", "gold")
              for key in ("calls", "retries", "timeouts", "errors", "parse_timeouts")}
    totals.update(group_timeout_questions=0, gold_timeout_questions=0)
    for start in range(0, len(question_ids), 5):
        selected = question_ids[start:start + 5]
        chunk = chunk_root / f"{start:04d}_{start + len(selected):04d}.parquet"
        audit = chunk.with_suffix(".json")
        if chunk.is_file() and audit.is_file():
            part = pd.read_parquet(chunk)
            with audit.open(encoding="utf-8") as stream:
                counts = json.load(stream)
            if (set(part["sample_id"]) != set(selected) or len(part) != 32 * len(selected)
                    or set(counts) != set(totals)):
                raise ValueError(f"Saved label chunk does not match the supplied export: {chunk}")
        else:
            _parsed_math.cache_clear()
            _math_equivalent.cache_clear()
            comparison_counts(reset=True)
            counts = dict.fromkeys(totals, 0)
            rows = []
            selected_frame = frame[frame["sample_id"].isin(selected)]
            for question_id, subset in selected_frame.groupby("sample_id", sort=False):
                subset = subset.sort_values("rollout_id").copy()
                if subset["gold"].astype(str).nunique() != 1:
                    raise ValueError(f"Gold answers conflict within question {question_id}.")
                metadata = json.loads(subset.iloc[0]["metadata"])
                answers = subset["answer"].tolist()
                subset["group"] = answer_groups(answers, dataset_name, metadata)
                group_counts = comparison_counts(reset=True)
                for key, value in group_counts.items():
                    counts[f"group_{key}"] += value
                counts["group_timeout_questions"] += int(
                    group_counts["timeouts"] + group_counts["parse_timeouts"] > 0)
                subset["correct"] = correctness(answers, subset.iloc[0]["gold"], dataset_name, metadata)
                gold_counts = comparison_counts(reset=True)
                for key, value in gold_counts.items():
                    counts[f"gold_{key}"] += value
                counts["gold_timeout_questions"] += int(
                    gold_counts["timeouts"] + gold_counts["parse_timeouts"] > 0)
                rows.append(subset)
            part = pd.concat(rows, ignore_index=True)
            save_frame(part, chunk)
            save_json(audit, counts)
        for key, value in counts.items():
            totals[key] += value
        parts.append(part)
        print(f"{dataset_name} labeled {start + len(selected)}/{len(question_ids)} questions; "
              f"unresolved timeouts={totals['group_timeouts'] + totals['gold_timeouts']}",
              flush=True)
    labeled = pd.concat(parts, ignore_index=True)
    save_frame(labeled, destination)
    save_json(Path(results_root) / "metrics" / f"{dataset_name}_comparison_audit.json",
              {**totals, "timeout_policy_seconds": {
                  "group_first": GROUP_COMPARISON_TIMEOUT,
                  "gold_first": GOLD_COMPARISON_TIMEOUT,
                  "retry": RETRY_COMPARISON_TIMEOUT}})
    return labeled


def evaluate_opportunity(labeled, splits, config, dataset_name, results_root):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_opportunity.parquet"
    if destination.is_file():
        return pd.read_parquet(destination)
    indexed = {question_id: {row["rollout_id"]: row for row in subset.to_dict("records")}
               for question_id, subset in labeled.groupby("sample_id", sort=False)}
    rows = []
    for split_name in ("development", "evaluation"):
        for question_id in splits[split_name]:
            pool = indexed[question_id]
            for order_number, order in enumerate(splits["orders"][question_id]):
                for budget in config["experiment"]["budgets"]:
                    candidates = [pool[key] for key in order[:budget]]
                    rows.append({"dataset_name": dataset_name, "split": split_name,
                                 "sample_id": question_id, "order": order_number, "budget": budget,
                                 "first_correct": bool(candidates[0]["correct"]),
                                 "oracle": any(item["correct"] for item in candidates),
                                 "distinct_answers": len(set(item["group"] for item in candidates)),
                                 "candidate_ids": order[:budget],
                                 "token_cost": int(sum(item["token_count"] for item in candidates))})
    result = pd.DataFrame(rows)
    save_frame(result, destination)
    return result


def evaluate_fixed_budgets(labeled, splits, config, dataset_name, results_root):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_fixed.parquet"
    if destination.is_file():
        return pd.read_parquet(destination)
    indexed = {question_id: {row["rollout_id"]: row for row in subset.to_dict("records")}
               for question_id, subset in labeled.groupby("sample_id", sort=False)}
    rows = []
    for split_name in ("development", "evaluation"):
        for question_number, question_id in enumerate(splits[split_name], 1):
            pool = indexed[question_id]
            for order_number, rollout_ids in enumerate(splits["orders"][question_id]):
                for budget in config["experiment"]["budgets"]:
                    candidates = [pool[rollout_id] for rollout_id in rollout_ids[:budget]]
                    summary, _ = prefix_summary(candidates, dataset_name == "mathvista", include_ablations=True)
                    for result in summary:
                        result.pop("votes")
                        rows.append({"dataset_name": dataset_name, "split": split_name,
                                     "sample_id": question_id, "order": order_number,
                                     "budget": budget, **result})
            if question_number % 100 == 0:
                print(f"{dataset_name} {split_name}: {question_number} questions")
    result_frame = pd.DataFrame(rows)
    save_frame(result_frame, destination)
    return result_frame
