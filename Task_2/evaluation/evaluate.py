# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Teacher-forced and greedy visual-grounding evaluations on fixed pairs."""

import json
from collections import defaultdict

import numpy as np
import torch

from common.shared import read_json, write_json
from data.answers import normalize_answer, vqa_soft_score


def _flatten(pairs):
    return [(pair, position, example) for pair in pairs
            for position, example in enumerate(pair["examples"])]


def _mismatch_mapping(pairs, path):
    """Map each question to an image from another pair in the same suite."""
    if path.is_file():
        mapping = read_json(path)
        allowed = {example["image_key"] for pair in pairs for example in pair["examples"]}
        for pair in pairs:
            originals = {example["image_key"] for example in pair["examples"]}
            for position in (0, 1):
                key = mapping.get(f"{pair['pair_id']}:{position}")
                if key not in allowed or key in originals:
                    raise RuntimeError("Saved mismatched-image map violates suite or pair isolation.")
        return mapping
    if len(pairs) < 2:
        raise ValueError("Mismatched controls require at least two pairs.")
    mapping = {}
    for index, pair in enumerate(pairs):
        donor = pairs[(index + 1) % len(pairs)]
        for position in (0, 1):
            mapping[f"{pair['pair_id']}:{position}"] = donor["examples"][position]["image_key"]
    write_json(path, mapping)
    return mapping


def _visual_keys(rows, condition, mismatch=None):
    keys = []
    for pair, position, example in rows:
        if condition == "correct":
            keys.append(example["image_key"])
        elif condition == "complementary":
            keys.append(pair["examples"][1 - position]["image_key"])
        elif condition == "mismatched":
            keys.append(mismatch[f"{pair['pair_id']}:{position}"])
        elif condition == "no_image":
            keys.append(None)
        else:
            raise ValueError(condition)
    return keys


@torch.inference_mode()
def reference_losses(model, examples, image_keys, store, include_image=True,
                     include_end=False, batch_size=8):
    """Average target-token NLL per example, excluding end token for answers."""
    losses = []
    for first in range(0, len(examples), batch_size):
        subset = examples[first:first + batch_size]
        features = (store.get(image_keys[first:first + batch_size]) if include_image else None)
        with torch.autocast("cuda", dtype=torch.float16):
            _, per_example = model.supervised_loss(subset, features, include_image=include_image,
                                                   include_end=include_end)
        losses.extend(float(value) for value in per_example.cpu())
    return losses


def run_development(model, selection, store, paths, update):
    """Evaluate scheduled caption and paired VQA development inputs."""
    output = paths["results"] / "metrics" / f"development_{update:04d}.json"
    detail_path = paths["results"] / "predictions" / f"development_{update:04d}.jsonl"
    if output.is_file() and detail_path.is_file():
        return read_json(output)
    was_training = model.training
    model.eval()
    captions = selection["caption_dev"]
    caption_keys = [item["image_key"] for item in captions]
    caption_losses = reference_losses(model, captions, caption_keys, store, include_end=True)
    caption_loss = float(np.mean(caption_losses))
    result = {"update": update, "caption_reference_loss": caption_loss, "suites": {}}
    detail = [{"update": update, "suite": "caption_dev", "example": item,
               "reference_loss": loss} for item, loss in zip(captions, caption_losses)]
    for suite in ("grounding_dev", "natural_dev"):
        rows = _flatten(selection[suite])
        examples = [example for _, _, example in rows]
        suite_scores = {}
        for condition in ("correct", "complementary", "no_image"):
            keys = _visual_keys(rows, condition)
            nll = reference_losses(model, examples, keys, store,
                                   include_image=condition != "no_image")
            suite_scores[condition] = nll
            for (pair, position, example), key, value in zip(rows, keys, nll):
                detail.append({"update": update, "suite": suite, "condition": condition,
                               "pair_id": pair["pair_id"], "position": position,
                               "example": example, "visual_image_key": key,
                               "reference_nll": value})
        result["suites"][suite] = {
            "correct_nll": float(np.mean(suite_scores["correct"])),
            "complementary_nll": float(np.mean(suite_scores["complementary"])),
            "no_image_nll": float(np.mean(suite_scores["no_image"])),
            "image_swap_penalty": float(np.mean(np.array(suite_scores["complementary"]) -
                                                np.array(suite_scores["correct"])))
        }
    write_json(output, result)
    with detail_path.open("w", encoding="utf-8") as stream:
        for item in detail:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    if was_training:
        model.train()
    return result


def _score_rows(rows, suite):
    for row in rows:
        prediction = row["prediction"]
        example = row["example"]
        row["normalized_prediction"] = normalize_answer(prediction)
        row["canonical_correct"] = float(row["normalized_prediction"] == example["canonical"])
        row["accuracy"] = (vqa_soft_score(prediction, example["references"])
                           if suite == "natural_test" else row["canonical_correct"])
        partner = row["partner_example"]
        row["partner_canonical_correct"] = float(row["normalized_prediction"] == partner["canonical"])
        row["partner_accuracy"] = (vqa_soft_score(prediction, partner["references"])
                                   if suite == "natural_test" else row["partner_canonical_correct"])
        row["unsupported_answer"] = float(not row["canonical_correct"] and
                                           not row["partner_canonical_correct"])
        row["empty_answer"] = float(not prediction.strip())


def _pair_measures(rows):
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    result = {}
    for pair_id, pair_rows in by_pair.items():
        pair_rows.sort(key=lambda row: row["position"])
        if len(pair_rows) != 2:
            raise RuntimeError(f"Pair {pair_id} is incomplete.")
        result[pair_id] = {
            "accuracy": float(np.mean([row["accuracy"] for row in pair_rows])),
            "partner_accuracy": float(np.mean([row["partner_accuracy"] for row in pair_rows])),
            "unsupported_answer_rate": float(np.mean([row["unsupported_answer"] for row in pair_rows])),
            "pair_accuracy": float(all(row["canonical_correct"] for row in pair_rows)),
            "identical_answer_rate": float(pair_rows[0]["normalized_prediction"] ==
                                           pair_rows[1]["normalized_prediction"]),
            "empty_answer_rate": float(np.mean([row["empty_answer"] for row in pair_rows])),
            "reference_nll": float(np.mean([row["reference_nll"] for row in pair_rows])),
            "generation_seconds_per_answer": float(np.mean([row["generation_seconds"] for row in pair_rows])),
        }
    return result


def _bootstrap(condition_pairs, count, seed):
    """Resample complete paired IDs and match IDs across image controls."""
    ids = sorted(condition_pairs["correct"])
    if not all(sorted(condition_pairs[name]) == ids for name in condition_pairs):
        raise RuntimeError("Image conditions have different pair IDs.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(count, len(ids)))
    summary = {}
    for condition, pairs in condition_pairs.items():
        summary[condition] = {}
        for metric in next(iter(pairs.values())):
            values = np.array([pairs[pair_id][metric] for pair_id in ids])
            replicates = values[draws].mean(axis=1)
            summary[condition][metric] = {"mean": float(values.mean()),
                                          "ci95": np.percentile(replicates, [2.5, 97.5]).tolist()}
    for metric, sign in (("accuracy", 1), ("reference_nll", -1)):
        first = np.array([condition_pairs["correct"][pair_id][metric] for pair_id in ids])
        second = np.array([condition_pairs["complementary"][pair_id][metric] for pair_id in ids])
        difference = sign * (first - second)
        summary[f"correct_advantage_{metric}"] = {
            "mean": float(difference.mean()),
            "ci95": np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]).tolist()
        }
    return summary


def run_final_evaluation(model, selection, store, config, paths, result_callback=None):
    """Score both held-out suites under all four prescribed image controls."""
    output = paths["results"] / "metrics" / "final_evaluation.json"
    if output.is_file():
        saved = read_json(output)
        if (set(saved.get("suites", {})) == {"grounding_test", "natural_test"} and
                all((paths["results"] / "predictions" / f"{suite}.jsonl").is_file()
                    for suite in ("grounding_test", "natural_test"))):
            return saved
    model.eval()
    complete = (read_json(output) if output.is_file() else
                {"update": 4500, "suites": {}, "bootstrap_pairs": config["evaluation"]["bootstrap_pairs"]})
    for suite in ("grounding_test", "natural_test"):
        if suite in complete["suites"] and (paths["results"] / "predictions" / f"{suite}.jsonl").is_file():
            continue
        pairs = selection[suite]
        rows = _flatten(pairs)
        examples = [example for _, _, example in rows]
        mismatch_path = paths["results"] / "splits" / f"{suite}_mismatches.json"
        mismatches = _mismatch_mapping(pairs, mismatch_path)
        all_rows = []
        condition_pairs = {}
        for condition in config["evaluation"]["conditions"]:
            keys = _visual_keys(rows, condition, mismatches)
            include_image = condition != "no_image"
            nlls = reference_losses(model, examples, keys, store, include_image=include_image)
            predictions = []
            batch_size = config["evaluation"]["generation_batch_size"]
            for first in range(0, len(examples), batch_size):
                subset = examples[first:first + batch_size]
                features = store.get(keys[first:first + batch_size]) if include_image else None
                predictions.extend(model.generate_answers(subset, features, include_image=include_image,
                    max_new_tokens=config["evaluation"]["generation_max_new_tokens"]))
            condition_rows = []
            for (pair, position, example), key, nll, generated in zip(rows, keys, nlls, predictions):
                item = {"suite": suite, "condition": condition, "pair_id": pair["pair_id"],
                        "kind": pair.get("kind"), "answer_type": pair.get("answer_type"),
                        "position": position, "example": example,
                        "partner_example": pair["examples"][1 - position], "visual_image_key": key,
                        "reference_nll": nll, **generated}
                condition_rows.append(item)
            _score_rows(condition_rows, suite)
            all_rows.extend(condition_rows)
            condition_pairs[condition] = _pair_measures(condition_rows)
            print(f"{suite}: {condition} complete ({len(condition_rows)} answers)", flush=True)
        prediction_path = paths["results"] / "predictions" / f"{suite}.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for item in all_rows:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        bootstrap = _bootstrap(condition_pairs, config["evaluation"]["bootstrap_pairs"],
                               config["experiment"]["seed"])
        breakdown = {}
        type_metrics = {}
        pair_types = {pair["pair_id"]: pair.get("answer_type") or pair.get("kind") for pair in pairs}
        for condition in config["evaluation"]["conditions"]:
            by_type = defaultdict(list)
            for row in all_rows:
                if row["condition"] == condition:
                    by_type[row["answer_type"] or row["kind"]].append(row["accuracy"])
            breakdown[condition] = {kind: float(np.mean(values)) for kind, values in by_type.items()}
            typed_pairs = defaultdict(list)
            for pair_id, values in condition_pairs[condition].items():
                typed_pairs[pair_types[pair_id]].append(values)
            type_metrics[condition] = {
                kind: {metric: float(np.mean([values[metric] for values in group]))
                       for metric in group[0]}
                for kind, group in typed_pairs.items()}
        complete["suites"][suite] = {"pair_count": len(pairs), "question_count": len(examples),
                                      "conditions": bootstrap, "question_type_accuracy": breakdown,
                                      "question_type_metrics": type_metrics}
        write_json(output, complete)
        if result_callback:
            result_callback()
    return complete
