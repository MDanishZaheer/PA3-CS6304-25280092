# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load supplied rollout exports and fix question splits and candidate orders."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.shared import read_json, save_frame, save_json


KEYS = ["dataset", "sample_id", "rollout_id"]
ALIASES = {
    "response": ["response", "generated_text", "generation", "text", "output_text"],
    "answer": ["extracted_answer", "answer_extracted", "predicted_answer", "prediction"],
    "gold": ["gold_answer", "ground_truth", "reference_answer", "ground_truth_answer"],
    "question": ["question_text", "question", "problem", "query", "prompt"],
    "token_ids": ["generated_token_ids", "token_ids", "output_token_ids"],
    "token_count": ["n_generated_text_tokens", "generated_tokens", "num_generated_tokens", "output_tokens", "token_count", "length"],
    "logp": ["logp", "sampled_token_logp", "token_logp", "log_probs"],
    "kl_u_p": ["kl_u_p", "token_kl_u_p", "kl_values"],
    "image_attention": ["img_attn_total", "image_attention", "attention_image", "a_image", "image_attention_total"],
    "prompt_attention": ["prompt_text_attn_total", "prompt_attention", "attention_prompt", "a_prompt", "prompt_attention_total"],
    "generated_attention": ["generated_text_attn_total", "generated_attention", "attention_generated", "a_generated", "generated_attention_total"],
}
EXPECTED_QUALITY = {"math": (20742094, 523, 521),
                    "mathvista": (8961703, 487, 3032)}


def find_export(data_root, folder_hint, filename):
    matches = [path for path in Path(data_root).rglob(filename)
               if folder_hint.lower() in str(path.parent).lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} under a folder containing {folder_hint!r}; found {matches} in {data_root}.")
    return matches[0]


def _rename_aliases(frame, names):
    changes = {}
    for target in names:
        if target in frame:
            continue
        found = [name for name in ALIASES[target] if name in frame]
        if len(found) == 1:
            changes[found[0]] = target
        elif not found:
            raise ValueError(f"Missing {target!r} column. Available columns: {list(frame.columns)}. Check the supplied README.MD schema.")
        else:
            raise ValueError(f"Ambiguous columns for {target}: {found}.")
    return frame.rename(columns=changes)


def _list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        return json.loads(value)
    raise ValueError(f"Expected a per-token list, got {type(value).__name__}.")


def _merge_export(generations, metrics, name):
    for frame in (generations, metrics):
        absent = set(KEYS) - set(frame)
        if absent:
            raise ValueError(f"{name} lacks join keys {sorted(absent)}; columns={list(frame.columns)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has duplicate (dataset, sample_id, rollout_id) keys; inspect README.MD.")
    overlaps = (set(generations) & set(metrics)) - set(KEYS)
    for column in overlaps:
        comparison = generations[KEYS + [column]].merge(
            metrics[KEYS + [column]], on=KEYS, how="inner", validate="one_to_one",
            suffixes=("_left", "_right"))
        if len(comparison) != len(generations) or not all(
                np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
                if isinstance(left, (list, np.ndarray)) and isinstance(right, (list, np.ndarray))
                else (left == right or (pd.isna(left) and pd.isna(right)))
                for left, right in zip(comparison[f"{column}_left"], comparison[f"{column}_right"])):
            raise ValueError(f"Conflicting {column!r} columns across supplied exports.")
    metrics = metrics.drop(columns=list(overlaps))
    merged = generations.merge(metrics, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{name} tables do not have identical candidate keys.")
    return merged.drop(columns="_merge")


def _read_metadata(data_root, configured_path):
    if configured_path:
        destination = Path(configured_path)
    else:
        supplied = list(Path(data_root).rglob("testmini.json"))
        if len(supplied) > 1:
            raise ValueError(f"Several MathVista testmini metadata files are attached: {supplied}")
        destination = (supplied[0] if supplied else
                       Path(__file__).resolve().parents[2] / "Data" / "Task_3" / "testmini.json")
    if not destination.is_file():
        raise FileNotFoundError(f"MathVista metadata is required at {destination}.")
    with destination.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if len(metadata) != 1000:
        raise ValueError(f"Expected 1000 MathVista testmini metadata records, found {len(metadata)}.")
    required = {"choices", "question_type", "answer_type", "precision"}
    for question_id, value in metadata.items():
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(f"MathVista question {question_id} lacks required answer metadata.")
        if value["question_type"] == "multi_choice" and not value["choices"]:
            raise ValueError(f"MathVista question {question_id} has no answer choices.")
    return {str(key): {field: value.get(field) for field in
                       ("choices", "question_type", "answer_type", "precision")}
            for key, value in metadata.items()}


def load_candidates(data_root, config, dataset_name):
    """Read one model-dataset export, validate one row per rollout and token alignment."""
    is_vision = dataset_name == "mathvista"
    hint = config["data"]["mathvista_folder_hint" if is_vision else "math_folder_hint"]
    generations = pd.read_parquet(find_export(data_root, hint, "generations.parquet"))
    probabilities = pd.read_parquet(find_export(data_root, hint, "probability_metrics.parquet"))
    frame = _merge_export(generations, probabilities, "generation/probability")
    if is_vision:
        attention = pd.read_parquet(find_export(data_root, hint, "attention_metrics.parquet"))
        if "n_generated_text_tokens" not in attention:
            raise ValueError("VLM attention export lacks preceding-token counts.")
        attention = attention.rename(columns={"n_generated_text_tokens": "preceding_token_count"})
        frame = _merge_export(frame, attention, "generation/attention")
    frame = _rename_aliases(frame, ["response", "answer", "gold", "question", "token_ids", "logp", "kl_u_p"])
    if is_vision:
        frame = _rename_aliases(frame, ["image_attention", "prompt_attention", "generated_attention"])
    if "token_count" not in frame:
        for alias in ALIASES["token_count"]:
            if alias in frame:
                frame = frame.rename(columns={alias: "token_count"})
                break
    if "token_count" not in frame:
        frame["token_count"] = frame["token_ids"].map(len)
    if frame.duplicated(KEYS).any() or len(frame) != 32000:
        raise ValueError(f"{dataset_name}: expected 32000 distinct rollouts, found {len(frame)}.")
    counts = frame.groupby("sample_id").size()
    if len(counts) != 1000 or not counts.eq(32).all():
        raise ValueError(f"{dataset_name}: expected 1000 questions with exactly 32 rollouts each.")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["rollout_id"] = frame["rollout_id"].astype(str)
    list_fields = ["token_ids", "logp", "kl_u_p"]
    if is_vision:
        list_fields += ["image_attention", "prompt_attention", "generated_attention"]
    for field in list_fields:
        frame[field] = frame[field].map(_list)
    for row in frame.itertuples(index=False):
        length = len(row.token_ids)
        if int(row.token_count) != length or any(len(getattr(row, field)) != length for field in list_fields):
            raise ValueError(f"Token alignment failed for {dataset_name}/{row.sample_id}/{row.rollout_id}.")
        if is_vision and _list(row.preceding_token_count) != list(range(length)):
            raise ValueError(f"VLM preceding-token counts failed for {row.sample_id}/{row.rollout_id}.")
    if is_vision:
        metadata = _read_metadata(data_root, config["data"].get("mathvista_metadata"))
        frame["metadata"] = frame["sample_id"].map(metadata)
        if frame["metadata"].isna().any():
            raise ValueError("Some MathVista sample IDs are absent from testmini metadata.")
        frame["metadata"] = frame["metadata"].map(json.dumps)
    else:
        frame["metadata"] = "{}"
    frame["dataset_name"] = dataset_name
    return frame


def save_splits_and_orders(frame, config, results_root, dataset_name):
    """Use one seeded question permutation and ten nested candidate orders."""
    destination = Path(results_root) / "splits" / f"{dataset_name}.json"
    if destination.is_file():
        saved = read_json(destination)
        if (saved.get("seed") != 6304 or len(saved["development"]) != 200 or
                len(saved["evaluation"]) != 800 or
                set(saved["development"] + saved["evaluation"]) != set(frame["sample_id"]) or
                any(len(saved["orders"].get(question_id, [])) != 10
                    for question_id in saved["development"] + saved["evaluation"]) or
                any(len(order) != 32 or len(set(order)) != 32
                    for orders in saved["orders"].values() for order in orders)):
            raise ValueError("Saved question split does not match the attached export.")
        return saved
    rng = np.random.default_rng(config["experiment"]["seed"])
    question_ids = sorted(frame["sample_id"].unique())
    question_ids = [question_ids[index] for index in rng.permutation(len(question_ids))]
    orders = {}
    for question_id in question_ids:
        ids = sorted(frame.loc[frame["sample_id"] == question_id, "rollout_id"].tolist())
        orders[question_id] = [[ids[index] for index in rng.permutation(32)]
                               for _ in range(config["experiment"]["orders"])]
    saved = {"seed": 6304, "development": question_ids[:200],
             "evaluation": question_ids[200:], "orders": orders}
    save_json(destination, saved)
    return saved


def prepare_dataset(data_root, config, results_root, dataset_name):
    destination = Path(results_root) / "predictions" / f"{dataset_name}_candidates.parquet"
    if destination.is_file():
        frame = pd.read_parquet(destination)
    else:
        frame = load_candidates(data_root, config, dataset_name)
        save_frame(frame, destination)
    quality = {"questions": int(frame["sample_id"].nunique()), "rollouts": len(frame),
               "retained_tokens": int(frame["token_count"].sum()),
               "capped_rollouts": int(frame["hit_max_new_tokens"].sum()),
               "null_extractions": int(frame["answer"].isna().sum())}
    actual = (quality["retained_tokens"], quality["capped_rollouts"], quality["null_extractions"])
    if actual != EXPECTED_QUALITY[dataset_name]:
        raise ValueError(f"{dataset_name} export totals {actual} differ from supplied README.MD {EXPECTED_QUALITY[dataset_name]}.")
    save_json(Path(results_root) / "metrics" / f"{dataset_name}_export_quality.json", quality)
    return frame, save_splits_and_orders(frame, config, results_root, dataset_name)
