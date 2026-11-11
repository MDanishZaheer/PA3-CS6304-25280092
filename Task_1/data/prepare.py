# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Select, tokenize, and lock the fixed MetaMathQA training examples."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

from common.shared import read_json, write_json


def normalize_question(question):
    """Apply the documented leakage check: NFKC, casefold, and word spacing."""
    normalized = unicodedata.normalize("NFKC", str(question)).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def format_training_example(tokenizer, question, response, max_length):
    """Return untruncated chat tokens with prompt tokens masked from SFT loss."""
    user_message = [{"role": "user", "content": question}]
    conversation = user_message + [{"role": "assistant", "content": response}]
    prompt_ids = tokenizer.apply_chat_template(
        user_message, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id not in full_ids:
        raise RuntimeError("Formatted assistant response has no end-of-turn token.")
    last_end = len(full_ids) - 1 - full_ids[::-1].index(end_id)
    if tokenizer.decode(full_ids[last_end + 1 :], skip_special_tokens=False).strip():
        raise RuntimeError("Unexpected content after the assistant end-of-turn token.")
    full_ids = full_ids[: last_end + 1]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Qwen chat prompt is not a prefix of its completed conversation.")
    if len(full_ids) > max_length:
        return None
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids or response_ids[-1] != end_id:
        raise RuntimeError("Formatted assistant response must end with the end-of-turn token.")
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "input_ids": full_ids,
        "labels": labels,
        "valid_token_count": len(response_ids),
    }


def prepare_training_examples(configuration, data_root, results_root):
    """Save the first 40,000 eligible examples in original MetaMathQA order."""
    data_root, results_root = Path(data_root), Path(results_root)
    tokenized_path = data_root / "selected_training_examples"
    manifest_path = results_root / "metrics" / "selected_training_ids.json"
    if tokenized_path.is_dir() and manifest_path.is_file():
        return read_json(manifest_path)
    if tokenized_path.exists():
        raise RuntimeError(f"Incomplete selected data directory: {tokenized_path}")
    prior_manifest = read_json(manifest_path) if manifest_path.is_file() else None

    revisions = read_json(results_root / "metrics" / "hub_revisions.json")
    dataset_settings = configuration["dataset"]
    cache_dir = data_root / "huggingface" / "datasets"
    metamath_repo = dataset_settings["repo_id"]
    gsm8k_repo = dataset_settings["gsm8k_repo_id"]
    metamath = load_dataset(
        metamath_repo, split="train", revision=revisions[f"dataset:{metamath_repo}"],
        cache_dir=str(cache_dir),
    )
    gsm8k_test = load_dataset(
        gsm8k_repo, "main", split="test", revision=revisions[f"dataset:{gsm8k_repo}"],
        cache_dir=str(cache_dir),
    )
    held_out_questions = {normalize_question(row["question"]) for row in gsm8k_test}
    model_path = data_root / "models" / "Qwen3-0.6B"
    if not model_path.is_dir():
        raise FileNotFoundError(f"Download the starting model before tokenization: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    selected, skipped = [], Counter()
    tokenized_hash = hashlib.sha256()
    limit = int(dataset_settings["eligible_examples"])
    max_length = int(configuration["model"]["max_sequence_length"])
    for source_index, row in enumerate(metamath):
        question, response = str(row["query"] or "").strip(), str(row["response"] or "").strip()
        if not question or not response:
            skipped["empty"] += 1
            continue
        if any(
            normalize_question(value) in held_out_questions
            for value in (question, row.get("original_question")) if value
        ):
            skipped["gsm8k_test_overlap"] += 1
            continue
        encoded = format_training_example(tokenizer, question, response, max_length)
        if encoded is None:
            skipped["over_2048_tokens"] += 1
            continue
        encoded["source_index"] = source_index
        tokenized_hash.update(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        tokenized_hash.update(b"\n")
        selected.append(encoded)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise RuntimeError(f"Only {len(selected)} eligible examples found; expected {limit}.")
    manifest = {
        "seed": configuration["experiment"]["seed"],
        "source_repo": metamath_repo,
        "source_revision": revisions[f"dataset:{metamath_repo}"],
        "gsm8k_revision": revisions[f"dataset:{gsm8k_repo}"],
        "normalization": "Unicode NFKC, casefold, contiguous Unicode word runs joined by spaces",
        "selected_source_indices": [row["source_index"] for row in selected],
        "source_rows_examined": source_index + 1,
        "skipped": dict(skipped),
        "selected_count": len(selected),
        "tokenized_sha256": tokenized_hash.hexdigest(),
        "total_supervised_tokens": sum(row["valid_token_count"] for row in selected),
    }
    if prior_manifest is not None and prior_manifest != manifest:
        raise RuntimeError("Rebuilt training examples do not match the previous session's selection manifest.")
    Dataset.from_list(selected).save_to_disk(str(tokenized_path))
    if prior_manifest is None:
        write_json(manifest_path, manifest)
    return manifest


def load_training_examples(data_root, expected_count=40000):
    """Load the locked tokenized set without changing its order."""
    path = Path(data_root) / "selected_training_examples"
    if not path.is_dir():
        raise FileNotFoundError(f"Prepare the training examples first: {path}")
    dataset = load_from_disk(str(path))
    if len(dataset) != expected_count:
        raise ValueError(f"Expected {expected_count} selected examples, found {len(dataset)}.")
    return dataset


def save_activation_probe_ids(configuration, data_root, results_root):
    """Fix 32 math and 32 non-math probe IDs before model inspection."""
    output_path = Path(results_root) / "metrics" / "activation_probe_ids.json"
    if output_path.is_file():
        return read_json(output_path)
    seed = configuration["experiment"]["seed"]
    count = configuration["evaluation"]["activation_probe_examples_per_dataset"]
    cache_dir = Path(data_root) / "huggingface" / "datasets"
    revisions = read_json(Path(results_root) / "metrics" / "hub_revisions.json")
    gsm_repo = configuration["dataset"]["gsm8k_repo_id"]
    hellaswag_repo = next(item["repo_id"] for item in configuration["evaluation_datasets"] if item["task"] == "hellaswag")
    gsm = load_dataset(gsm_repo, "main", split="test", revision=revisions[f"dataset:{gsm_repo}"], cache_dir=str(cache_dir))
    hellaswag = load_dataset(hellaswag_repo, split="validation", revision=revisions[f"dataset:{hellaswag_repo}"], cache_dir=str(cache_dir))
    generator = np.random.default_rng(seed)
    indices = {
        "gsm8k_test_indices": generator.choice(len(gsm), count, replace=False).tolist(),
        "hellaswag_validation_indices": generator.choice(len(hellaswag), count, replace=False).tolist(),
    }
    write_json(output_path, indices)
    return indices

