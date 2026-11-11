# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Score complete official lm-eval splits and save every document result."""

import importlib.metadata
import json
import os
from pathlib import Path

import torch

from models.conditions import load_condition
from common.shared import CONDITIONS, configure_huggingface_cache, read_configuration, read_json, task_paths, write_json


TASKS = ("gsm8k_cot", "hellaswag", "boolq", "arc_easy")
TASK_BATCH_SIZES = {"gsm8k_cot": 32, "hellaswag": 32, "boolq": 8, "arc_easy": 8}
EXPECTED = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}
METRICS = {
    "gsm8k_cot": ("exact_match,flexible-extract",),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "boolq": ("acc,none",),
    "arc_easy": ("acc_norm,none", "acc,none"),
}


def sample_to_record(sample, tokenizer, task, max_new_tokens):
    """Preserve prompts, scores, extracted answers, and token-cap flags."""
    record = dict(sample)
    raw = sample.get("resps", [])
    arguments = sample.get("arguments", [])
    if arguments:
        record["context_token_counts"] = [len(tokenizer.encode(str(request[0]), add_special_tokens=False))
                                          for request in arguments]
        if task == "gsm8k_cot":
            record["context_may_be_left_truncated"] = any(
                length > 2048 - max_new_tokens for length in record["context_token_counts"])
        else:
            record["continuation_token_counts"] = [
                len(tokenizer.encode(str(request[1]), add_special_tokens=False)) for request in arguments]
            record["context_may_be_left_truncated"] = any(
                context + continuation > 2048 for context, continuation in zip(
                    record["context_token_counts"], record["continuation_token_counts"]))
    raw_text = raw[0][0] if task == "gsm8k_cot" and raw and raw[0] else None
    if task == "gsm8k_cot":
        extracted = sample.get("filtered_resps", [None])[0]
        record["raw_generation"] = raw_text
        record["extracted_answer"] = extracted
        record["unextractable"] = extracted is None or not any(ch.isdigit() for ch in str(extracted))
        record["generation_tokens"] = len(tokenizer.encode(raw_text, add_special_tokens=False)) if raw_text else 0
        record["at_token_cap"] = record["generation_tokens"] >= max_new_tokens
        record["gold_answer"] = sample.get("target")
    return record


def run_benchmark_condition(condition, device="cuda", result_callback=None):
    """Run zero-shot official tasks, complete splits, without a limit argument."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    configuration = read_configuration()
    paths = task_paths(configuration)
    configure_huggingface_cache(paths["data"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    output_path = paths["results"] / "metrics" / f"{condition}_benchmarks.json"
    existing = read_json(output_path) if output_path.is_file() else None
    if existing and set(existing.get("task_results", {})) == set(TASKS) and all(
        (paths["results"] / "predictions" / f"{condition}_{task}.jsonl").is_file() for task in TASKS
    ):
        return existing
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:
        raise ImportError("Install lm-evaluation-harness in the ATML environment before Step 5.") from error

    model, tokenizer, model_info = load_condition(condition, device=device)
    model_device = next(model.parameters()).device
    if device.startswith("cuda") and model_device.type != "cuda":
        raise RuntimeError(f"Evaluation model is on {model_device}, expected CUDA.")
    summary = existing or {
        "condition": condition, "model": model_info, "lm_eval_version": importlib.metadata.version("lm_eval"),
        "task_results": {}, "task_configs": {}, "expected_split_sizes": EXPECTED,
        "zero_shot": True, "no_split_limit": True,
    }
    summary["evaluation_batch_sizes"] = TASK_BATCH_SIZES.copy()
    for task in TASKS:
        prediction_path = paths["results"] / "predictions" / f"{condition}_{task}.jsonl"
        if task in summary["task_results"] and prediction_path.is_file():
            continue
        batch_size = TASK_BATCH_SIZES[task]
        harness_model = HFLM(
            pretrained=model, tokenizer=tokenizer, device=device, batch_size=batch_size,
            max_length=configuration["model"]["max_sequence_length"],
            enable_thinking=False, truncation=False,
        )
        if harness_model.batch_size != batch_size:
            raise RuntimeError(f"{task}: lm-eval batch size is {harness_model.batch_size}, expected {batch_size}.")
        print(f"{condition}: evaluating {task} ({EXPECTED[task]} examples), "
              f"batch size {harness_model.batch_size}, model device {model_device}", flush=True)
        result = lm_eval.simple_evaluate(
            model=harness_model, tasks=[task], num_fewshot=0, batch_size=batch_size,
            limit=None, log_samples=True, apply_chat_template=(task == "gsm8k_cot"),
            gen_kwargs={"max_gen_toks": configuration["evaluation"]["gsm8k_max_new_tokens"], "do_sample": False}
            if task == "gsm8k_cot" else None,
            random_seed=configuration["experiment"]["seed"],
            numpy_random_seed=configuration["experiment"]["seed"],
            torch_random_seed=configuration["experiment"]["seed"],
            fewshot_random_seed=configuration["experiment"]["seed"],
            bootstrap_iters=1000,
        )
        task_result = result["results"][task]
        count = result["n-samples"][task]
        if count["original"] != EXPECTED[task] or count["effective"] != EXPECTED[task]:
            raise RuntimeError(f"{task}: expected all {EXPECTED[task]} rows, got {count}.")
        for metric in METRICS[task]:
            if metric not in task_result:
                raise KeyError(f"{task} is missing required lm-eval metric {metric}.")
        samples = result["samples"][task]
        if task == "gsm8k_cot":
            samples = [sample for sample in samples if sample.get("filter") == "flexible-extract"]
        if len(samples) != EXPECTED[task]:
            raise RuntimeError(f"{task}: expected {EXPECTED[task]} saved samples, got {len(samples)}.")
        with prediction_path.open("w", encoding="utf-8") as stream:
            for sample in sorted(samples, key=lambda item: item["doc_id"]):
                record = sample_to_record(sample, tokenizer, task, configuration["evaluation"]["gsm8k_max_new_tokens"])
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        summary["task_results"][task] = {key: value for key, value in task_result.items() if key in METRICS[task] or key.endswith("_stderr,none") or key.endswith("_stderr,flexible-extract")}
        summary["task_configs"][task] = result["configs"][task]
        write_json(output_path, summary)
        if result_callback:
            result_callback((prediction_path, output_path))
        del harness_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
