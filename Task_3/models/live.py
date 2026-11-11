# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Fixed FP16 Qwen generation for the 50-question live comparison."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from common.shared import read_json, save_json
from data.answers import answer_groups, correctness, last_boxed


SYSTEM = "You are a helpful assistant."
SUFFIX = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVISION = ("Review the solution above. Identify and correct any errors; if it is correct, "
            "retain its answer. Give your final answer inside \\boxed{}.")


def _file_stem(sample_id):
    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:16]


def _seed(sample_id, label):
    digest = hashlib.sha256(f"6304:{sample_id}:{label}".encode("utf-8")).hexdigest()
    return 6304 + int(digest[:8], 16) % 100000000


def _generate(model, tokenizer, messages, cap, seed, device):
    import torch

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=True, return_dict=True,
                                            return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Transformers samples from the process CUDA RNG; generate() does not take
    # a per-call torch.Generator argument in the supported versions.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8,
                                top_k=20, repetition_penalty=1.0, max_new_tokens=cap,
                                pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize(device)
    tokens = output[0, inputs["input_ids"].shape[-1]:].tolist()
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    response = tokenizer.decode(tokens, skip_special_tokens=True)
    seconds = time.perf_counter() - start
    return {"messages": messages, "response": response, "extracted_answer": last_boxed(response),
            "seed": seed, "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": len(tokens), "truncated": len(tokens) >= cap and (not tokens or tokens[-1] not in eos_ids),
            "runtime_seconds": seconds, "peak_gpu_gb": torch.cuda.max_memory_allocated(device) / 2**30}


def _question_calls(model, tokenizer, question, results_root, device, revision, deadline_epoch):
    question_id = question["sample_id"]
    root = Path(results_root) / "progress" / "live_calls"
    root.mkdir(parents=True, exist_ok=True)
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{question['question']}\n\n{SUFFIX}"}]
    calls = {}
    for label, cap in [("single", 3072), ("parallel_0", 1024),
                       ("parallel_1", 1024), ("parallel_2", 1024), ("revision", 2048)]:
        path = root / f"{_file_stem(question_id)}_{label}.json"
        if path.is_file():
            saved = read_json(path)
            if (saved["sample_id"] != question_id or saved["label"] != label or
                    saved["model_revision"] != revision or saved["seed"] != _seed(question_id, label)):
                raise ValueError(f"Live result identity mismatch: {path}")
            calls[label] = saved
            continue
        if time.time() >= deadline_epoch:
            print(f"Live {question_id}: reached the 11-hour cutoff; saved calls remain resumable.", flush=True)
            return calls
        messages = base if label != "revision" else base + [
            {"role": "assistant", "content": calls["parallel_0"]["response"]},
            {"role": "user", "content": REVISION}]
        result = _generate(model, tokenizer, messages, cap, _seed(question_id, label), device)
        calls[label] = {"sample_id": question_id, "label": label, "model_revision": revision,
                        "gold": question["gold"], "question": question["question"], **result}
        save_json(path, calls[label])
        print(f"Live {question_id}: saved {label}, {result['output_tokens']} output tokens", flush=True)
    return calls


def _load_model(config, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = config["model"].get("revision")
    if not revision:
        raise ValueError("Set model.revision to the exact Qwen2.5-3B-Instruct revision in the supplied README.MD.")
    cache_dir = os.environ.get("HF_HUB_CACHE")
    if not cache_dir or not Path(cache_dir).is_dir():
        raise FileNotFoundError("Set HF_HUB_CACHE to the writable, predownloaded model cache.")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], revision=revision,
                                              cache_dir=cache_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(config["model"]["name"], revision=revision,
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True,
                                                 cache_dir=cache_dir,
                                                 local_files_only=True).to(device).eval()
    return tokenizer, model


def smoke_live_loading(config):
    """Catch cache, model-loading, and generation errors before offline analysis."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Live generation requires a CUDA GPU.")
    device = torch.device("cuda:0")
    tokenizer, model = _load_model(config, device)
    try:
        probe = _generate(model, tokenizer,
                          [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": "What is 2 + 2?\n\n" + SUFFIX}],
                          1, 6304, device)
        if probe["output_tokens"] != 1:
            raise RuntimeError("One-token live generation smoke check returned no token.")
    finally:
        del model, tokenizer
        torch.cuda.empty_cache()
    return probe["input_tokens"]


def run_worker(question_file, results_root, config, shard, num_shards, deadline_epoch):
    import torch

    questions = read_json(question_file)[shard::num_shards]
    pending = [question for question in questions if not all(
        (Path(results_root) / "progress" / "live_calls" /
         f"{_file_stem(question['sample_id'])}_{label}.json").is_file()
        for label in ("single", "parallel_0", "parallel_1", "parallel_2", "revision"))]
    if not pending:
        return
    if time.time() >= deadline_epoch:
        print("Live worker reached the session reserve before model loading.", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Live generation requires a CUDA GPU.")
    device = torch.device("cuda:0")
    tokenizer, model = _load_model(config, device)
    revision = config["model"]["revision"]
    for question in pending:
        if time.time() >= deadline_epoch:
            print("Live worker reached the session reserve; stopping after saved calls.", flush=True)
            break
        _question_calls(model, tokenizer, question, results_root, device, revision, deadline_epoch)


def live_questions(labeled, split, results_root):
    indexed = labeled.groupby("sample_id", sort=False).first()
    questions = [{"sample_id": question_id, "question": str(indexed.loc[question_id, "question"]),
                  "gold": str(indexed.loc[question_id, "gold"])}
                 for question_id in split["evaluation"][:50]]
    destination = Path(results_root) / "progress" / "live_questions.json"
    if destination.is_file() and read_json(destination) != questions:
        raise ValueError("Saved live-question subset differs from the current export or split.")
    save_json(destination, questions)
    return destination


def launch_live(question_file, config, results_root, deadline_epoch):
    """Use one worker per available T4, with a separate FP16 replica on each GPU."""
    import torch

    if not config["model"].get("revision"):
        raise ValueError("Set model.revision from the supplied README.MD before live generation.")
    gpu_count = min(torch.cuda.device_count(), 2)
    if gpu_count < 1:
        raise RuntimeError("Kaggle GPU must be enabled for Task 3 live generation.")
    if gpu_count == 1:
        run_worker(question_file, results_root, config, 0, 1, deadline_epoch)
    else:
        processes = []
        for shard in range(gpu_count):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(shard)
            command = [sys.executable, "-m", "models.live_worker", str(question_file),
                       str(results_root), str(shard), str(gpu_count), str(deadline_epoch),
                       config["model"]["revision"]]
            processes.append(subprocess.Popen(command, env=environment,
                                              cwd=Path(__file__).resolve().parents[1]))
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"A Task 3 live worker exited with code {process.returncode}.")
    return collect_live(question_file, results_root, config["model"]["revision"])


def collect_live(question_file, results_root, revision):
    """Build one row per question and strategy; incomplete questions stay pending."""
    rows = []
    for question in read_json(question_file):
        root = Path(results_root) / "progress" / "live_calls"
        names = ("single", "parallel_0", "parallel_1", "parallel_2", "revision")
        paths = {name: root / f"{_file_stem(question['sample_id'])}_{name}.json" for name in names}
        if not all(path.is_file() for path in paths.values()):
            continue
        calls = {name: read_json(path) for name, path in paths.items()}
        if any(call["model_revision"] != revision or
               call["seed"] != _seed(question["sample_id"], name)
               for name, call in calls.items()):
            raise ValueError(f"Saved live calls do not match the pinned revision or seeds for {question['sample_id']}.")
        parallel = [calls[f"parallel_{index}"] for index in range(3)]
        groups = answer_groups([call["extracted_answer"] for call in parallel], "math", {})
        votes = {group: groups.count(group) for group in groups}
        winner = next(index for index, group in enumerate(groups) if votes[group] == max(votes.values()))
        definitions = {"single": [calls["single"]], "parallel": parallel,
                       "refinement": [parallel[0], calls["revision"]]}
        selected = {"single": calls["single"], "parallel": parallel[winner],
                    "refinement": calls["revision"]}
        for strategy, used in definitions.items():
            answer = selected[strategy]["extracted_answer"]
            rows.append({"sample_id": question["sample_id"], "strategy": strategy,
                         "answer": answer, "gold": question["gold"],
                         "correct": correctness([answer], question["gold"], "math", {})[0],
                         "extracted": answer is not None,
                         "input_tokens": sum(call["input_tokens"] for call in used),
                         "output_tokens": sum(call["output_tokens"] for call in used),
                         "runtime_seconds": sum(call["runtime_seconds"] for call in used),
                         "truncated_calls": sum(call["truncated"] for call in used),
                         "peak_gpu_gb": max(call["peak_gpu_gb"] for call in used),
                         "selected_call": selected[strategy]["label"]})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        from common.shared import save_frame
        save_frame(frame, Path(results_root) / "predictions" / "live_strategies.parquet")
    return frame
