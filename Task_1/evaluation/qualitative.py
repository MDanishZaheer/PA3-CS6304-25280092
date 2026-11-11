# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Generate the supplied 24 prompts and prepare 42 manual annotation rows."""

import csv
import json
from pathlib import Path

import torch
import yaml

from models.conditions import load_condition
from models.qwen import assert_non_thinking_prompt
from common.shared import CONDITIONS, read_configuration, task_paths, write_json


FORMATS = ("plain", "json", "template")
ANNOTATION_FIELDS = (
    "format_compliant", "coherent", "relevant", "repetitive",
    "unnecessary_math", "inappropriate_content", "truncated", "notes",
)


def load_qualitative_prompts(path):
    """Accept either 8 question groups or 24 explicit prompt records."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Place the course-supplied YAML at {path} before Step 7.")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = document.get("questions", document.get("prompts")) if isinstance(document, dict) else document
    if not isinstance(questions, list):
        raise ValueError("Qualitative YAML must contain a questions or prompts list.")
    records = []
    if len(questions) == 8:
        for position, item in enumerate(questions):
            if not isinstance(item, dict):
                raise ValueError("Each question group needs plain, json, and template prompt text.")
            group_id = str(item.get("id", position + 1))
            for format_name in FORMATS:
                prompt = item.get(format_name)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Question {group_id} is missing its {format_name} prompt.")
                records.append({"question_id": group_id, "format": format_name, "prompt": prompt})
    elif len(questions) == 24 and all(
            isinstance(item, dict) and {"id", "paraphrases"}.issubset(item) for item in questions):
        for item in questions:
            identifier = str(item["id"])
            format_name = next((name for name in ("json", "template")
                                if identifier.endswith("_" + name)), "plain")
            group_id = identifier.removesuffix("_" + format_name) if format_name != "plain" else identifier
            paraphrases = item["paraphrases"]
            if not isinstance(paraphrases, list) or len(paraphrases) != 1:
                raise ValueError(f"{identifier} must contain one supplied paraphrase.")
            system = item.get("system")
            if format_name == "json" and (not isinstance(system, str) or not system.strip()):
                raise ValueError(f"{identifier} needs its JSON system instruction.")
            records.append({"question_id": group_id, "format": format_name,
                            "prompt": paraphrases[0], "system": system})
    elif len(questions) == 24:
        for item in questions:
            if not isinstance(item, dict) or not {"question_id", "format", "prompt"}.issubset(item):
                raise ValueError("Each of 24 prompts needs question_id, format, and prompt fields.")
            records.append({"question_id": str(item["question_id"]),
                            "format": str(item["format"]).lower(), "prompt": str(item["prompt"]),
                            "system": item.get("system")})
    else:
        raise ValueError(f"Expected 8 groups or 24 prompts, found {len(questions)}.")
    group_ids = list(dict.fromkeys(item["question_id"] for item in records))
    if len(group_ids) != 8 or len(records) != 24:
        raise ValueError("The YAML must define exactly 8 underlying questions and 24 prompts.")
    for group_id in group_ids:
        formats = [item["format"] for item in records if item["question_id"] == group_id]
        if sorted(formats) != sorted(FORMATS):
            raise ValueError(f"Question {group_id} needs one prompt of each format.")
    if any(not isinstance(item["prompt"], str) or not item["prompt"].strip() for item in records):
        raise ValueError("Qualitative prompt text cannot be empty.")
    return records


def _generate_one(model, tokenizer, prompt, max_new_tokens, device, system=None):
    ids = assert_non_thinking_prompt(tokenizer, prompt, system=system)
    if len(ids) + max_new_tokens > 2048:
        raise ValueError("A qualitative prompt exceeds the 2,048-token context budget with 512 generation tokens.")
    inputs = torch.tensor([ids], device=device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
    new_ids = output[0, len(ids):]
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids), len(ids)


def generate_qualitative_condition(condition, device="cuda"):
    """Save exact raw text, prompt ID, output length, and token-cap flag."""
    configuration = read_configuration()
    paths = task_paths(configuration)
    prompt_path = paths["root"] / configuration["paths"]["qualitative_questions"]
    prompts = load_qualitative_prompts(prompt_path)
    output_path = paths["results"] / "predictions" / f"{condition}_qualitative.jsonl"
    if output_path.is_file():
        with output_path.open(encoding="utf-8") as stream:
            existing = [json.loads(line) for line in stream]
        if len(existing) == len(prompts) and all(
            (item["question_id"], item["format"], item["prompt"], item.get("system")) ==
            (prompt["question_id"], prompt["format"], prompt["prompt"], prompt.get("system"))
            for item, prompt in zip(existing, prompts)
        ):
            return existing
        if len(existing) >= len(prompts):
            raise RuntimeError(f"Saved qualitative outputs disagree with the supplied YAML: {output_path}")
    model, tokenizer, _ = load_condition(condition, device=device)
    maximum = configuration["evaluation"]["qualitative_max_new_tokens"]
    records = []
    with output_path.open("w", encoding="utf-8") as stream:
        for prompt in prompts:
            response, response_tokens, prompt_tokens = _generate_one(
                model, tokenizer, prompt["prompt"], maximum, device, prompt.get("system"))
            record = {"condition": condition, **prompt, "response": response,
                      "prompt_tokens": prompt_tokens, "response_tokens": response_tokens,
                      "at_token_cap": response_tokens >= maximum}
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def prepare_annotations():
    """Create 42 first/last-question rows, preserving existing human entries."""
    configuration = read_configuration()
    paths = task_paths(configuration)
    prompt_path = paths["root"] / configuration["paths"]["qualitative_questions"]
    prompts = load_qualitative_prompts(prompt_path)
    group_ids = list(dict.fromkeys(item["question_id"] for item in prompts))
    selected_ids = {group_ids[0], group_ids[-1]}
    annotation_path = paths["results"] / "annotations" / "qualitative_42.csv"
    if annotation_path.is_file():
        return annotation_path
    rows = []
    for condition in CONDITIONS:
        output_path = paths["results"] / "predictions" / f"{condition}_qualitative.jsonl"
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        with output_path.open(encoding="utf-8") as stream:
            generated = [json.loads(line) for line in stream]
        if len(generated) != 24:
            raise RuntimeError(f"Expected 24 outputs for {condition}.")
        for item in generated:
            if item["question_id"] in selected_ids:
                rows.append({"condition": condition, "question_id": item["question_id"],
                             "format": item["format"], "prompt": item["prompt"],
                             "system": item.get("system") or "",
                             "response": item["response"], "excerpt": item["response"][:300],
                             "at_token_cap": item["at_token_cap"],
                             **{field: "" for field in ANNOTATION_FIELDS}})
    if len(rows) != 42:
        raise RuntimeError(f"Expected exactly 42 annotation rows, got {len(rows)}.")
    with annotation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(paths["results"] / "metrics" / "qualitative_selection.json",
               {"question_ids": [group_ids[0], group_ids[-1]], "formats": FORMATS,
                "conditions": CONDITIONS, "annotation_rows": len(rows)})
    return annotation_path
