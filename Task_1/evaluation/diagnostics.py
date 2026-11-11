# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compute down-projection displacement and fixed-prompt activation effects."""

import csv
import gc
from pathlib import Path

import torch
from datasets import load_dataset

from data.prepare import save_activation_probe_ids
from models.qwen import assert_non_thinking_prompt, load_qwen
from common.shared import configure_huggingface_cache, read_configuration, read_json, task_paths, write_json


DIAGNOSTIC_CONDITIONS = ("full_ft_mid", "full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16")


def _probe_prompts(configuration, paths, probe_ids):
    """Use original GSM questions and stock HellaSwag completion contexts."""
    from lm_eval.tasks import TaskManager

    revisions = read_json(paths["results"] / "metrics" / "hub_revisions.json")
    repo = configuration["dataset"]["gsm8k_repo_id"]
    gsm = load_dataset(repo, "main", split="test", revision=revisions[f"dataset:{repo}"],
                       cache_dir=str(paths["data"] / "huggingface" / "datasets"))
    task = TaskManager().load(["hellaswag"])["tasks"]["hellaswag"]
    hellaswag_docs = task.eval_docs
    if len(hellaswag_docs) != 10042:
        raise RuntimeError("Stock HellaSwag preprocessing changed the expected validation row count.")
    prompts = [("gsm8k", index, gsm[index]["question"], True)
               for index in probe_ids["gsm8k_test_indices"]]
    prompts.extend(("hellaswag", index, task.doc_to_text(hellaswag_docs[index]), False)
                   for index in probe_ids["hellaswag_validation_indices"])
    return prompts


def _baseline_activations(configuration, paths, prompts, layers, device):
    model, tokenizer = load_qwen(paths["data"], "float32", device=device)
    model.eval()
    activations = []
    for dataset_name, source_index, prompt, chat in prompts:
        ids = assert_non_thinking_prompt(tokenizer, prompt) if chat else tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > configuration["model"]["max_sequence_length"]:
            raise ValueError(f"Invalid diagnostic probe length: {dataset_name} row {source_index}.")
        captured = {}
        hooks = [model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
            lambda module, inputs, layer_index=layer: captured.update({layer_index: inputs[0][0, -1].detach().float().cpu()})
        ) for layer in layers]
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device))
        for hook in hooks:
            hook.remove()
        if set(captured) != set(layers):
            raise RuntimeError("A diagnostic activation hook did not fire.")
        activations.append({"dataset": dataset_name, "source_index": source_index,
                            "prompt_tokens": len(ids), "vectors": captured})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return activations


def _starting_weights_and_bases(paths, layers):
    model, _ = load_qwen(paths["data"], "float32", device="cpu")
    weights = {layer: model.model.layers[layer].mlp.down_proj.weight.detach().float().cpu().clone()
               for layer in layers}
    del model
    gc.collect()
    bases = {}
    for layer, weight in weights.items():
        left, _, _ = torch.linalg.svd(weight, full_matrices=False)
        bases[layer] = left[:, :128].contiguous()
    return weights, bases


def _delta(condition, layer, baseline_weight, paths):
    method = "full_ft" if condition == "full_ft_mid" else condition
    update = 625 if condition == "full_ft_mid" else 1250
    checkpoint_dir = paths["results"] / "checkpoints" / method
    if method.startswith(("lora", "oplora")):
        factor_path = checkpoint_dir / "exported_projected_factors.pt"
        if not factor_path.is_file():
            raise FileNotFoundError(f"Missing exported adapter factors: {factor_path}")
        factors = torch.load(factor_path, map_location="cpu", weights_only=True)
        prefix = f"model.layers.{layer}.mlp.down_proj"
        configuration = read_configuration(method)["method"]
        return (factors[f"{prefix}.B"].float() @ factors[f"{prefix}.A"].float()) * (
            configuration["alpha"] / configuration["rank"])
    checkpoint = checkpoint_dir / f"update_{update:04d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["model_state"]
    weight = state[f"model.layers.{layer}.mlp.down_proj.weight"].float()
    return weight - baseline_weight


def run_diagnostics(device="cuda"):
    """Save all six conditions × three layers × 64 fixed activation probes."""
    configuration = read_configuration()
    paths = task_paths(configuration)
    configure_huggingface_cache(paths["data"])
    output_path = paths["results"] / "metrics" / "diagnostics.json"
    if output_path.is_file():
        return read_json(output_path)
    probe_ids = save_activation_probe_ids(configuration, paths["data"], paths["results"])
    prompts = _probe_prompts(configuration, paths, probe_ids)
    layers = configuration["evaluation"]["diagnostic_layers"]
    activations = _baseline_activations(configuration, paths, prompts, layers, device)
    baseline_weights, bases = _starting_weights_and_bases(paths, layers)
    rows = []
    for condition in DIAGNOSTIC_CONDITIONS:
        for layer in layers:
            weight = baseline_weights[layer]
            delta = _delta(condition, layer, weight, paths)
            delta_norm = float(torch.linalg.vector_norm(delta))
            overlap = float(torch.linalg.vector_norm(bases[layer].T @ delta))
            rho = (overlap / delta_norm) ** 2 if delta_norm else 0.0
            eta = delta_norm / float(torch.linalg.vector_norm(weight))
            for probe in activations:
                vector = probe["vectors"][layer]
                numerator = float(torch.linalg.vector_norm(delta @ vector))
                denominator = float(torch.linalg.vector_norm(weight @ vector)) + 1e-8
                rows.append({"condition": condition, "layer": layer,
                             "dataset": probe["dataset"], "source_index": probe["source_index"],
                             "prompt_tokens": probe["prompt_tokens"],
                             "rho": rho, "eta": eta, "response_ratio": numerator / denominator})
    result = {"probe_ids": probe_ids, "layers": layers, "conditions": DIAGNOSTIC_CONDITIONS,
              "formulae": {"rho": "||U_k^T DeltaW||_F^2 / ||DeltaW||_F^2",
                           "eta": "||DeltaW||_F / ||W0||_F",
                           "response_ratio": "||DeltaW z||_2 / (||W0 z||_2 + 1e-8)"},
              "rows": rows}
    write_json(output_path, result)
    csv_path = paths["results"] / "metrics" / "diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
