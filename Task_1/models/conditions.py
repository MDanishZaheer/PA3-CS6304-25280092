# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load each evaluation condition from the same original checkpoint."""

import time

import torch

from methods.adapters import cast_frozen_backbone_fp16, install_adapters, load_adapter_state, merge_adapters
from models.qwen import assert_non_thinking_prompt, load_qwen
from common.shared import CONDITIONS, read_configuration, task_paths


def condition_checkpoint(condition, paths):
    if condition == "baseline":
        return None
    method = "full_ft" if condition == "full_ft_mid" else condition
    update = 625 if condition == "full_ft_mid" else 1250
    checkpoint = paths["results"] / "checkpoints" / method / f"update_{update:04d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train {method} through update {update}: {checkpoint}")
    return checkpoint


def load_condition(condition, device="cuda", merge=True, verify_merge=True):
    """Return an FP16 inference model; merge adapters only after a logit check."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown evaluation condition: {condition}")
    configuration = read_configuration()
    paths = task_paths(configuration)
    is_adapter = condition.startswith(("lora", "oplora"))
    model, tokenizer = load_qwen(paths["data"], "float32" if is_adapter else "float16",
                                 device="cpu" if is_adapter else device)
    checkpoint_path = condition_checkpoint(condition, paths)
    merge_error = None
    adapter_preprocessing_seconds = 0.0
    merge_seconds = 0.0
    if checkpoint_path is not None:
        method = "full_ft" if condition == "full_ft_mid" else condition
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["configuration"] != read_configuration(method):
            raise ValueError(f"Checkpoint configuration differs from {method} YAML.")
        if method.startswith(("lora", "oplora")):
            adapter_info = install_adapters(model, checkpoint["configuration"]["method"],
                                            configuration["experiment"]["seed"])
            adapter_preprocessing_seconds = adapter_info["preprocessing_seconds"]
            cast_frozen_backbone_fp16(model)
            model.to(device)
            load_adapter_state(model, checkpoint["model_state"])
            model.eval()
            if merge:
                if verify_merge:
                    ids = torch.tensor([assert_non_thinking_prompt(tokenizer, "What is 2 + 3?")], device=device)
                    with torch.inference_mode():
                        before = model(input_ids=ids).logits[:, -1, :].float().cpu()
                merge_started = time.perf_counter()
                merge_adapters(model)
                merge_seconds = time.perf_counter() - merge_started
                if verify_merge:
                    with torch.inference_mode():
                        after = model(input_ids=ids).logits[:, -1, :].float().cpu()
                    merge_error = float((before - after).abs().max())
                    if merge_error > 0.1:
                        raise RuntimeError(f"Adapter merge changed logits by {merge_error:.4g}.")
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, {"condition": condition, "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                               "max_merge_logit_difference": merge_error,
                               "adapter_preprocessing_seconds": adapter_preprocessing_seconds,
                               "adapter_merge_seconds": merge_seconds}
