# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Configuration, reproducibility, and small result-file utilities."""

import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


METHODS = ("full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16")
CONDITIONS = ("baseline", "full_ft_mid", "full_ft", "full_ft_l2sp", "lora_r4", "lora_r16", "oplora_r16")


def project_root():
    """Locate the assignment root from this module rather than the notebook CWD."""
    return Path(__file__).resolve().parents[2]


def read_configuration(method_name=None):
    """Merge the fixed base settings with exactly one method configuration."""
    config_root = Path(__file__).resolve().parents[1] / "configs"
    with (config_root / "base.yaml").open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    if method_name is not None:
        if method_name not in METHODS:
            raise ValueError(f"Unknown Task 1 method: {method_name}")
        with (config_root / f"{method_name}.yaml").open("r", encoding="utf-8") as stream:
            changes = yaml.safe_load(stream)
        configuration = deep_merge(configuration, changes)
        validate_configuration(configuration)
    return configuration


def deep_merge(base, changes):
    """Copy nested settings and replace only explicitly changed values."""
    merged = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_configuration(configuration):
    """Reject changes to the assignment's fixed Task 1 protocol."""
    training = configuration["training"]
    method = configuration["method"]
    expected = {
        "epochs": 1, "microbatch_size": 2, "gradient_accumulation_steps": 16,
        "effective_batch_size": 32, "optimizer_updates": 1250,
        "checkpoint_updates": [312, 625, 937, 1250], "warmup_updates": 38,
    }
    for key, value in expected.items():
        if training[key] != value:
            raise ValueError(f"Task 1 requires training.{key}={value!r}.")
    if configuration["experiment"]["seed"] != 6304:
        raise ValueError("Task 1 requires seed 6304.")
    if configuration["model"]["enable_thinking"] or configuration["model"]["quantization"]:
        raise ValueError("Thinking and quantization must both be disabled.")
    if method["name"] in ("full_ft", "full_ft_l2sp"):
        if method["peak_learning_rate"] != 2e-5:
            raise ValueError("Full FT peak learning rate must be 2e-5.")
    else:
        if method["peak_learning_rate"] != 2e-4:
            raise ValueError("Adapter peak learning rate must be 2e-4.")
        if method["alpha"] != 2 * method["rank"] or method["dropout"] != 0.05:
            raise ValueError("Adapter alpha and dropout differ from the protocol.")
    if method["name"] == "full_ft_l2sp" and method["l2_sp_lambda"] != 1e-3:
        raise ValueError("L2-SP lambda must be 1e-3.")
    if method["name"] == "oplora_r16" and method["protection_rank"] != 128:
        raise ValueError("OPLoRA protection rank must be 128.")
    return configuration


def task_paths(configuration):
    """Resolve configured storage paths under the assignment root."""
    root = project_root()
    data_root = Path(os.environ.get("ATML_DATA_ROOT", root / configuration["paths"]["data_root"]))
    results_root = root / configuration["paths"]["results_root"]
    data_root.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "histories", "metrics", "predictions", "annotations", "figures"):
        (results_root / folder).mkdir(parents=True, exist_ok=True)
    return {"root": root, "data": data_root, "results": results_root}


def configure_huggingface_cache(data_root):
    """Keep model, dataset, and harness caches inside the ignored Data folder."""
    cache_root = Path(data_root) / "huggingface"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    os.environ["LM_HARNESS_CACHE_PATH"] = str(cache_root / "lm_eval")


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch, and CUDA RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    """Write a UTF-8 JSON result atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    """Load a UTF-8 JSON result with a clear missing-file error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv(path, row):
    """Append a named metric row while preserving a stable CSV header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not existing:
            writer.writeheader()
        writer.writerow(row)
