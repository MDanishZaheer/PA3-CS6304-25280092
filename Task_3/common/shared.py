# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Fixed Task 3 configuration, paths, reproducibility, and atomic result files."""

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml


def task_root():
    return Path(__file__).resolve().parents[1]


def read_configuration():
    with (task_root() / "configs" / "base.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    experiment = config["experiment"]
    expected = {"seed": 6304, "budgets": [1, 2, 4, 8, 16, 32], "orders": 10,
                "development_questions": 200, "evaluation_questions": 800,
                "live_questions": 50, "bootstrap_resamples": 1000}
    for key, value in expected.items():
        if experiment[key] != value:
            raise ValueError(f"Assignment requires experiment.{key}={value!r}.")
    model = config["model"]
    if (model["name"] != "Qwen/Qwen2.5-3B-Instruct" or model["dtype"] != "fp16"
            or model["batch_size"] != 1 or model["quantization"]):
        raise ValueError("Live model settings differ from Task 3 requirements.")
    if [model[k] for k in ("single_new_tokens", "parallel_new_tokens", "revision_new_tokens")] != [3072, 1024, 2048]:
        raise ValueError("Live output allowances differ from Task 3 requirements.")
    expected_sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                         "repetition_penalty": 1.0}
    if any(model[key] != value for key, value in expected_sampling.items()):
        raise ValueError("Live sampling settings differ from Task 3 requirements.")
    return config


def paths(config, data_override=None, results_override=None):
    root = task_root().parent
    data = Path(data_override or os.environ.get("ATML_TASK3_DATA", root / config["paths"]["data_root"]))
    results = Path(results_override or os.environ.get("ATML_TASK3_RESULTS", root / config["paths"]["results_root"]))
    for folder in ("splits", "progress", "predictions", "metrics", "tables", "figures"):
        (results / folder).mkdir(parents=True, exist_ok=True)
    return {"root": task_root(), "data": data, "results": results}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)
        temporary = Path(stream.name)
    temporary.replace(path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint():
    """Identify the Python and YAML protocol that produced saved results."""
    root = task_root()
    digest = hashlib.sha256()
    for path in sorted([*root.rglob("*.py"), *root.rglob("*.yaml")]):
        if path.relative_to(root).parts[0] not in ("common", "configs", "data", "methods", "models", "evaluation"):
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_frame(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def require_result(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved Task 3 result is missing: {path}. Copy the complete Kaggle Task_3/results folder here.")
    return path
