# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Fixed Task 2 configuration, paths, and reproducible result utilities."""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def read_configuration():
    path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    train = config["training"]
    stages = train["stages"]
    expected = [2250, 1500, 750]
    if [stage["updates"] for stage in stages] != expected:
        raise ValueError("Task 2 stage update budgets must be 2250, 1500, 750.")
    if (config["experiment"]["seed"], train["microbatch_size"], train["accumulation_steps"],
            train["effective_batch_size"], train["total_updates"], train["checkpoint_interval"]) != (6304, 8, 2, 16, 4500, 250):
        raise ValueError("Task 2 fixed seed, batch, update, or checkpoint settings changed.")
    if config["models"]["vision_tokens"] != 196 or config["models"]["enable_thinking"] or config["models"]["quantization"]:
        raise ValueError("The main Task 2 run requires 196 tokens, no thinking, and no quantization.")
    return config


def task_paths(config):
    root = Path(__file__).resolve().parents[2]
    data = Path(os.environ.get("ATML_TASK2_DATA_ROOT", root / config["paths"]["data_root"]))
    results = root / config["paths"]["results_root"]
    data.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "histories", "metrics", "predictions", "figures", "splits", "examples"):
        (results / folder).mkdir(parents=True, exist_ok=True)
    return {"root": root, "data": data, "results": results}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def image_key(split, image_id):
    return f"{split}:{int(image_id):012d}"


def split_image_key(key):
    split, number = key.split(":")
    if split not in ("train2014", "val2014"):
        raise ValueError(key)
    return split, int(number)
