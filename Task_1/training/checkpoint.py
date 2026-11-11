# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Save and restore complete optimizer-update boundaries."""

import random
from pathlib import Path

import numpy as np
import torch

from methods.adapters import adapter_state_dict, load_adapter_state


def rng_state():
    """Capture every seeded random generator used by the experiment."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    """Restore random generators before the next effective batch."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scheduler, scaler, configuration, update, examples_seen):
    """Save state only after a complete successful optimizer update."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_method = configuration["method"]["name"].startswith(("lora", "oplora"))
    model_state = adapter_state_dict(model) if adapter_method else model.state_dict()
    checkpoint = {
        "configuration": configuration,
        "update": int(update),
        "examples_seen": int(examples_seen),
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return path.stat().st_size


def load_checkpoint(path, model, optimizer, scheduler, scaler, configuration):
    """Resume only a checkpoint from the identical locked configuration."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["configuration"] != configuration:
        raise ValueError(f"Configuration mismatch in checkpoint: {path}")
    if configuration["method"]["name"].startswith(("lora", "oplora")):
        load_adapter_state(model, checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    expected_examples = checkpoint["update"] * configuration["training"]["effective_batch_size"]
    if checkpoint["examples_seen"] != expected_examples:
        raise ValueError("Checkpoint data position is inconsistent with optimizer update count.")
    return checkpoint["update"]


def latest_checkpoint(checkpoint_dir, required_updates=None, resume_dir=None):
    """Find the newest complete checkpoint, including a restored prior output."""
    candidates = []
    for priority, directory in enumerate((resume_dir, checkpoint_dir)):
        if directory is None:
            continue
        for path in Path(directory).glob("update_*.pt"):
            try:
                update = int(path.stem.removeprefix("update_"))
            except ValueError:
                continue
            candidates.append((update, priority, path))
    return max(candidates)[2] if candidates else None

