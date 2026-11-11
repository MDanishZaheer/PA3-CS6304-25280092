# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Run all five methods with one fixed, response-normalized SFT loop."""

import csv
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from data.collator import ResponseOnlyCollator
from data.prepare import load_training_examples
from methods.adapters import adapter_state_dict, cast_frozen_backbone_fp16, check_projection, install_adapters
from models.qwen import load_qwen
from common.shared import append_csv, read_configuration, read_json, seed_everything, task_paths, write_json
from training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from training.offloaded_adamw import CpuOffloadedAdamW


def unique_named_parameters(model):
    """Iterate tied parameters once, including biases and normalization weights."""
    return list(model.named_parameters(remove_duplicate=True))


def optimizer_groups(model, weight_decay):
    """Exclude biases and all rank-one parameters from AdamW decay."""
    decay, no_decay = [], []
    for name, parameter in unique_named_parameters(model):
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or parameter.ndim < 2 else decay).append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("The selected method has no trainable parameters.")
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_multiplier(update, warmup_updates, total_updates):
    """Linearly warm up, then cosine-decay the peak rate to zero."""
    if update < warmup_updates:
        return (update + 1) / warmup_updates
    progress = (update - warmup_updates + 1) / (total_updates - warmup_updates)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def starting_parameters(model):
    """Copy the unique starting weights to CPU FP32 for L2-SP and displacement."""
    return {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, parameter in unique_named_parameters(model)
    }


def add_l2_sp_gradient(model, reference, coefficient):
    """Add exact unaveraged L2-SP gradients after AMP unscaling."""
    omega, transfer_seconds = 0.0, 0.0
    for name, parameter in unique_named_parameters(model):
        started = time.perf_counter()
        starting_value = reference[name].to(parameter.device)
        transfer_seconds += time.perf_counter() - started
        difference = parameter.detach().float() - starting_value
        omega += 0.5 * float(difference.square().sum().item())
        if parameter.grad is None:
            parameter.grad = difference.mul(coefficient)
        else:
            parameter.grad.add_(difference, alpha=coefficient)
    return omega, transfer_seconds


def parameter_displacement(model, reference):
    """Return the full-model L2 displacement from the fixed starting point."""
    squared = 0.0
    with torch.no_grad():
        for name, parameter in unique_named_parameters(model):
            difference = parameter.detach().float().cpu() - reference[name]
            squared += float(difference.square().sum())
    return math.sqrt(squared)


def supervised_loss_sum(model, batch):
    """Sum next-token cross-entropy only over assistant and end-turn tokens."""
    hidden = model.model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"], use_cache=False).last_hidden_state
    shifted_labels = batch["labels"][:, 1:]

    def chunk_loss(hidden_chunk, label_chunk):
        logits = model.lm_head(hidden_chunk)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               label_chunk.reshape(-1), ignore_index=-100, reduction="sum")

    losses = []
    for start in range(0, shifted_labels.size(1), 128):
        labels = shifted_labels[:, start:start + 128]
        if not bool((labels != -100).any()):
            continue
        chunk_hidden = hidden[:, start:start + labels.size(1), :]
        losses.append(checkpoint(chunk_loss, chunk_hidden, labels, use_reentrant=False))
    if not losses:
        raise RuntimeError("A supervised microbatch has no assistant response tokens.")
    return sum(losses)


def trim_history(path, completed_updates):
    """Discard rows written after the last durable checkpoint on resume."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if int(row["update"]) <= completed_updates]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_method(method_name, device="cuda", target_update=None, deadline=None,
                 checkpoint_reserve_seconds=900, resume_checkpoint_dir=None,
                 latest_only=False, checkpoint_callback=None):
    """Train to a requested update, or stop early with a resumable checkpoint."""
    configuration = read_configuration(method_name)
    paths = task_paths(configuration)
    seed_everything(configuration["experiment"]["seed"])
    training = configuration["training"]
    method = configuration["method"]
    target_update = training["optimizer_updates"] if target_update is None else int(target_update)
    if not 1 <= target_update <= training["optimizer_updates"]:
        raise ValueError(f"target_update must be between 1 and {training['optimizer_updates']}.")
    if latest_only and deadline is None:
        raise ValueError("Checkpoint rotation requires a session deadline.")
    dataset = load_training_examples(paths["data"], configuration["dataset"]["eligible_examples"])
    adapter_method = method_name.startswith(("lora", "oplora"))
    model, tokenizer = load_qwen(
        paths["data"], "float32", device="cpu" if adapter_method else device,
        gradient_checkpointing=True,
    )
    adapter_resources = install_adapters(model, method, configuration["experiment"]["seed"]) if adapter_method else {}
    if method_name == "oplora_r16":
        lora_resources = read_json(paths["results"] / "metrics" / "lora_r16_resources.json")
        if adapter_resources["initial_factor_sha256"] != lora_resources["initial_factor_sha256"]:
            raise RuntimeError("LoRA16 and OPLoRA16 initial unprojected factors differ.")
    if adapter_method:
        cast_frozen_backbone_fp16(model)
        model.to(device)
    reference = starting_parameters(model) if not adapter_method else None
    reference_bytes = sum(value.numel() * 4 for value in reference.values()) if reference else 0
    collator = ResponseOnlyCollator(tokenizer.pad_token_id)
    optimizer_class = torch.optim.AdamW if adapter_method else CpuOffloadedAdamW
    optimizer = optimizer_class(
        optimizer_groups(model, training["weight_decay"]),
        lr=method["peak_learning_rate"], betas=tuple(training["adam_betas"]),
        eps=training["adam_epsilon"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: cosine_multiplier(update, training["warmup_updates"], training["optimizer_updates"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    checkpoint_dir = paths["results"] / "checkpoints" / method_name
    resume_path = latest_checkpoint(checkpoint_dir, training["checkpoint_updates"], resume_checkpoint_dir)
    completed_updates = load_checkpoint(resume_path, model, optimizer, scheduler, scaler, configuration) if resume_path else 0
    starting_update = completed_updates
    if not adapter_method and completed_updates in (625, 1250):
        displacement_path = paths["results"] / "metrics" / f"{method_name}_displacement_{completed_updates}.json"
        if not displacement_path.is_file():
            write_json(displacement_path, {"update": completed_updates,
                                           "l2_displacement": parameter_displacement(model, reference)})
    history_path = paths["results"] / "histories" / f"{method_name}.csv"
    skip_path = paths["results"] / "histories" / f"{method_name}_skipped_updates.csv"
    resources_path = paths["results"] / "metrics" / f"{method_name}_resources.json"
    export_path = checkpoint_dir / "exported_projected_factors.pt"
    if completed_updates >= target_update and resources_path.is_file():
        return read_json(resources_path)
    trim_history(history_path, completed_updates)
    trim_history(skip_path, completed_updates)
    prior_history = []
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            prior_history = list(csv.DictReader(stream))
    prior_seconds = float(prior_history[-1]["elapsed_seconds"]) if prior_history else 0.0
    prior_skips = int(prior_history[-1]["skipped_attempts_cumulative"]) if prior_history else 0
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    skipped_attempts = prior_skips
    reference_transfer_seconds = float(prior_history[-1].get("reference_transfer_seconds", 0)) if prior_history else 0.0
    prior_resources = read_json(resources_path) if resources_path.is_file() else {}
    checkpoint_sizes = dict(prior_resources.get("checkpoint_sizes_bytes", {}))
    if resume_path is not None:
        checkpoint_sizes[str(completed_updates)] = resume_path.stat().st_size
    stopped_for_time = False
    print(f"{method_name}: training from update {completed_updates} to {target_update}", flush=True)
    for update in range(completed_updates, target_update):
        if deadline is not None and time.monotonic() >= deadline - checkpoint_reserve_seconds:
            stopped_for_time = True
            break
        start_index = update * training["effective_batch_size"]
        examples = [dataset[index] for index in range(start_index, start_index + training["effective_batch_size"])]
        supervised_tokens = sum(int(item["valid_token_count"]) for item in examples)
        if supervised_tokens <= 0:
            raise RuntimeError(f"No supervised response tokens at update {update + 1}.")
        attempts = 0
        while True:
            attempts += 1
            if attempts > 10:
                raise RuntimeError(f"Repeated AMP overflow at update {update + 1}.")
            optimizer.zero_grad(set_to_none=True)
            sft_loss_sum = 0.0
            for offset in range(0, len(examples), training["microbatch_size"]):
                batch = collator(examples[offset : offset + training["microbatch_size"]])
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_sum = supervised_loss_sum(model, batch)
                scaler.scale(loss_sum / supervised_tokens).backward()
                sft_loss_sum += float(loss_sum.detach())
            scaler.unscale_(optimizer)
            omega = 0.0
            if method_name == "full_ft_l2sp":
                omega, transfer = add_l2_sp_gradient(model, reference, method["l2_sp_lambda"])
                reference_transfer_seconds += transfer
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, training["max_gradient_norm"])
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_attempts += 1
                append_csv(skip_path, {"update": update + 1, "attempt": attempts, "scale": old_scale})
                continue
            break
        used_learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        completed = update + 1
        completed_updates = completed
        sft_loss = sft_loss_sum / supervised_tokens
        append_csv(history_path, {
            "update": completed, "examples_seen": completed * training["effective_batch_size"],
            "supervised_tokens": supervised_tokens, "sft_loss": sft_loss,
            "l2_sp_omega": omega, "l2_sp_penalty": method.get("l2_sp_lambda", 0.0) * omega,
            "total_loss": sft_loss + method.get("l2_sp_lambda", 0.0) * omega,
            "learning_rate": used_learning_rate,
            "gradient_norm_before_clipping": float(gradient_norm),
            "skipped_attempts_cumulative": skipped_attempts,
            "elapsed_seconds": prior_seconds + time.perf_counter() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "reference_transfer_seconds": reference_transfer_seconds,
        })
        if completed == starting_update + 1 or completed % 25 == 0 or completed == target_update:
            elapsed = time.perf_counter() - started
            print(f"{method_name}: update {completed}/{target_update}, "
                  f"loss {sft_loss:.4f}, elapsed {elapsed / 3600:.2f} h", flush=True)
        if completed in training["checkpoint_updates"] and not latest_only:
            checkpoint_path = checkpoint_dir / f"update_{completed:04d}.pt"
            checkpoint_sizes[str(completed)] = save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, configuration,
                completed, completed * training["effective_batch_size"],
            )
            print(f"{method_name}: saved {checkpoint_path.name}", flush=True)
        if completed in training["checkpoint_updates"] and not adapter_method and (
                method_name == "full_ft_l2sp" or completed in (625, 1250)):
            displacement_path = paths["results"] / "metrics" / f"{method_name}_displacement_{completed}.json"
            write_json(displacement_path, {"update": completed, "l2_displacement": parameter_displacement(model, reference)})
        if completed in training["checkpoint_updates"] and not latest_only and checkpoint_callback:
            checkpoint_callback(checkpoint_path)

    if (latest_only or stopped_for_time) and completed_updates > starting_update:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if latest_only:
            for old_checkpoint in checkpoint_dir.glob("update_*.pt"):
                old_checkpoint.unlink()
        checkpoint_path = checkpoint_dir / f"update_{completed_updates:04d}.pt"
        if not checkpoint_path.is_file():
            checkpoint_sizes[str(completed_updates)] = save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, scaler, configuration,
                completed_updates, completed_updates * training["effective_batch_size"],
            )
            print(f"{method_name}: saved recovery checkpoint {checkpoint_path.name}", flush=True)
            if checkpoint_callback:
                checkpoint_callback(checkpoint_path)

    exported_size, export_seconds = 0, 0.0
    if adapter_method and completed_updates == training["optimizer_updates"]:
        export_started = time.perf_counter()
        export_path = checkpoint_dir / "exported_projected_factors.pt"
        torch.save(adapter_state_dict(model, projected=True), export_path)
        exported_size = export_path.stat().st_size
        export_seconds = time.perf_counter() - export_started
    resources = {
        "method": method_name,
        "completed_updates": completed_updates,
        "target_update": target_update,
        "stopped_for_time": stopped_for_time,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": prior_seconds + time.perf_counter() - started,
        "peak_gpu_memory_bytes": max([torch.cuda.max_memory_allocated()] +
                                     [int(row.get("peak_gpu_memory_bytes", 0)) for row in prior_history]),
        "checkpoint_sizes_bytes": checkpoint_sizes,
        "skipped_update_attempts": skipped_attempts,
        "reference_storage_bytes": reference_bytes,
        "optimizer_moment_bytes": sum(
            state[key].numel() * state[key].element_size()
            for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq") if key in state),
        "cpu_master_bytes": sum(value.numel() * value.element_size()
                                for value in getattr(optimizer, "_masters", {}).values()),
        "reference_transfer_seconds": reference_transfer_seconds,
        "adapter_export_bytes": exported_size,
        "adapter_export_seconds": export_seconds,
        "optimizer_state_device": "cuda" if adapter_method else "cpu",
        **adapter_resources,
    }
    if method_name == "oplora_r16":
        resources.update(check_projection(model))
    write_json(paths["results"] / "metrics" / f"{method_name}_resources.json", resources)
    write_json(paths["results"] / "metrics" / f"{method_name}_config.json", configuration)
    print(f"{method_name}: finished at update {completed_updates}; "
          f"stopped_for_time={stopped_for_time}", flush=True)
    return resources
