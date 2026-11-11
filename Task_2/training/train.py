# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Train the fixed three-stage visual curriculum with resumable updates."""

import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from common.shared import append_csv, read_json, write_json
from models.vlm import GroundedQwen


def stage_bounds(config):
    start = 0
    bounds = {}
    for stage in config["training"]["stages"]:
        bounds[stage["name"]] = (start, start + stage["updates"], stage)
        start += stage["updates"]
    return bounds


def latest_checkpoint(results_root):
    checkpoints = sorted((Path(results_root) / "checkpoints").glob("update_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _rng_state(sampler):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "sampler": sampler.getstate()}


def _restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    sampler.setstate(state["sampler"])


def _selection_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrainingSession:
    """Own one Qwen model and continue from the last complete optimizer update."""

    def __init__(self, config, paths, selection, store, language_path):
        if not torch.cuda.is_available():
            raise RuntimeError("Task 2 training requires a CUDA GPU.")
        self.config, self.paths, self.selection, self.store = config, paths, selection, store
        self.bounds = stage_bounds(config)
        self.sampler = random.Random(config["experiment"]["seed"] + 200)
        self.model = GroundedQwen(language_path, config)
        self.update = 0
        self.skipped_attempts = 0
        self.stage_mixture = {"presence": 0, "counting": 0, "caption": 0, "natural": 0}
        self.stage_draws = {name: {kind: 0 for kind in self.stage_mixture} for name in self.bounds}
        self.stage_seconds = {name: 0.0 for name in self.bounds}
        self.stage_peak_gib = {name: 0.0 for name in self.bounds}
        self.caption_orders = [np.random.default_rng(config["experiment"]["seed"] + pass_index)
                               .permutation(len(selection["caption_train"])) for pass_index in range(2)]
        self.pools = {
            "natural": selection["natural_train"],
            "presence": [pair for pair in selection["grounding_train"] if pair["kind"] == "presence"],
            "counting": [pair for pair in selection["grounding_train"] if pair["kind"] == "counting"],
        }
        self.selection_hash = _selection_digest(paths["results"] / "splits" / "selection.json")
        self.resume_payload = None
        checkpoint = latest_checkpoint(paths["results"])
        if checkpoint is None:
            train_keys = selection["image_keys"]["train2014"][:256]
            sample = store.get(train_keys)
            calibration = self.model.calibrate_connector(sample)
            write_json(paths["results"] / "metrics" / "connector_calibration.json", calibration)
        else:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["configuration"] != config or payload["selection_sha256"] != self.selection_hash:
                raise RuntimeError("Checkpoint recipe or fixed data selection changed.")
            self.update = payload["update"]
            if payload["adapter_state"] is not None:
                self.model.enable_adapters()
                set_peft_model_state_dict(self.model.language, payload["adapter_state"])
            self.model.connector.load_state_dict(payload["connector_state"])
            self.skipped_attempts = payload["skipped_attempts"]
            self.stage_mixture = payload["stage_mixture"]
            self.stage_draws.update(payload.get("stage_draws", {}))
            self.stage_seconds.update(payload.get("stage_seconds", {}))
            self.stage_peak_gib.update(payload.get("stage_peak_gib", {}))
            _restore_rng(payload["rng"], self.sampler)
            self.resume_payload = payload
            print(f"Restored Task 2 update {self.update} from {checkpoint}", flush=True)

    def check_interface(self):
        """Exercise the visual path without advancing an optimizer update."""
        examples = self.selection["caption_train"][:2]
        features = self.store.get([item["image_key"] for item in examples])
        if features.shape != (2, 196, 768) or features.dtype != torch.float16:
            raise RuntimeError("Frozen SigLIP features have the wrong shape or dtype.")
        if any(parameter.requires_grad for name, parameter in self.model.language.named_parameters()
               if "lora_" not in name):
            raise RuntimeError("Original Qwen parameters must remain frozen.")
        batch = self.model.build_batch(examples, features)
        valid_labels = (batch["labels"] != -100).sum(dim=1).tolist()
        if not all(count > 1 for count in valid_labels):
            raise RuntimeError("Missing caption or end-of-turn training labels.")
        if not all(batch["labels"][index, :196].eq(-100).all().item() for index in range(2)):
            raise RuntimeError("Prompt or image tokens are unexpectedly supervised.")
        self.model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = self.model.supervised_loss(examples, features)
        loss.backward()
        gradient = self.model.connector.layers[0].weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient.norm().item() <= 0:
            raise RuntimeError("Connector receives no finite gradient through frozen Qwen.")
        report = {"feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
                  "connector_gradient_norm": float(gradient.norm()), "supervised_tokens": valid_labels,
                  "original_qwen_trainable_parameters": 0,
                  "connector_scale": float(self.model.connector.output_scale)}
        self.model.zero_grad(set_to_none=True)
        write_json(self.paths["results"] / "metrics" / "interface_diagnostic.json", report)
        return report

    def _caption_batch(self, update):
        """Two seeded passes, each consuming all 18,000 captions exactly once."""
        records = self.selection["caption_train"]
        if len(records) != 18000:
            raise RuntimeError("Stage 1 requires exactly 18,000 caption examples.")
        examples_per_update = self.config["training"]["effective_batch_size"]
        pass_index, update_within = divmod(update, len(records) // examples_per_update)
        order = self.caption_orders[pass_index]
        indices = order[update_within * examples_per_update:(update_within + 1) * examples_per_update]
        if len(indices) != examples_per_update:
            raise RuntimeError("Caption sampler crossed a pass boundary.")
        return [records[int(index)] for index in indices]

    def _paired_batch(self, stage):
        probabilities = {"instruction": (("presence", 0.5), ("counting", 0.25), ("caption", 0.25)),
                         "natural": (("natural", 0.7), ("presence", 0.1),
                                     ("counting", 0.1), ("caption", 0.1))}[stage]
        kinds, weights = zip(*probabilities)
        examples = []
        for _ in range(self.config["training"]["effective_batch_size"] // 2):
            kind = self.sampler.choices(kinds, weights=weights, k=1)[0]
            self.stage_mixture[kind] += 1
            self.stage_draws[stage][kind] += 1
            if kind == "caption":
                examples.extend(self.sampler.choices(self.selection["caption_train"], k=2))
            else:
                examples.extend(self.sampler.choice(self.pools[kind])["examples"])
        return examples

    @staticmethod
    def _lr_scale(index, total, warmup_fraction, final_fraction):
        warmup = max(1, math.ceil(total * warmup_fraction))
        if index < warmup:
            return (index + 1) / warmup
        progress = (index - warmup) / max(1, total - warmup - 1)
        return final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def _optimizer(self, stage):
        train = self.config["training"]
        connector = [parameter for parameter in self.model.connector.parameters() if parameter.requires_grad]
        groups = [{"params": connector, "lr": stage["connector_lr"]}]
        if stage["adapter_lr"] is not None:
            adapters = [parameter for parameter in self.model.language.parameters() if parameter.requires_grad]
            if not adapters or any(parameter.dtype != torch.float32 for parameter in adapters):
                raise RuntimeError("Stage 2-3 require FP32 trainable LoRA parameters.")
            groups.append({"params": adapters, "lr": stage["adapter_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=train["weight_decay"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda index: self._lr_scale(index, stage["updates"],
                                                    train["warmup_fraction"], train["final_lr_fraction"])
        )
        scaler = torch.amp.GradScaler("cuda", enabled=train["dynamic_gradient_scaling"])
        return optimizer, scheduler, scaler

    def _save_checkpoint(self, stage_name, optimizer, scheduler, scaler, callback=None):
        path = self.paths["results"] / "checkpoints" / f"update_{self.update:04d}.pt"
        payload = {"update": self.update, "stage": stage_name, "configuration": self.config,
                   "selection_sha256": self.selection_hash,
                   "caption_cursor": min(self.update * 16, 36000),
                   "connector_state": {key: value.detach().cpu() for key, value in self.model.connector.state_dict().items()},
                   "adapter_state": (get_peft_model_state_dict(self.model.language) if self.model.adapters_enabled else None),
                   "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                   "scaler_state": scaler.state_dict(), "rng": _rng_state(self.sampler),
                   "skipped_attempts": self.skipped_attempts, "stage_mixture": self.stage_mixture.copy(),
                   "stage_draws": {name: values.copy() for name, values in self.stage_draws.items()},
                   "stage_seconds": self.stage_seconds.copy(), "stage_peak_gib": self.stage_peak_gib.copy()}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        if callback:
            callback(path)
        print(f"Saved Task 2 checkpoint at update {self.update}", flush=True)

    def _export_stage(self, stage_name, callback=None):
        output = self.paths["results"] / "checkpoints" / f"{stage_name}_weights.pt"
        temporary = output.with_suffix(".pt.tmp")
        torch.save({"update": self.update, "connector": self.model.connector.state_dict(),
                    "adapter": get_peft_model_state_dict(self.model.language) if self.model.adapters_enabled else None},
                   temporary)
        temporary.replace(output)
        if callback:
            callback(output)

    def _resource_report(self, stage_name, end):
        stage_export = self.paths["results"] / "checkpoints" / f"{stage_name}_weights.pt"
        report = {"stage": stage_name, "end_update": end,
                  "seconds_all_sessions": self.stage_seconds[stage_name],
                  "peak_gpu_gib_all_sessions": self.stage_peak_gib[stage_name],
                  "stage_export_mib": stage_export.stat().st_size / 2 ** 20 if stage_export.is_file() else None,
                  "trainable_parameters": sum(parameter.numel() for parameter in self.model.trainable_parameters()),
                  "mixture_draw_counts": self.stage_draws[stage_name].copy(),
                  "skipped_attempts_cumulative": self.skipped_attempts}
        write_json(self.paths["results"] / "metrics" / f"{stage_name}_resources.json", report)
        return report

    def run_stage(self, stage_name, development_callback=None, checkpoint_callback=None,
                  result_callback=None, deadline=None):
        """Run one named assignment stage; completed updates are never repeated."""
        start, end, stage = self.bounds[stage_name]
        if self.update >= end:
            if self.update > end and not (self.paths["results"] / "checkpoints" /
                                          f"{stage_name}_weights.pt").is_file():
                raise RuntimeError(f"Missing {stage_name} stage export; restore it from Drive.")
            if self.update == end and development_callback:
                development_callback(self.model, end)
                if result_callback:
                    result_callback()
            if self.update == end and not (self.paths["results"] / "checkpoints" /
                                           f"{stage_name}_weights.pt").is_file():
                self._export_stage(stage_name, checkpoint_callback)
            if self.update == end and not (self.paths["results"] / "metrics" /
                                           f"{stage_name}_resources.json").is_file():
                self._resource_report(stage_name, end)
                if result_callback:
                    result_callback()
            print(f"{stage_name} already complete through update {end}.")
            return {"stage": stage_name, "completed_updates": self.update}
        if self.update < start:
            raise RuntimeError(f"Finish prior stages before {stage_name}.")
        if stage_name != "caption":
            self.model.enable_adapters()
        optimizer, scheduler, scaler = self._optimizer(stage)
        if self.resume_payload and self.resume_payload["stage"] == stage_name and self.update < end:
            optimizer.load_state_dict(self.resume_payload["optimizer_state"])
            scheduler.load_state_dict(self.resume_payload["scheduler_state"])
            scaler.load_state_dict(self.resume_payload["scaler_state"])
        self.resume_payload = None
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        last_runtime_record = started
        if self.update == 0 and development_callback:
            development_callback(self.model, 0)
        elif development_callback and (self.update % self.config["evaluation"]["development_interval"] == 0
                                       or self.update in (2250, 3750)):
            development_callback(self.model, self.update)
            if result_callback:
                result_callback()
        loop_started = time.perf_counter()
        first_update_this_session = self.update
        while self.update < end:
            if deadline and time.monotonic() >= deadline - 1800:
                now = time.perf_counter()
                self.stage_seconds[stage_name] += now - last_runtime_record
                self.stage_peak_gib[stage_name] = max(self.stage_peak_gib[stage_name],
                                                      torch.cuda.max_memory_allocated() / 2 ** 30)
                self._save_checkpoint(stage_name, optimizer, scheduler, scaler, checkpoint_callback)
                if result_callback:
                    result_callback()
                raise RuntimeError(f"Colab time reserve reached at update {self.update}; resume in a new session.")
            examples = (self._caption_batch(self.update) if stage_name == "caption" else
                        self._paired_batch(stage_name))
            while True:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for first in range(0, 16, 8):
                    micro = examples[first:first + 8]
                    features = self.store.get([item["image_key"] for item in micro])
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss, _ = self.model.supervised_loss(micro, features)
                    scaler.scale(loss / 2).backward()
                    losses.append(float(loss.detach()))
                scaler.unscale_(optimizer)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(),
                                                                      self.config["training"]["gradient_clip"]))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    break
                self.skipped_attempts += 1
                print(f"Skipped scaled optimizer attempt at update {self.update + 1}; retrying same batch.", flush=True)
            rates_used = [group["lr"] for group in optimizer.param_groups]
            scheduler.step()
            self.update += 1
            append_csv(self.paths["results"] / "histories" / "training.csv",
                       {"update": self.update, "stage": stage_name, "sft_loss": sum(losses) / 2,
                        "connector_lr": rates_used[0],
                        "adapter_lr": rates_used[1] if len(rates_used) > 1 else "",
                        "gradient_norm": gradient_norm, "skipped_attempts": self.skipped_attempts,
                        "presentations_successful": self.update * 16})
            if self.update == 1 or self.update % 25 == 0:
                print(f"{stage_name}: update {self.update}/{end}, loss {sum(losses) / 2:.4f}", flush=True)
            if self.update % self.config["training"]["checkpoint_interval"] == 0:
                now = time.perf_counter()
                self.stage_seconds[stage_name] += now - last_runtime_record
                last_runtime_record = now
                self.stage_peak_gib[stage_name] = max(self.stage_peak_gib[stage_name],
                                                      torch.cuda.max_memory_allocated() / 2 ** 30)
                self._save_checkpoint(stage_name, optimizer, scheduler, scaler, checkpoint_callback)
                if result_callback:
                    result_callback()
            if development_callback and (self.update % self.config["evaluation"]["development_interval"] == 0
                                         or self.update == end):
                self.model.eval()
                development_callback(self.model, self.update)
                self.model.train()
                if result_callback:
                    result_callback()
            updates_this_session = self.update - first_update_this_session
            if updates_this_session >= 50 and self.update % 100 == 0:
                seconds_per_update = (time.perf_counter() - loop_started) / updates_this_session
                stage_hours_left = (end - self.update) * seconds_per_update / 3600
                print(f"{stage_name}: {seconds_per_update:.2f} s/update this session; "
                      f"stage ETA {stage_hours_left:.1f} h", flush=True)
        self.stage_seconds[stage_name] += time.perf_counter() - last_runtime_record
        self.stage_peak_gib[stage_name] = max(self.stage_peak_gib[stage_name],
                                              torch.cuda.max_memory_allocated() / 2 ** 30)
        self._export_stage(stage_name, checkpoint_callback)
        report = self._resource_report(stage_name, end)
        if result_callback:
            result_callback()
        return report
