# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Shared LoRA layers and orthogonal protection for Qwen projection weights."""

import hashlib
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


class AdapterLinear(nn.Module):
    """Freeze one linear weight and add FP32 low-rank trainable factors."""

    def __init__(self, base, rank, alpha, dropout, generator, protection_rank=0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA targets must be torch.nn.Linear modules.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.protection_rank = int(protection_rank)
        self.A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)
        self.u_basis = None
        self.v_basis = None
        if self.protection_rank:
            self.compute_bases()
        self.A.data = self.A.data.to(base.weight.device)
        self.B.data = self.B.data.to(base.weight.device)

    def compute_bases(self):
        """Compute exact leading singular bases once from the frozen FP32 weight."""
        weight = self.base.weight.detach().to(device="cpu", dtype=torch.float32)
        rank_limit = min(weight.shape)
        if self.protection_rank > rank_limit:
            raise ValueError(f"Protection rank exceeds the target weight's rank limit: {weight.shape}")
        left, _, right_transpose = torch.linalg.svd(weight, full_matrices=False)
        self.u_basis = left[:, : self.protection_rank].contiguous()
        self.v_basis = right_transpose[: self.protection_rank, :].T.contiguous()

    def effective_factors(self):
        """Project OPLoRA factors without constructing dense projection matrices."""
        if not self.protection_rank:
            return self.A, self.B
        left = self.u_basis.to(device=self.B.device, non_blocking=True)
        right = self.v_basis.to(device=self.A.device, non_blocking=True)
        effective_b = self.B - left @ (left.T @ self.B)
        effective_a = self.A - (self.A @ right) @ right.T
        return effective_a, effective_b

    def forward(self, inputs):
        base_output = self.base(inputs)
        effective_a, effective_b = self.effective_factors()
        adapter_input = self.dropout(inputs)
        intermediate = F.linear(adapter_input, effective_a.to(adapter_input.dtype))
        correction = F.linear(intermediate, effective_b.to(intermediate.dtype))
        return base_output + correction * self.scale


def adapter_modules(model):
    """Yield named adapters in stable model order."""
    for name, module in model.named_modules():
        if isinstance(module, AdapterLinear):
            yield name, module


def cast_frozen_backbone_fp16(model):
    """Keep adapter factors FP32 while storing the frozen backbone in FP16."""
    for parameter in model.parameters():
        if not parameter.requires_grad:
            parameter.data = parameter.data.half()


def install_adapters(model, method, seed):
    """Wrap every requested projection and report preprocessing overhead."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    target_names = list(method["targets"])
    if len(target_names) != len(set(target_names)):
        raise ValueError("Adapter target names must be unique and ordered.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    protection_rank = int(method.get("protection_rank", 0))
    target_count, basis_bytes = 0, 0
    started = time.perf_counter()
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for target_name in target_names:
                if not hasattr(block, target_name):
                    continue
                base = getattr(block, target_name)
                wrapped = AdapterLinear(
                    base, method["rank"], method["alpha"], method["dropout"],
                    generator, protection_rank,
                )
                setattr(block, target_name, wrapped)
                target_count += 1
                if protection_rank:
                    basis_bytes += (wrapped.u_basis.numel() + wrapped.v_basis.numel()) * 4
    expected_count = len(model.model.layers) * len(target_names)
    if target_count != expected_count:
        raise RuntimeError(f"Wrapped {target_count} projections; expected {expected_count}.")
    digest = hashlib.sha256()
    for name, module in adapter_modules(model):
        digest.update(name.encode("utf-8"))
        digest.update(module.A.detach().cpu().contiguous().numpy().tobytes())
        digest.update(module.B.detach().cpu().contiguous().numpy().tobytes())
    return {
        "target_count": target_count,
        "basis_storage_bytes": basis_bytes,
        "preprocessing_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_factor_sha256": digest.hexdigest(),
    }


def adapter_state_dict(model, projected=False):
    """Return only adapter factors, optionally exported after projection."""
    state = {}
    for name, module in adapter_modules(model):
        effective_a, effective_b = module.effective_factors() if projected else (module.A, module.B)
        state[f"{name}.A"] = effective_a.detach().cpu().contiguous()
        state[f"{name}.B"] = effective_b.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No adapters found in the model.")
    return state


def load_adapter_state(model, state):
    """Copy saved factors into a matching adapter architecture."""
    expected = set()
    with torch.no_grad():
        for name, module in adapter_modules(model):
            for factor_name in ("A", "B"):
                key = f"{name}.{factor_name}"
                expected.add(key)
                if key not in state:
                    raise KeyError(f"Missing adapter factor: {key}")
                target = getattr(module, factor_name)
                if target.shape != state[key].shape:
                    raise ValueError(f"Shape mismatch for {key}.")
                target.copy_(state[key].to(target.device))
    if set(state) != expected:
        raise ValueError("Adapter checkpoint has unexpected or missing factors.")


def merge_adapters(model):
    """Add each correction to a fresh frozen backbone in inference precision."""
    merged = 0
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name, module in list(block.named_children()):
                if isinstance(module, AdapterLinear):
                    effective_a, effective_b = module.effective_factors()
                    delta = (effective_b @ effective_a) * module.scale
                    module.base.weight.data.add_(delta.to(module.base.weight.dtype))
                    setattr(block, name, module.base)
                    merged += 1
    if merged == 0:
        raise RuntimeError("No adapters were merged.")
    return merged


def check_projection(model):
    """Measure OPLoRA constraint residuals in FP32."""
    maximum_left, maximum_right = 0.0, 0.0
    for _, module in adapter_modules(model):
        if not module.protection_rank:
            continue
        effective_a, effective_b = module.effective_factors()
        left = module.u_basis.to(effective_b.device)
        right = module.v_basis.to(effective_a.device)
        maximum_left = max(maximum_left, float((left.T @ effective_b).norm()))
        maximum_right = max(maximum_right, float((effective_a @ right).norm()))
    return {"max_left_residual": maximum_left, "max_right_residual": maximum_right}
