# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""AdamW with FP32 CPU moments and a FP32 GPU training model."""

import math

import torch


class CpuOffloadedAdamW(torch.optim.Optimizer):
    """Match AdamW updates while keeping two moment tensors off GPU."""

    def __init__(self, parameters, lr, betas, eps):
        super().__init__(parameters, {"lr": lr, "betas": betas, "eps": eps})
        self._masters = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta_1, beta_2 = group["betas"]
            learning_rate = group["lr"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("CPU-offloaded AdamW does not support sparse gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
                    state["exp_avg_sq"] = torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
                if parameter not in self._masters:
                    self._masters[parameter] = parameter.detach().to("cpu", torch.float32).clone()
                master = self._masters[parameter]
                gradient = parameter.grad.detach().to("cpu", torch.float32)
                state["step"] += 1
                step = state["step"]
                master.mul_(1 - learning_rate * weight_decay)
                state["exp_avg"].lerp_(gradient, 1 - beta_1)
                state["exp_avg_sq"].mul_(beta_2).addcmul_(gradient, gradient, value=1 - beta_2)
                correction_1 = 1 - beta_1 ** step
                correction_2 = math.sqrt(1 - beta_2 ** step)
                denominator = state["exp_avg_sq"].sqrt().div_(correction_2).add_(epsilon)
                master.addcdiv_(state["exp_avg"], denominator, value=-learning_rate / correction_1)
                parameter.copy_(master.to(parameter.device))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for state in self.state.values():
            state["exp_avg"] = state["exp_avg"].to("cpu", torch.float32)
            state["exp_avg_sq"] = state["exp_avg_sq"].to("cpu", torch.float32)
            state["step"] = int(state["step"])
        self._masters = {}
