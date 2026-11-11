# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check CPU-moment updates and checkpoint reload against PyTorch AdamW."""

import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.offloaded_adamw import CpuOffloadedAdamW


class OffloadedAdamWTest(unittest.TestCase):
    def test_updates_and_resume(self):
        starting = torch.tensor([[0.4, -0.2], [1.1, 0.7]], dtype=torch.float32)
        reference = torch.nn.Parameter(starting.clone())
        offloaded = torch.nn.Parameter(starting.clone())
        settings = {"lr": 2e-5, "betas": (0.9, 0.999), "eps": 1e-8}
        official = torch.optim.AdamW([{"params": [reference], "weight_decay": 0.01}], **settings)
        custom = CpuOffloadedAdamW([{"params": [offloaded], "weight_decay": 0.01}], **settings)
        gradients = [torch.tensor([[0.3, -0.5], [0.1, 0.8]]) * scale
                     for scale in (1.0, -0.4, 2.1, 0.7)]
        for gradient in gradients[:3]:
            reference.grad = gradient.clone()
            offloaded.grad = gradient.clone()
            official.step()
            custom.step()
            self.assertTrue(torch.allclose(reference, offloaded, atol=1e-7, rtol=1e-6))
        resumed = torch.nn.Parameter(offloaded.detach().clone())
        restored = CpuOffloadedAdamW([{"params": [resumed], "weight_decay": 0.01}], **settings)
        restored.load_state_dict(copy.deepcopy(custom.state_dict()))
        reference.grad = gradients[3].clone()
        resumed.grad = gradients[3].clone()
        official.step()
        restored.step()
        self.assertTrue(torch.allclose(reference, resumed, atol=1e-7, rtol=1e-6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_grad_scaler_integration(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.2], device="cuda"))
        optimizer = CpuOffloadedAdamW([{"params": [parameter], "weight_decay": 0.01}],
                                      lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
        scaler = torch.amp.GradScaler("cuda")
        before = parameter.detach().clone()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = (parameter.square()).sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        scaler.step(optimizer)
        scaler.update()
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertEqual(optimizer.state[parameter]["exp_avg"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
