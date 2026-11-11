# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check exact protected-space projection and matched initial LoRA factors."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.adapters import AdapterLinear


class AdapterProjectionTest(unittest.TestCase):
    def test_projection_and_shared_initialization(self):
        torch.manual_seed(6304)
        base = nn.Linear(9, 7, bias=False)
        plain = AdapterLinear(nn.Linear(9, 7, bias=False), 2, 4, 0.05,
                              torch.Generator().manual_seed(6304))
        protected = AdapterLinear(base, 2, 4, 0.05,
                                  torch.Generator().manual_seed(6304), protection_rank=3)
        self.assertTrue(torch.equal(plain.A, protected.A))
        self.assertTrue(torch.equal(plain.B, protected.B))
        with torch.no_grad():
            plain.B.normal_()
            protected.B.copy_(plain.B)
        effective_a, effective_b = protected.effective_factors()
        delta = effective_b @ effective_a
        self.assertLess(float((protected.u_basis.T @ delta).norm().detach()), 1e-5)
        self.assertLess(float((delta @ protected.v_basis).norm().detach()), 1e-5)


if __name__ == "__main__":
    unittest.main()
