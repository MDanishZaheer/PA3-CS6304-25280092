# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check chunked response-token loss against the direct shifted CE formula."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train import supervised_loss_sum


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 7)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBody()
        self.lm_head = nn.Linear(7, 13, bias=False)


class ResponseLossTest(unittest.TestCase):
    def test_mask_and_gradient(self):
        torch.manual_seed(6304)
        model = TinyModel()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, -100, 4, 5, 6],
                                    [-100, -100, 4, 5, 6, 7]]),
        }
        expected_logits = model.lm_head(model.model(batch["input_ids"], batch["attention_mask"]).last_hidden_state)
        expected = F.cross_entropy(expected_logits[:, :-1].reshape(-1, 13),
                                   batch["labels"][:, 1:].reshape(-1),
                                   ignore_index=-100, reduction="sum")
        expected.backward()
        expected_gradients = [parameter.grad.clone() for parameter in model.parameters()]
        model.zero_grad(set_to_none=True)
        actual = supervised_loss_sum(model, batch)
        actual.backward()
        self.assertAlmostEqual(float(expected.detach()), float(actual.detach()), places=5)
        for parameter, gradient in zip(model.parameters(), expected_gradients):
            self.assertTrue(torch.allclose(parameter.grad, gradient, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
