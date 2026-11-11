# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Pad supervised language-model batches without exposing prompt tokens."""

import torch


class ResponseOnlyCollator:
    """Right-pad a microbatch and preserve response-only loss masks."""

    def __init__(self, pad_token_id):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids, labels, attention_masks = [], [], []
        for example in examples:
            length = len(example["input_ids"])
            padding = max_length - length
            input_ids.append(example["input_ids"] + [self.pad_token_id] * padding)
            labels.append(example["labels"] + [-100] * padding)
            attention_masks.append([1] * length + [0] * padding)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
        expected = sum(int(item["valid_token_count"]) for item in examples)
        if int((batch["labels"] != -100).sum()) != expected:
            raise RuntimeError("The collated supervised-token count changed.")
        return batch
