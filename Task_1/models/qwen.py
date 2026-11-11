# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Load the pinned post-trained Qwen checkpoint with the required precision."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen(data_root, precision, device="cuda", gradient_checkpointing=False):
    """Return a fresh model and tokenizer from the downloaded checkpoint."""
    model_path = Path(data_root) / "models" / "Qwen3-0.6B"
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Download Qwen3-0.6B first: {model_path}")
    if precision not in ("float16", "float32"):
        raise ValueError(f"Unsupported precision: {precision}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the specified FP16 training recipe.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if precision == "float32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, local_files_only=True, attn_implementation="sdpa"
    )
    model.to(device)
    model.config.use_cache = not gradient_checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer


def assert_non_thinking_prompt(tokenizer, question, system=None):
    """Build a user prompt with Qwen's non-thinking assistant prefix."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )

