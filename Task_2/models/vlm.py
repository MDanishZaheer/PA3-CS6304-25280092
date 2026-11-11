# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Frozen Qwen, trainable visual connector, and Stage 2-3 LoRA adapters."""

import time
from functools import lru_cache

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


IMAGE_MARKER = "__ATML_VISUAL_SLOT__"


class VisualConnector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 1024))
        self.register_buffer("output_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, features):
        return self.layers(features.float()) * self.output_scale


class GroundedQwen(nn.Module):
    """Insert 196 continuous image tokens before text in the Qwen user message."""

    def __init__(self, language_path, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(language_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.end_id is None or self.end_id == self.tokenizer.unk_token_id:
            raise RuntimeError("Qwen end-of-turn token is unavailable.")
        self.language = AutoModelForCausalLM.from_pretrained(
            language_path, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
        ).to(device)
        if self.language.config.hidden_size != 1024:
            raise RuntimeError("Task 2 requires Qwen hidden width 1024.")
        self.language.requires_grad_(False)
        self.language.config.use_cache = False
        self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.connector = VisualConnector().to(device)
        self.adapters_enabled = False

    def enable_adapters(self):
        if self.adapters_enabled:
            return
        settings = self.config["training"]["lora"]
        self.language = get_peft_model(self.language, LoraConfig(
            r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
            bias="none", target_modules=settings["targets"], task_type="CAUSAL_LM"
        ))
        for name, parameter in self.language.named_parameters():
            if parameter.requires_grad:
                if "lora_" not in name:
                    raise RuntimeError(f"Unexpected trainable Qwen parameter: {name}")
                parameter.data = parameter.data.float()
        self.adapters_enabled = True

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @lru_cache(maxsize=65536)
    def _prompt_parts(self, kind, question):
        if kind == "caption":
            body = "Describe the image."
        else:
            cap = self.config["training"]["question_token_cap"]
            question_ids = self.tokenizer.encode(question, add_special_tokens=False)[:cap]
            body = "Answer using the image. Give a short answer.\n" + self.tokenizer.decode(question_ids)
        messages = [{"role": "user", "content": IMAGE_MARKER + "\n" + body}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if prompt.count(IMAGE_MARKER) != 1:
            raise RuntimeError("Visual slot missing or duplicated in Qwen chat prompt.")
        first, second = prompt.split(IMAGE_MARKER)
        return (self.tokenizer.encode(first, add_special_tokens=False),
                self.tokenizer.encode(second, add_special_tokens=False))

    def _target_ids(self, example):
        caption = example.get("caption")
        text = caption if caption is not None else example["target"]
        cap = self.config["training"]["caption_token_cap" if caption is not None else "answer_token_cap"]
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:cap]
        return ids + [self.end_id]

    def build_batch(self, examples, image_features, include_image=True, include_targets=True, left_pad=False):
        """Create embedding, mask, position, and supervised-label tensors."""
        embedding = self.language.get_input_embeddings()
        visual = self.connector(image_features).to(embedding.weight.dtype) if include_image else None
        rows, labels = [], []
        for index, example in enumerate(examples):
            kind = "caption" if "caption" in example else "question"
            first, second = self._prompt_parts(kind, example.get("question", ""))
            sections = [embedding(torch.tensor(first, device=self.device_name))]
            if include_image:
                sections.append(visual[index])
            sections.append(embedding(torch.tensor(second, device=self.device_name)))
            prompt_length = sum(section.shape[0] for section in sections)
            target_ids = self._target_ids(example) if include_targets else []
            if target_ids:
                sections.append(embedding(torch.tensor(target_ids, device=self.device_name)))
            rows.append(torch.cat(sections, dim=0))
            labels.append([-100] * prompt_length + target_ids)
        width = max(row.shape[0] for row in rows)
        hidden = rows[0].shape[1]
        inputs = rows[0].new_zeros((len(rows), width, hidden))
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=self.device_name)
        targets = torch.full((len(rows), width), -100, dtype=torch.long, device=self.device_name)
        for index, (row, label) in enumerate(zip(rows, labels)):
            start = width - row.shape[0] if left_pad else 0
            end = start + row.shape[0]
            inputs[index, start:end] = row
            mask[index, start:end] = 1
            targets[index, start:end] = torch.tensor(label, device=self.device_name)
        positions = (mask.cumsum(dim=1) - 1).clamp_min(0)
        return {"inputs_embeds": inputs, "attention_mask": mask,
                "position_ids": positions, "labels": targets}

    def supervised_loss(self, examples, image_features, include_image=True, include_end=True):
        batch = self.build_batch(examples, image_features, include_image=include_image)
        outputs = self.language(inputs_embeds=batch["inputs_embeds"],
                                attention_mask=batch["attention_mask"],
                                position_ids=batch["position_ids"], use_cache=False)
        logits = outputs.logits[:, :-1].float()
        labels = batch["labels"][:, 1:].clone()
        if not include_end:
            labels[labels == self.end_id] = -100
        loss = nn.functional.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
        valid = (labels != -100).sum(dim=1)
        if (valid == 0).any():
            raise RuntimeError("A caption or answer has no supervised content tokens.")
        return (loss.sum(dim=1) / valid).mean(), loss.sum(dim=1) / valid

    @torch.inference_mode()
    def generate_answers(self, examples, image_features, include_image=True, max_new_tokens=16):
        """Greedy batched decoding with a correctly left-padded multimodal prefix."""
        self.eval()
        batch = self.build_batch(examples, image_features, include_image=include_image,
                                 include_targets=False, left_pad=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        mask = batch["attention_mask"]
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device_name.startswith("cuda")):
            outputs = self.language(inputs_embeds=batch["inputs_embeds"], attention_mask=mask,
                                    position_ids=batch["position_ids"], use_cache=True)
            next_token = outputs.logits[:, -1].argmax(dim=-1)
            cache = outputs.past_key_values
            finished = torch.zeros(len(examples), dtype=torch.bool, device=self.device_name)
            generated = [[] for _ in examples]
            for step in range(max_new_tokens):
                for index, token in enumerate(next_token.tolist()):
                    if not finished[index]:
                        generated[index].append(token)
                finished |= (next_token == self.end_id) | (next_token == self.tokenizer.eos_token_id)
                if bool(finished.all()) or step == max_new_tokens - 1:
                    break
                mask = torch.cat((mask, torch.ones((len(examples), 1), device=mask.device, dtype=mask.dtype)), dim=1)
                positions = mask.sum(dim=1, keepdim=True) - 1
                outputs = self.language(input_ids=next_token.unsqueeze(1), attention_mask=mask,
                                        position_ids=positions, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_token = outputs.logits[:, -1].argmax(dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds_each = (time.perf_counter() - started) / len(examples)
        results = []
        for ids in generated:
            terminated = bool(ids and ids[-1] in (self.end_id, self.tokenizer.eos_token_id))
            text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            results.append({"prediction": text, "generated_tokens": len(ids),
                            "at_token_cap": len(ids) == max_new_tokens and not terminated,
                            "generation_seconds": seconds_each})
        return results

    @torch.no_grad()
    def calibrate_connector(self, training_features):
        """Match initial connector RMS to Qwen token-embedding RMS using train only."""
        self.connector.output_scale.fill_(1.0)
        output = self.connector.layers(training_features.float()).float()
        visual_rms = output.square().mean().sqrt()
        text_rms = self.language.get_input_embeddings().weight.float().square().mean().sqrt()
        scale = (text_rms / visual_rms.clamp_min(1e-8)).item()
        self.connector.output_scale.fill_(scale)
        return {"rule": "text_embedding_rms / initial_connector_rms", "training_images": len(training_features),
                "text_rms": text_rms.item(), "visual_rms_before": visual_rms.item(), "scale": scale}
