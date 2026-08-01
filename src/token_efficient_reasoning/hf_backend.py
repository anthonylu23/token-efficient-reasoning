"""Hugging Face model adapter for the budgeted generation protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import CheckpointConfig, DecodingConfig
from .protocol import BackendOutput, StopCallback


def render_model_prompt(
    tokenizer: Any,
    checkpoint: CheckpointConfig,
    user_prompt: str,
) -> tuple[str, tuple[int, ...]]:
    """Render the exact model prefix shared by training and evaluation."""

    messages = [{"role": "user", "content": user_prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **checkpoint.template_kwargs,
    )
    if not rendered.endswith(checkpoint.reasoning_prefix):
        rendered += checkpoint.reasoning_prefix
    token_ids = tuple(int(token) for token in tokenizer.encode(rendered, add_special_tokens=False))
    return rendered, token_ids


class HuggingFaceBackend:
    """Single-item Transformers generation backend.

    M0 intentionally uses batch size one so token-boundary behavior and per-item
    seeds remain straightforward and auditable.
    """

    def __init__(self, model: Any, tokenizer: Any, decoding: DecodingConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.decoding = decoding

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def generate(
        self,
        input_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        should_stop: StopCallback,
    ) -> BackendOutput:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        prompt_length = len(input_ids)

        class CallbackStoppingCriteria(StoppingCriteria):
            def __call__(self, ids: Any, scores: Any, **kwargs: Any) -> Any:
                generated = tuple(int(token) for token in ids[0, prompt_length:].tolist())
                decision = should_stop(generated)
                return torch.tensor([decision], dtype=torch.bool, device=ids.device)

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_tensor)
        generation_kwargs = self.decoding.generation_kwargs()
        generation_kwargs.update(
            {
                "max_new_tokens": max_new_tokens,
                "stopping_criteria": StoppingCriteriaList([CallbackStoppingCriteria()]),
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "return_dict_in_generate": True,
            }
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        latency = time.perf_counter() - started
        generated = tuple(int(token) for token in output.sequences[0, prompt_length:].tolist())

        eos_ids = self.tokenizer.eos_token_id
        eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        if generated and generated[-1] in eos_set:
            return BackendOutput(generated[:-1], "eos", latency)
        if should_stop(generated):
            return BackendOutput(generated, "stop", latency)
        return BackendOutput(generated, "length", latency)

    def create_sampling_states(self, seeds: tuple[int, ...]) -> tuple[Any, ...]:
        """Create one persistent device RNG stream per batch row."""

        import torch

        if any(seed < 0 for seed in seeds):
            raise ValueError("sampling seeds must be non-negative")
        return tuple(
            torch.Generator(device=self.device).manual_seed(seed) for seed in seeds
        )

    def _logits_processors(self) -> Any:
        from transformers import (
            LogitsProcessorList,
            MinPLogitsWarper,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        processors = LogitsProcessorList()
        if not self.decoding.do_sample:
            return processors
        if self.decoding.temperature is not None and self.decoding.temperature != 1:
            processors.append(TemperatureLogitsWarper(self.decoding.temperature))
        if self.decoding.top_k is not None and self.decoding.top_k > 0:
            processors.append(TopKLogitsWarper(self.decoding.top_k))
        if self.decoding.top_p is not None and self.decoding.top_p < 1:
            processors.append(TopPLogitsWarper(self.decoding.top_p))
        if self.decoding.min_p is not None and self.decoding.min_p > 0:
            processors.append(MinPLogitsWarper(self.decoding.min_p))
        return processors

    def generate_batch(
        self,
        input_ids: tuple[tuple[int, ...], ...],
        *,
        max_new_tokens: int,
        should_stop: tuple[StopCallback, ...],
        sampling_states: tuple[Any, ...],
    ) -> tuple[BackendOutput, ...]:
        """Decode a left-padded batch with independent stopping and RNG streams."""

        import torch

        if not input_ids:
            raise ValueError("batch must contain at least one input")
        if len(input_ids) != len(should_stop) or len(input_ids) != len(sampling_states):
            raise ValueError("inputs, callbacks, and sampling states must have equal length")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if any(not row for row in input_ids):
            raise ValueError("batch inputs must be non-empty")

        batch_size = len(input_ids)
        maximum_prompt = max(map(len, input_ids))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer defines neither a pad nor EOS token")
        padded = [
            [pad_id] * (maximum_prompt - len(row)) + list(row) for row in input_ids
        ]
        masks = [
            [0] * (maximum_prompt - len(row)) + [1] * len(row) for row in input_ids
        ]
        sequences = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        generated: list[list[int]] = [[] for _ in input_ids]
        finish_reasons = ["length"] * batch_size
        active = [True] * batch_size
        eos_ids = self.tokenizer.eos_token_id
        eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        processors = self._logits_processors()
        past_key_values = None
        cache_position = torch.arange(
            sequences.shape[1], dtype=torch.long, device=self.device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                model_inputs = self.model.prepare_inputs_for_generation(
                    sequences,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                )
                output = self.model(**model_inputs, return_dict=True)
                past_key_values = output.past_key_values
                cache_position = cache_position[-1:] + 1
                scores = output.logits[:, -1, :]
                scores = processors(sequences, scores)
                next_tokens: list[int] = []
                for index in range(batch_size):
                    if not active[index]:
                        next_tokens.append(int(pad_id))
                    elif self.decoding.do_sample:
                        probabilities = torch.softmax(scores[index], dim=-1)
                        token = torch.multinomial(
                            probabilities,
                            num_samples=1,
                            generator=sampling_states[index],
                        )
                        next_tokens.append(int(token.item()))
                    else:
                        next_tokens.append(int(torch.argmax(scores[index]).item()))
                next_tensor = torch.tensor(
                    next_tokens, dtype=torch.long, device=self.device
                ).unsqueeze(1)
                active_mask = torch.tensor(
                    [int(value) for value in active],
                    dtype=attention_mask.dtype,
                    device=self.device,
                ).unsqueeze(1)
                sequences = torch.cat((sequences, next_tensor), dim=1)
                attention_mask = torch.cat((attention_mask, active_mask), dim=1)
                for index, token in enumerate(next_tokens):
                    if not active[index]:
                        continue
                    generated[index].append(token)
                    current = tuple(generated[index])
                    if token in eos_set:
                        active[index] = False
                        finish_reasons[index] = "eos"
                    elif should_stop[index](current):
                        active[index] = False
                        finish_reasons[index] = "stop"
                if not any(active):
                    break
        latency = time.perf_counter() - started
        results = []
        for tokens, reason in zip(generated, finish_reasons, strict=True):
            if reason == "eos" and tokens and tokens[-1] in eos_set:
                tokens = tokens[:-1]
            results.append(BackendOutput(tuple(tokens), reason, latency))
        return tuple(results)


@dataclass
class ModelSession:
    checkpoint: CheckpointConfig
    tokenizer: Any
    model: Any
    backend: HuggingFaceBackend

    def render_prompt(self, user_prompt: str) -> tuple[str, tuple[int, ...]]:
        return render_model_prompt(self.tokenizer, self.checkpoint, user_prompt)

    def encode_protocol_text(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=False))

    def encode_delimiter_variants(self, text: str) -> tuple[tuple[int, ...], ...]:
        """Tokenize a delimiter with common fused trailing-whitespace variants."""

        variants = {self.encode_protocol_text(text + suffix) for suffix in ("", "\n", "\n\n", " ")}
        return tuple(sorted(variants, key=lambda value: (len(value), value)))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def load_model_session(
    checkpoint: CheckpointConfig,
    decoding: DecodingConfig,
    device: str,
    *,
    adapter_path: str | Path | None = None,
) -> ModelSession:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[checkpoint.dtype]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.model_id,
        revision=checkpoint.tokenizer_revision,
        trust_remote_code=checkpoint.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.model_id,
        revision=checkpoint.revision,
        trust_remote_code=checkpoint.trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if adapter_path is not None:
        from peft import PeftModel

        adapter = Path(adapter_path).resolve()
        if not adapter.is_dir():
            raise ValueError(f"adapter directory does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    backend = HuggingFaceBackend(model, tokenizer, decoding)
    return ModelSession(checkpoint=checkpoint, tokenizer=tokenizer, model=model, backend=backend)
