"""Hugging Face model adapter for the budgeted generation protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .config import CheckpointConfig, DecodingConfig
from .protocol import BackendOutput, StopCallback


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


@dataclass
class ModelSession:
    checkpoint: CheckpointConfig
    tokenizer: Any
    model: Any
    backend: HuggingFaceBackend

    def render_prompt(self, user_prompt: str) -> tuple[str, tuple[int, ...]]:
        messages = [{"role": "user", "content": user_prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.checkpoint.template_kwargs,
        )
        if not rendered.endswith(self.checkpoint.reasoning_prefix):
            rendered += self.checkpoint.reasoning_prefix
        token_ids = tuple(
            int(token) for token in self.tokenizer.encode(rendered, add_special_tokens=False)
        )
        return rendered, token_ids

    def encode_protocol_text(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=False))

    def encode_delimiter_variants(self, text: str) -> tuple[tuple[int, ...], ...]:
        """Tokenize a delimiter with common fused trailing-whitespace variants."""

        variants = {self.encode_protocol_text(text + suffix) for suffix in ("", "\n", "\n\n", " ")}
        return tuple(sorted(variants, key=lambda value: (len(value), value)))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def load_model_session(
    checkpoint: CheckpointConfig, decoding: DecodingConfig, device: str
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
    model.eval()
    backend = HuggingFaceBackend(model, tokenizer, decoding)
    return ModelSession(checkpoint=checkpoint, tokenizer=tokenizer, model=model, backend=backend)
