"""Local 4-bit QLoRA execution, checkpointing, and reload verification for M2."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from token_efficient_reasoning.artifacts import atomic_write_json
from token_efficient_reasoning.config import CheckpointConfig, ExperimentConfig
from token_efficient_reasoning.controlled.schema import load_examples
from token_efficient_reasoning.provenance import (
    collect_environment,
    collect_hardware,
    collect_repository_state,
    sha256_file,
)

from .artifacts import tree_manifest, write_failure
from .config import M2Config, QLoRAConfig
from .data import (
    CausalDataCollator,
    CausalTrainingDataset,
    TokenizedTrainingExample,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)

_CHECKPOINT_NAME = re.compile(r"^checkpoint-(\d+)$")


def _resolve_from_config(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    repository_candidate = Path.cwd() / candidate
    if repository_candidate.exists():
        return repository_candidate.resolve()
    return (config_path.parent / candidate).resolve()


def load_m2_contract(
    protocol_path: str | Path,
    qlora_path: str | Path,
) -> tuple[M2Config, QLoRAConfig, CheckpointConfig, Path]:
    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    protocol = M2Config.from_json(protocol_source)
    checkpoint = protocol.validate_references(protocol_source)
    qlora = QLoRAConfig.from_json(qlora_source)
    if qlora.seed != protocol.training_seed:
        raise ValueError("QLoRA seed must match the frozen M2 training seed")
    dataset_dir = _resolve_from_config(protocol_source, protocol.dataset_dir)
    return protocol, qlora, checkpoint, dataset_dir


def seed_training(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_run_directory(
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None,
) -> tuple[Path, Path | None]:
    """Create a new run directory or validate a same-run checkpoint resume."""

    run_dir = Path(output_dir).resolve()
    checkpoint = (
        Path(resume_from_checkpoint).resolve()
        if resume_from_checkpoint is not None
        else None
    )
    if checkpoint is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, None
    if not run_dir.is_dir():
        raise ValueError("resumed training requires an existing run directory")
    if (
        not checkpoint.is_dir()
        or checkpoint.parent != run_dir / "checkpoints"
        or _CHECKPOINT_NAME.fullmatch(checkpoint.name) is None
    ):
        raise ValueError(
            "resume checkpoint must be a checkpoint-N directory inside the run's "
            "checkpoints directory"
        )
    return run_dir, checkpoint


def load_tokenizer(checkpoint: CheckpointConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.model_id,
        revision=checkpoint.tokenizer_revision,
        trust_remote_code=checkpoint.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer defines neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_quantized_lora_model(
    checkpoint: CheckpointConfig,
    qlora: QLoRAConfig,
    *,
    adapter_path: str | Path | None = None,
    trainable: bool = True,
) -> Any:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("M2 QLoRA requires a CUDA device")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[qlora.compute_dtype]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.quantization,
        bnb_4bit_use_double_quant=qlora.double_quantization,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.model_id,
        revision=checkpoint.revision,
        trust_remote_code=checkpoint.trust_remote_code,
        quantization_config=quantization,
        dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if adapter_path is not None:
        return PeftModel.from_pretrained(
            model,
            str(Path(adapter_path).resolve()),
            is_trainable=trainable,
        )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=qlora.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=qlora.lora_rank,
        lora_alpha=qlora.lora_alpha,
        lora_dropout=qlora.lora_dropout,
        target_modules=list(qlora.target_modules),
        bias="none",
    )
    return get_peft_model(model, peft_config)


def _parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(b"\0")
        value = parameter.detach().float().cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def _parameter_summary(model: Any) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
    }


def _generate_fixture(model: Any, tokenized: TokenizedTrainingExample) -> dict[str, Any]:
    import torch

    prompt_ids = tokenized.input_ids[: tokenized.prompt_token_count]
    device = next(model.parameters()).device
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=model.config.eos_token_id,
        )
    model.config.use_cache = previous_cache
    return {
        "schema_version": 1,
        "item_id": tokenized.item_id,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": tuple(int(token) for token in output[0, len(prompt_ids) :].tolist()),
    }


def train_condition(
    protocol_path: str | Path,
    qlora_path: str | Path,
    condition_id: str,
    output_dir: str | Path,
    *,
    max_items: int | None = None,
    max_steps: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one M2 condition and retain enough state to resume or audit it."""

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    protocol, qlora, checkpoint, dataset_dir = load_m2_contract(
        protocol_source, qlora_source
    )
    return train_condition_resolved(
        protocol,
        qlora,
        checkpoint,
        dataset_dir,
        condition_id,
        output_dir,
        protocol_path=protocol_source,
        qlora_path=qlora_source,
        max_items=max_items,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def train_condition_resolved(
    protocol: Any,
    qlora: QLoRAConfig,
    checkpoint: CheckpointConfig,
    dataset_dir: str | Path,
    condition_id: str,
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    qlora_path: str | Path,
    max_items: int | None = None,
    max_steps: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train one resolved supervised condition under either an M2 or M3 contract."""

    import torch
    from transformers import Trainer, TrainingArguments

    protocol_source = Path(protocol_path).resolve()
    qlora_source = Path(qlora_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    conditions = {condition.id: condition for condition in protocol.conditions}
    if condition_id not in conditions:
        raise ValueError(f"unknown trained condition: {condition_id!r}")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    run_dir, resume_checkpoint = prepare_training_run_directory(
        output_dir,
        resume_from_checkpoint,
    )
    resuming = resume_checkpoint is not None
    run_id = run_dir.name
    try:
        seed_training(qlora.seed)
        previous_resolved: dict[str, Any] | None = None
        resume_history: list[str] = []
        if resuming:
            resolved_path = run_dir / "resolved_config.json"
            if not resolved_path.is_file():
                raise ValueError("resumed training lacks resolved_config.json")
            with resolved_path.open(encoding="utf-8") as handle:
                previous_resolved = json.load(handle)
            resume_history = list(previous_resolved.get("resume_history", ()))
            legacy_resume = previous_resolved.get("resume_from_checkpoint")
            if legacy_resume and legacy_resume not in resume_history:
                resume_history.append(legacy_resume)
            resume_history.append(str(resume_checkpoint))
            attempt_number = len(resume_history)
            atomic_write_json(
                run_dir / f"resume_environment_{attempt_number:03d}.json",
                collect_environment(),
            )
            atomic_write_json(
                run_dir / f"resume_hardware_before_{attempt_number:03d}.json",
                collect_hardware(),
            )
            atomic_write_json(
                run_dir / f"resume_repository_{attempt_number:03d}.json",
                collect_repository_state(Path.cwd()),
            )
            failure_path = run_dir / "failure.json"
            if failure_path.is_file():
                attempts_dir = run_dir / "attempts"
                attempts_dir.mkdir(parents=True, exist_ok=True)
                archived = (
                    attempts_dir
                    / f"failure_before_resume_{attempt_number:03d}.json"
                )
                if archived.exists():
                    raise ValueError(f"resume failure archive already exists: {archived}")
                failure_path.replace(archived)
        else:
            atomic_write_json(run_dir / "environment.json", collect_environment())
            atomic_write_json(run_dir / "hardware_before.json", collect_hardware())
            atomic_write_json(
                run_dir / "repository.json",
                collect_repository_state(Path.cwd()),
            )
        source_path = dataset_dir / "splits" / f"{protocol.train_split}.jsonl"
        source_examples = load_examples(source_path)
        if max_items is not None:
            source_examples = source_examples[:max_items]
        tokenizer = load_tokenizer(checkpoint)
        rows = build_training_examples(source_examples, protocol, checkpoint)[condition_id]
        tokenized = [
            tokenize_training_example(
                row,
                tokenizer,
                checkpoint,
                max_sequence_length=qlora.max_sequence_length,
            )
            for row in rows
        ]
        exposure = summarize_token_exposure({condition_id: tokenized})
        atomic_write_json(run_dir / "token_exposure.json", exposure)
        resolved = {
            "schema_version": 1,
            "run_id": run_id,
            "condition": asdict(conditions[condition_id]),
            "protocol": protocol.to_dict(),
            "qlora": qlora.to_dict(),
            "protocol_path": str(protocol_source),
            "protocol_sha256": sha256_file(protocol_source),
            "qlora_path": str(qlora_source),
            "qlora_sha256": sha256_file(qlora_source),
            "checkpoint": asdict(checkpoint),
            "source_split": str(source_path),
            "source_split_sha256": sha256_file(source_path),
            "item_count": len(rows),
            "item_ids": [row.item_id for row in rows],
            "max_items_override": max_items,
            "max_steps_override": max_steps,
            "resume_from_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
            "resume_history": resume_history,
        }
        if previous_resolved is not None:
            immutable_previous = {
                key: value
                for key, value in previous_resolved.items()
                if key not in {"resume_from_checkpoint", "resume_history"}
            }
            immutable_current = {
                key: value
                for key, value in resolved.items()
                if key not in {"resume_from_checkpoint", "resume_history"}
            }
            if immutable_previous != immutable_current:
                raise ValueError(
                    "resumed training configuration differs from the original run"
                )
        atomic_write_json(run_dir / "resolved_config.json", resolved)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = load_quantized_lora_model(checkpoint, qlora)
        parameter_summary = _parameter_summary(model)
        initial_digest = _parameter_digest(model)
        atomic_write_json(
            run_dir / "model_parameters.json",
            {**parameter_summary, "initial_trainable_sha256": initial_digest},
        )
        training_args = TrainingArguments(
            output_dir=str(run_dir / "checkpoints"),
            overwrite_output_dir=False,
            do_train=True,
            per_device_train_batch_size=qlora.per_device_train_batch_size,
            gradient_accumulation_steps=qlora.gradient_accumulation_steps,
            learning_rate=qlora.learning_rate,
            weight_decay=qlora.weight_decay,
            max_grad_norm=qlora.max_grad_norm,
            num_train_epochs=qlora.epochs,
            max_steps=max_steps if max_steps is not None else -1,
            lr_scheduler_type=qlora.lr_scheduler_type,
            warmup_ratio=qlora.warmup_ratio,
            logging_strategy="steps",
            logging_first_step=True,
            logging_steps=qlora.logging_steps,
            save_strategy="steps",
            save_steps=qlora.save_steps,
            save_total_limit=qlora.save_total_limit,
            save_safetensors=True,
            seed=qlora.seed,
            data_seed=qlora.seed,
            bf16=qlora.compute_dtype == "bfloat16",
            fp16=qlora.compute_dtype == "float16",
            optim=qlora.optimizer,
            report_to="none",
            disable_tqdm=False,
            remove_unused_columns=False,
            gradient_checkpointing=qlora.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
            skip_memory_metrics=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=CausalTrainingDataset(tokenized),
            data_collator=CausalDataCollator(
                pad_token_id=int(tokenizer.pad_token_id),
                pad_to_multiple_of=qlora.pad_to_multiple_of,
            ),
            processing_class=tokenizer,
        )
        started = time.perf_counter()
        result = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )
        wall_time = time.perf_counter() - started
        final_digest = _parameter_digest(model)
        adapter_dir = run_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)
        fixture = _generate_fixture(model, tokenized[0])
        atomic_write_json(run_dir / "reload_fixture.json", fixture)
        telemetry = {
            "schema_version": 1,
            "run_id": run_id,
            "condition_id": condition_id,
            "wall_time_seconds": wall_time,
            "trainer_metrics": result.metrics,
            "log_history": trainer.state.log_history,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "trainable_parameters_changed": final_digest != initial_digest,
            "initial_trainable_sha256": initial_digest,
            "final_trainable_sha256": final_digest,
            "torch_cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "torch_cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            "adapter_size_bytes": sum(
                path.stat().st_size for path in adapter_dir.rglob("*") if path.is_file()
            ),
        }
        atomic_write_json(run_dir / "telemetry.json", telemetry)
        atomic_write_json(run_dir / "hardware_after.json", collect_hardware())
        manifest = tree_manifest(run_dir, run_id)
        return {
            "run_dir": str(run_dir),
            "condition_id": condition_id,
            "global_step": trainer.state.global_step,
            "trainable_parameters_changed": telemetry["trainable_parameters_changed"],
            "peak_allocated_bytes": telemetry["torch_cuda_peak_allocated_bytes"],
            "manifest_files": len(manifest["files"]),
        }
    except BaseException as error:
        write_failure(run_dir, error, run_id=run_id, condition_id=condition_id)
        tree_manifest(run_dir, run_id)
        raise


def verify_adapter_reload(run_dir: str | Path) -> dict[str, Any]:
    """Reload a saved adapter in a fresh process and reproduce its greedy fixture."""

    import torch

    root = Path(run_dir).resolve()
    with (root / "resolved_config.json").open(encoding="utf-8") as handle:
        resolved = json.load(handle)
    with (root / "reload_fixture.json").open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    checkpoint = CheckpointConfig.from_dict(resolved["checkpoint"])
    qlora = QLoRAConfig.from_dict(resolved["qlora"])
    model = load_quantized_lora_model(
        checkpoint,
        qlora,
        adapter_path=root / "adapter",
        trainable=False,
    )
    model.config.use_cache = True
    model.eval()
    prompt_ids = tuple(fixture["prompt_token_ids"])
    device = next(model.parameters()).device
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=len(fixture["generated_token_ids"]),
            pad_token_id=model.config.eos_token_id,
        )
    actual = tuple(int(token) for token in output[0, len(prompt_ids) :].tolist())
    expected = tuple(fixture["generated_token_ids"])
    result = {
        "schema_version": 1,
        "run_id": resolved["run_id"],
        "condition_id": resolved["condition"]["id"],
        "expected_token_ids": expected,
        "actual_token_ids": actual,
        "identical": actual == expected,
    }
    atomic_write_json(root / "reload_verification.json", result)
    tree_manifest(root, resolved["run_id"])
    if not result["identical"]:
        raise RuntimeError("fresh-process adapter reload did not reproduce the saved fixture")
    return result


def load_m0_config(protocol_path: str | Path, protocol: M2Config) -> ExperimentConfig:
    source = Path(protocol_path).resolve()
    return ExperimentConfig.from_json(_resolve_from_config(source, protocol.m0_config_path))
