"""M2 training configuration and data contracts."""

from .config import M2ConditionConfig, M2Config, M2ConfigError, QLoRAConfig
from .data import (
    CausalDataCollator,
    CausalTrainingDataset,
    TokenizedTrainingExample,
    TrainingExample,
    audit_matched_examples,
    build_training_examples,
    summarize_token_exposure,
    tokenize_training_example,
)

__all__ = [
    "M2ConditionConfig",
    "M2Config",
    "M2ConfigError",
    "QLoRAConfig",
    "CausalDataCollator",
    "CausalTrainingDataset",
    "TokenizedTrainingExample",
    "TrainingExample",
    "audit_matched_examples",
    "build_training_examples",
    "summarize_token_exposure",
    "tokenize_training_example",
]
