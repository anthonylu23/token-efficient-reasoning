"""Condition-controlled user prompts for the frozen C0 and C1 baselines."""

from __future__ import annotations

from .config import ConditionConfig
from .data import EvaluationItem

_FORMAT_INSTRUCTION = (
    "Reason through the problem. After the reasoning region is closed, the evaluator will begin "
    "an answer region. In that region, give only the final answer with no explanation."
)


def build_user_prompt(item: EvaluationItem, condition: ConditionConfig) -> str:
    sections = [_FORMAT_INSTRUCTION]
    if condition.instruction.strip():
        sections.append(condition.instruction.strip())
    for shot in condition.few_shots:
        sections.append(f"Example problem:\n{shot.user}\nExample response:\n{shot.assistant}")
    sections.append(f"Problem:\n{item.prompt}")
    return "\n\n".join(sections)
