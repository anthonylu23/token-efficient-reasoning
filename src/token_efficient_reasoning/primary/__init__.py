"""Milestone 3 primary-experiment contracts and orchestration."""

from .config import M3Config, M3ConfigError
from .planning import build_run_plan

__all__ = ["M3Config", "M3ConfigError", "build_run_plan"]
