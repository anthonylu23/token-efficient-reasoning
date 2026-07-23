"""Interface shared by controlled task families."""

from __future__ import annotations

from typing import Protocol

from ..schema import CanonicalExample, Transition


class TaskFamily(Protocol):
    name: str

    def generate(self, *, seed: int, depth: int, split: str) -> CanonicalExample: ...

    def render(self, transitions: tuple[Transition, ...], representation: str) -> str: ...

    def parse(self, text: str, representation: str) -> tuple[Transition, ...]: ...

    def solve(self, example: CanonicalExample) -> str: ...

    def replay(self, example: CanonicalExample, transitions: tuple[Transition, ...]) -> str: ...
