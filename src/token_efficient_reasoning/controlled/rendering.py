"""Derive all textual representations from one canonical transition path."""

from __future__ import annotations

from dataclasses import replace

from .families import FAMILIES
from .schema import REPRESENTATIONS, CanonicalExample


def render_example(example: CanonicalExample) -> CanonicalExample:
    try:
        family = FAMILIES[example.family]
    except KeyError as error:
        raise ValueError(f"unknown controlled task family: {example.family!r}") from error
    renderings = {
        representation: family.render(example.transitions, representation)
        for representation in REPRESENTATIONS
    }
    return replace(example, renderings=renderings)
