from __future__ import annotations

from dataclasses import replace

import pytest

from token_efficient_reasoning.controlled.families import FAMILIES
from token_efficient_reasoning.controlled.rendering import render_example
from token_efficient_reasoning.controlled.schema import REPRESENTATIONS
from token_efficient_reasoning.controlled.verification import verify_example


@pytest.mark.parametrize("family_name", sorted(FAMILIES))
@pytest.mark.parametrize("depth", [2, 4, 8, 12, 16])
def test_family_round_trips_every_representation(family_name: str, depth: int) -> None:
    family = FAMILIES[family_name]
    example = render_example(family.generate(seed=depth * 101, depth=depth, split="train"))

    verify_example(example)
    assert len(example.transitions) == depth
    assert set(example.renderings) == set(REPRESENTATIONS)
    for representation, text in example.renderings.items():
        assert family.parse(text, representation) == example.transitions


@pytest.mark.parametrize("family_name", sorted(FAMILIES))
def test_replay_rejects_changed_transition(family_name: str) -> None:
    family = FAMILIES[family_name]
    example = family.generate(seed=919, depth=4, split="test")
    first = example.transitions[0]
    changed = first.after + 1 if isinstance(first.after, int) else f"{first.after}_bad"
    transitions = (replace(first, after=changed), *example.transitions[1:])

    with pytest.raises(ValueError):
        family.replay(example, transitions)


def test_compositional_templates_are_withheld() -> None:
    for family in FAMILIES.values():
        train = family.generate(seed=7, depth=2, split="train")
        compositional = family.generate(seed=7, depth=2, split="test_compositional")

        assert train.template_id != compositional.template_id
