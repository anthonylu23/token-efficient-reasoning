from __future__ import annotations

from token_efficient_reasoning.provenance import (
    collect_repository_state,
    sha256_file,
    stable_item_seed,
)


def test_workspace_state_includes_untracked_source_files() -> None:
    state = collect_repository_state(".")

    assert state["workspace_file_count"] > 10
    assert len(state["workspace_sha256"]) == 64
    assert isinstance(state["dirty"], bool)


def test_file_hash_and_item_seed_are_stable(tmp_path) -> None:
    path = tmp_path / "value"
    path.write_text("same", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)
    assert stable_item_seed(7, "item", 16) == stable_item_seed(7, "item", 16)
    assert stable_item_seed(7, "item", 16) != stable_item_seed(7, "item", 32)


def test_workspace_hash_falls_back_without_git_metadata(tmp_path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    generated = tmp_path / "artifacts" / "runs"
    generated.mkdir(parents=True)
    (generated / "ignored.json").write_text("{}", encoding="utf-8")

    state = collect_repository_state(tmp_path)

    assert state["workspace_file_count"] == 1
    assert len(state["workspace_sha256"]) == 64
