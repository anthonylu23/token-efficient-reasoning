# Milestone 0 evaluator

Milestone 0 establishes a reproducible frozen-checkpoint baseline and the evaluation contracts used by later training experiments. The evaluator enforces reasoning and answer budgets over tokenizer IDs; decoded text is retained for inspection but is not the source of accounting truth.

## Protocol

Each checkpoint adapter renders a model-native chat prompt ending at `<think>\n`. Generation then proceeds in two phases:

1. Generate at most `B` reasoning-content tokens. A model-generated `</think>` ends the phase early. Otherwise the evaluator injects the close delimiter after the budget is consumed.
2. Inject `<answer>`, generate at most the configured answer allowance, and close the answer region when the model emits the whitespace-independent `</answer>` delimiter, reaches EOS, or exhausts the allowance.

Delimiters, prompt tokens, and answer tokens never count toward the reasoning budget. Multi-token delimiters receive bounded lookahead so a delimiter beginning at a budget boundary can complete. Checkpoint adapters precompute audited variants in which a tokenizer fuses trailing whitespace into the final delimiter token; all variants map to one canonical injected delimiter. Any lookahead token that proves not to be part of a delimiter is retained in the raw phase output and excluded from the effective transcript.

The evaluator saves raw sampled tokens separately for the reasoning and answer phases. It also saves the effective canonical transcript, delimiter provenance, termination reasons, overflow tokens, exact prompt, per-item seed, timing, correctness, and tokenizer/model revisions.

## Configuration

[`configs/m0.json`](../configs/m0.json) pins the two checkpoints by commit, defines conditions C0 and C1, and preregisters budgets `16, 32, 64, 128, 256, 512`. The checked-in development fixture contains short arithmetic items. It tests the evaluator and is not a substitute for the controlled-depth dataset introduced in M1.

Validate the configuration without loading a model:

```bash
uv run ter-eval validate-config configs/m0.json
```

Run a minimal GPU smoke evaluation:

```bash
uv run ter-eval evaluate configs/m0.json \
  --checkpoint qwen3-0.6b \
  --condition c0 \
  --budgets 16 \
  --max-items 1
```

Run directories contain resolved configuration, a hash of every non-ignored workspace file, environment and pre-load/post-load hardware snapshots, item-level JSONL, a summary, and a SHA-256 manifest. Compare repeated runs while ignoring runtime-only latency:

```bash
uv run ter-eval compare-runs artifacts/runs/RUN_A artifacts/runs/RUN_B
```

Replay the exact checkpoint, condition, item prefix, budgets, and seeds from a saved run. Replay fails if the checked-in fixture hash or item ordering has changed:

```bash
uv run ter-eval replay-run artifacts/runs/RUN_A
```

## Verification levels

- `uv run pytest`: CPU tests using scripted token streams; no model download is required.
- `RUN_MODEL_TESTS=1 uv run pytest -m model`: load pinned tokenizers and verify real template/delimiter boundaries.
- GPU smoke: run one item at budget 16 on each checkpoint before any full matrix.
- Gate run: execute C0 and C1 across all budgets and fixtures, repeat the same seeded matrix, compare item-level outputs, and record the RTX 3060 Ti hardware profile.

M0 does not gate on model accuracy. Advancement requires trusted accounting, forced transitions, reproducibility, complete artifacts, and passing tests.
