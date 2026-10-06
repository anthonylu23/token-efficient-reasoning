# Token-Efficient Reasoning

Research project investigating whether post-training can teach small language models to complete more verified conceptual reasoning steps within a fixed generated-token budget.

This follow-up worktree contains supplemental analysis, answer-only target preparation,
arithmetic composition holdouts, and external-data preparation. See the
[CPU-only follow-up implementations](docs/FOLLOWUP_DATA.md).
All 342,000 M3 generations completed on 2026-10-03; primary and supplemental reports
were recovered on 2026-10-06 after a reporting-only manifest compatibility fix. See
[the primary report](artifacts/results/m3-20261006/primary/M3_REPORT.md),
[supplemental diagnostics](artifacts/results/m3-20261006/budget/REPORT.md),
[recovery provenance and commands](docs/M3_EXECUTION.md), and the
[follow-up protocol](docs/FOLLOWUP_PROTOCOL.md). Keep the original checkout unchanged
for frozen-input validation and reproducibility.

The initial study targets 0.6B–1.7B parameter models trainable locally on an RTX 3060 Ti using 4-bit QLoRA. It compares ordinary chain-of-thought, prompt-only compression, full-CoT fine-tuning, telegraphic reasoning, and structured reasoning sketches across controlled reasoning depths and fixed token budgets.

See the [project proposal and experimental specification](PROJECT_PROPOSAL.md) for the research questions, hypotheses, controls, and evaluation protocol. The [experimental roadmap](docs/ROADMAP.md) defines milestone deliverables, acceptance gates, verification procedures, and required artifacts.

## Development

Milestone 0 implements token-native fixed-budget evaluation for the frozen C0/C1 baselines.
Milestone 1 adds deterministic controlled-depth arithmetic and graph data, executable replay
verification, R0–R3 traces, split auditing, and pinned-tokenizer compression measurement.
Milestone 2 completes the local Qwen3-0.6B QLoRA workflow and the full C0–C4 validation frontier;
its gate decision is **revise or hold** because C4 misses the preregistered very-low-budget
non-inferiority and all-budget structured-trace thresholds. See the
[M0 evaluator documentation](docs/M0_EVALUATOR.md),
[M1 dataset documentation](docs/M1_DATASET.md), and
[M2 smoke-test report](docs/M2_PROTOCOL.md) for protocols, workflows, and results. Milestone 3
implementation and its validation-only bridge are complete; the bridge and the 34-item M1
human review passed, with documented limitations. Primary execution and recovered reporting
are complete: central fixed-budget criteria pass, but C4 does not outperform C3 under H5.
See the
[M3 primary protocol](docs/M3_PROTOCOL.md) and
[bridge gate review](artifacts/M3_BRIDGE_GATE_REVIEW.md). The full final matrix is
executed sequentially and resumably with `python scripts/run_m3_primary.py --stage all`; primary
runs and reporting fail if the repository workspace differs from the freeze.

Install the Linux-only `training` extra before QLoRA work; CPU-only data and evaluation
workflows can use plain `uv sync`.

```bash
uv sync --extra training
uv run pytest
uv run ter-eval validate-config configs/m0.json
uv run ter-data validate-config configs/m1.json
uv run ter-m2 validate
uv run ter-m3 validate
```
