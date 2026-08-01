# Token-Efficient Reasoning

Research project investigating whether post-training can teach small language models to complete more verified conceptual reasoning steps within a fixed generated-token budget.

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
implementation and its validation-only bridge are complete; the bridge passed, while held-out test
access remains fail-closed pending the 34-item M1 human review and protocol freeze. See the
[M3 primary protocol](docs/M3_PROTOCOL.md) and
[bridge gate review](artifacts/M3_BRIDGE_GATE_REVIEW.md). After freeze, the full final matrix is
executed sequentially and resumably with `python scripts/run_m3_primary.py --stage all`; primary
runs and reporting fail if the repository workspace differs from the freeze.

```bash
uv sync
uv run pytest
uv run ter-eval validate-config configs/m0.json
uv run ter-data validate-config configs/m1.json
uv run ter-m2 validate
uv run ter-m3 validate
```
