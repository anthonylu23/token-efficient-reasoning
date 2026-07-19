# Token-Efficient Reasoning

Research project investigating whether post-training can teach small language models to complete more verified conceptual reasoning steps within a fixed generated-token budget.

The initial study targets 0.6B–1.7B parameter models trainable locally on an RTX 3060 Ti using 4-bit QLoRA. It compares ordinary chain-of-thought, prompt-only compression, full-CoT fine-tuning, telegraphic reasoning, and structured reasoning sketches across controlled reasoning depths and fixed token budgets.

See the [project proposal and experimental specification](PROJECT_PROPOSAL.md) for the research questions, hypotheses, controls, and evaluation protocol. The [experimental roadmap](docs/ROADMAP.md) defines milestone deliverables, acceptance gates, verification procedures, and required artifacts.

## Development

Milestone 0 implements token-native fixed-budget evaluation for the frozen C0/C1 baselines. See the [M0 evaluator documentation](docs/M0_EVALUATOR.md) for the protocol, configuration, artifact schema, and verification workflow.

```bash
uv sync
uv run pytest
uv run ter-eval validate-config configs/m0.json
```
