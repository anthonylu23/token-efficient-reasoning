# Token-Efficient Reasoning

Research project investigating whether post-training can teach small language models to complete more verified conceptual reasoning steps within a fixed generated-token budget.

The initial study targets 0.6B–1.7B parameter models trainable locally on an RTX 3060 Ti using 4-bit QLoRA. It compares ordinary chain-of-thought, prompt-only compression, full-CoT fine-tuning, telegraphic reasoning, and structured reasoning sketches across controlled reasoning depths and fixed token budgets.

See the [project proposal and experimental specification](PROJECT_PROPOSAL.md) for the research questions, hypotheses, controls, evaluation protocol, milestones, and success criteria.
