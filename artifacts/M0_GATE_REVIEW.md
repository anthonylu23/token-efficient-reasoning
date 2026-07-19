# Milestone 0 gate review

Date: 2026-07-18

## Decision

Milestone 0 passes its evaluator-correctness and reproducibility gate. The implementation is ready to support M1 data/tokenizer work. Baseline accuracy on the small M0 arithmetic fixture is diagnostic only and is not a gate.

The final evidence uses the `m0-v3-*` runs. Earlier smoke, v1, and v2 runs are retained as debugging evidence but are superseded by v3.

## Frozen implementation

- Workspace files: 30 non-generated files.
- Validated v3 workspace SHA-256: `26c15de39ee28137a41292fb82557482094ed996c1bed8b925603a28338a7bff`.
- Committed implementation workspace SHA-256: `9f47e730387bb2db77d7f2b7b6da3670584f7e01c6163770ff8d928f3d143ca6`.
- Qwen checkpoint: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
- DeepSeek checkpoint: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- Environment: Python 3.11.15, PyTorch 2.13.0, Transformers 4.57.6, Accelerate 1.14.0.
- Hardware: Fedora 44, RTX 3060 Ti 8 GiB, NVIDIA driver 595.71.05.

The local and remote workspace hashes matched before the v3 runs. Every v3 resolved configuration records the same workspace hash, pinned checkpoint/tokenizer revisions, exact chat-template hash, dependency versions, item IDs, budgets, seed, delimiter variants, precision, and device.

The committed implementation hash differs from the validated v3 hash only because the commit whitespace gate removed terminal blank lines from `.gitignore`, `data/m0_dev.jsonl`, and `pyproject.toml`. No executable source, configuration value, fixture item, or dependency resolution changed. The complete local test suite was rerun after this cleanup.

## Verification results

- Local tests: 27 passed; 2 model tests skipped by default.
- Explicit pinned-tokenizer tests: 2 passed on macOS and Fedora.
- Final A matrix: 288 item evaluations (2 checkpoints x 2 conditions x 6 budgets x 12 items).
- Exact replay matrix: another 288 evaluations.
- Reproducibility: all four 72-item A/B pairs were identical in prompt IDs, item seeds, raw reasoning IDs, raw answer IDs, effective transcript IDs, parsed answers, parse status, and correctness.
- Budget enforcement: zero reasoning spans over budget and zero answer spans over allowance across all 576 final evaluations.
- Artifact integrity: every file listed in all eight v3 manifests matched its SHA-256 digest.
- Seed behavior: a Qwen C0 diagnostic with seed 1730 differed in both reasoning and answer tokens from seed 1729; seeds remain item-level records and are not silently averaged.

| Checkpoint | Condition | Accuracy | Parse success | Forced reasoning | Median reasoning tokens | Mean item latency | Peak PyTorch allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | C0 | 37.5% | 100.0% | 81.9% | 96 | 3.25 s | 1.24 GiB |
| Qwen3-0.6B | C1 | 55.6% | 97.2% | 79.2% | 96 | 2.89 s | 1.24 GiB |
| DeepSeek-R1-Distill-Qwen-1.5B | C0 | 52.8% | 100.0% | 43.1% | 56 | 1.24 s | 3.36 GiB |
| DeepSeek-R1-Distill-Qwen-1.5B | C1 | 52.8% | 95.8% | 37.5% | 39.5 | 1.20 s | 3.41 GiB |

Accuracy aggregates all six budgets and should not be interpreted as a scientific checkpoint comparison. The fixture is intentionally small and the conditions use sampled decoding.

## Known limitations

1. Five C1 records are explicitly malformed because the model immediately emitted `</answer>` after the evaluator opened the answer region: two Qwen records and three DeepSeek records. These empty-answer failures reproduced exactly and remain visible rather than being repaired or scored implicitly.
2. The M0 development fixture contains only short arithmetic items. Controlled-depth task families and executable semantic verification belong to M1.
3. Desktop GPU activity varied during the runs. Per-run pre-load snapshots capture this variation; PyTorch peak allocated/reserved memory is the more useful model-memory measurement.
4. The source tree is not committed because no commit was requested. A source-hash-identical clean remote copy passed installation, lint, CPU tests, tokenizer tests, CUDA discovery, and the full gate. A literal clean Git checkout should be confirmed after the human chooses to commit.

## Final evidence directories

- `runs/m0-v3-qwen-c0-a` and `runs/m0-v3-qwen-c0-b`
- `runs/m0-v3-qwen-c1-a` and `runs/m0-v3-qwen-c1-b`
- `runs/m0-v3-deepseek-c0-a` and `runs/m0-v3-deepseek-c0-b`
- `runs/m0-v3-deepseek-c1-a` and `runs/m0-v3-deepseek-c1-b`
- `runs/m0-v3-seed-variation`
