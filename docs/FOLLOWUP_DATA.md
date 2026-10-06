# CPU-only follow-up implementations

Implemented and validated on 2026-09-29 in the separate follow-up worktree.
These preparations do not modify M3, start model training, or access held-out model
results. The primary checkout must remain unchanged until its report is complete.

## Answer-only supervision

`token_efficient_reasoning.answer_only` prepares C2, C3, C4, and the separate
`c5_answer_only` condition from the same canonical M1 training and validation items.
The versioned configuration is `configs/followup/answer_only.json`; it pins the
primary protocol, QLoRA configuration, and checkpoint configuration by SHA-256.
The source dataset manifest and its files are verified before preparation.

The answer-only target has no reasoning text. It supervises the reasoning-close
delimiter, answer delimiters, and reference answer. Prompt tokens—including the
reasoning prefix—and padding remain masked out of the loss. Overflow raises an
error rather than truncating. The original M2/M3 condition validators still reject
this new condition; it belongs only to the exploratory follow-up.

Preparation confirmed exact question, answer, source identity, and item-order
matching for 8,000 training and 800 validation items. All four conditions fit the
560-token sequence limit with the pinned DeepSeek-R1-Distill-Qwen-1.5B tokenizer.
The answer-only maximum sequence length is 364 tokens. Its training split contains
83,533 supervised tokens, compared with 1,048,482 for C2, 572,516 for C3, and
702,137 for C4. These are corpus exposures, not measured optimizer-run exposures:
matching examples and updates does **not** equalize supervised token counts.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../token-efficient-reasoning/.venv/bin/python \
  -m token_efficient_reasoning.answer_only \
  --output-dir artifacts/runs/followup/answer-only-v1
```

Use the follow-up worktree as the working directory. Existing output directories
are never overwritten. `targets/<condition>/<split>.jsonl` contains training
targets; `preparation.json` records matching audits, token exposure, source hashes,
the tokenizer/checkpoint, and the resolved training contract. `manifest.json`
binds all output files. This command loads a tokenizer, not model weights.

The resolved contract retains the three primary training seeds, 500 optimizer
updates, and QLoRA settings. Actual training is deferred: a post-M3 launcher must
set each declared seed and the fixed update count explicitly, retain adapter reload
verification, and write a separate experiment artifact. No training job is queued
by preparation. Freeze the follow-up evaluation and diagnostic policy before model
evaluation; an absent trace must not be counted as an invalid attempted proof.

## Arithmetic composition holdouts

`token_efficient_reasoning.composition` implements the development design in
`configs/followup/composition_dev.json`. It reserves adjacent action pairs
`mul → sub` and `div → add` while exposing all four primitive operations in training.

- Training, validation, and IID testing exclude both reserved pairs.
- Each composition item contains exactly one **kind** of reserved pair; it can
  occur more than once. Pair kinds have separate balanced quotas.
- Every split shares `arith_counter` and `arith_machine` templates and depths
  2, 4, 8, and 12. This isolates composition from unseen-template/depth tests.
- Seeds are deterministic per split/depth/template/pair cell. Candidate rejection
  is bounded; no token-length or model-performance filtering is used.
- Duplicate calculation hashes ignore distractors and template wording. IDs,
  seeds, and prompts are also checked for duplicates within and across splits.
- Exact arithmetic, integer division, all R0–R3 parse/replay round trips, and
  counterfactual transition/problem rejection must pass.

```bash
PYTHONPATH=src ../token-efficient-reasoning/.venv/bin/python \
  -m token_efficient_reasoning.composition \
  --output-dir artifacts/runs/followup/composition-dev-v1
```

The validated development artifact contains 400 training, 80 validation, 80 IID,
and 160 composition items. `audit.json` records exact cell quotas, primitive
coverage, action-sequence overlap, answer histograms, revisited states, and
distractor/answer coincidences. The latter two are reported, not filtered away.
`sampling.json` records attempts and rejection counts; manifests bind source
hashes and outputs. Composition action sequences have no overlap with training.

This is a development corpus, not a final evaluation freeze. Reserved pairs
necessarily alter operation distributions, so passing the leakage audit does not
establish a matched-distribution causal test. Select final sample sizes, separate
seeds, budget profiles, and statistical contrasts before training/evaluation.
Additional unseen-depth and two-register/branching families remain future work.

## PrOntoQA and verification

The pinned upstream generator wrapper and evaluator converter are implemented;
see [external data preparation](EXTERNAL_DATA.md#prontoqa-v1).

Tests cover positive paths, label masking and padding, delimiter supervision,
source/config tampering, condition isolation, partition leakage, semantic
duplicates, quota omissions, bounded exhaustion, malformed external data,
prompt-field leakage, immutable output paths, and failure retention. Run them
without exposing the GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../token-efficient-reasoning/.venv/bin/python -m pytest -q
```

The preparations are CPU-validated; no new adapter or external model result is
claimed. Keep generated artifact directories when archiving the research; they
are intentionally Git-ignored.

Validation on 2026-09-29: 156 tests passed, with three opt-in model tests skipped;
Ruff and whitespace checks passed. The real pinned tokenizer was exercised by
the full answer-only preparation despite those optional tests being disabled.
All generated manifests verified, repeated PrOntoQA generation was byte-identical,
and the original M3 workspace still matched its frozen hash. No GPU work or
changes to the primary environment were introduced.
