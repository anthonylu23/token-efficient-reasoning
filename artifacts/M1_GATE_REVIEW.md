# Milestone 1 gate review

Date: 2026-07-21

## Decision

The automated Milestone 1 gates pass. Final milestone acceptance remains pending human review of
the generated stratified sample; this record does not treat agent inspection as human sign-off.

## Frozen configuration

- Dataset: `m1-controlled-depth-v1`, generator version `1.0.0`.
- Configuration: `configs/m1.json`.
- Families: multi-operation arithmetic and labeled directed-graph traversal.
- Depths: 2, 4, 8, and 12 in the ordinary distributions; depth 16 held out from training.
- Qwen tokenizer: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- DeepSeek tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.

## Automated evidence

- Accepted examples: 12,600 total; 8,000 train, 800 validation, 2,000 ordinary test,
  800 compositional test, and 1,000 depth-generalization test.
- Replay verification: 12,600 of 12,600 canonical examples and all four renderings pass.
- Counterfactual verification: 25,200 of 25,200 output and operand/edge mutations are rejected.
- Split audit: zero seed, exact-instance, exact-prompt, value-signature, withheld-template, or
  train/depth-test collisions.
- Compression: 32.0% overall median R3 reduction from R0 under both pinned tokenizers, above the
  frozen 30% threshold. Every accepted R2 and R3 trace is shorter than R0 under both tokenizers.
- Rejected attempts: 258, comprising 253 repeated value signatures and five repeated semantic
  instances. All are retained in `data/m1/rejections.jsonl`.
- Reproducibility: independent regeneration produced the identical root manifest SHA-256
  `2bca4e9e3022f0fde902cbaea80d73e5ef71dfc8ecc45af06d807883a8005917`.
- Tests: 48 CPU tests pass and the two explicit pinned-tokenizer tests pass. The default suite
  skips those two network/model tests unless `RUN_MODEL_TESTS=1` is set.

## Remaining review

`data/m1/review_sample.json` selects one item per split, family, and depth and includes every
R0–R3 rendering. A human reviewer should verify problem clarity, distractor irrelevance, identity
preservation, arithmetic/edge replay, and whether the compressed forms remain legible. Review
status and notes should be recorded without editing the immutable generated dataset; a separate
signed review artifact should reference item IDs and the root manifest hash.

After that sign-off, update the decision and gate status. Until then, M2 preparation may use the
pipeline for engineering work, but the M1 scientific gate should be described as conditionally
complete rather than final.
