# Milestone 2 gate adjudication

**Date:** 2026-07-27
**Original gate decision:** `REVISE OR HOLD`
**Next action:** validation-only 1.5B bridge

## Decision

Preserve the original M2 result and its failed gates. Do not relabel M2 as passed. Proceed with
implementation and a one-seed, validation-only DeepSeek-R1-Distill-Qwen-1.5B bridge before
freezing or inspecting any M3 test result.

Final M3 test evaluation remains prohibited until:

1. the separate M1 human-review artifact is signed;
2. the bridge gate below passes;
3. the M3 protocol, run plan, configurations, manifests, evaluator revision, and statistical
   rules are frozen and hashed.

## Rationale

The M2 all-budget structured-usability gate combines format learning with deliberate truncation.
Under the pinned primary tokenizer, canonical depth-2 R3 traces have median lengths of 22 tokens
for graph traversal and 24 tokens for arithmetic. A 16-token reasoning budget therefore cannot
contain most complete depth-2 R3 traces. At depth 4, the corresponding medians are 44 and 49
tokens, so budgets 16 and 32 are also structurally unable to contain most complete traces.

This does not erase the observed low-budget accuracy regressions. Budgets 16 and 32 remain
mandatory M3 diagnostic cells. It does justify separating complete-trace learnability at feasible
budgets from forced-prefix behavior below the representation's minimum length.

## Bridge gate

The validation-only bridge passes only when all of the following hold:

- C2, C3, and C4 complete one matched training seed without non-finite loss or unhandled
  out-of-memory failure.
- Every adapter changes trainable parameters and reproduces its fixed reload fixture in a fresh
  process.
- All conditions have zero post-enforcement reasoning-budget violations.
- C4 region parseability is at least 99% at budgets 64 and above.
- C4 structured-trace parseability is at least 99% at budgets 64 and above for depths 2 and 4.
- At budgets 64 and 128, C4 is no more than one absolute percentage point below the strongest
  comparable trained control on depths 2 and 4.
- C4 shows no empty-reasoning, prompt-copy, or repeated-line collapse.
- No validation outcome is used to change the test manifests, primary hypotheses, success
  thresholds, or representation definitions.

Failure returns the project to representation revision or a documented exploratory design. It
does not authorize test inspection.

## Scope boundary

The bridge may resolve hardware-derived batch size, gradient accumulation, precision, and
checkpoint frequency. It may not select a test-facing configuration by inspecting test,
depth-generalization, or compositional results.
