# Milestone 3 validation-bridge gate review

**Decision:** PASS
**Evaluation completed:** 2026-07-28
**Report regenerated from saved artifacts:** 2026-08-06
**Scope:** DeepSeek-R1-Distill-Qwen-1.5B, validation split only

## Gate decision

The validation bridge passes every preregistered engineering and shallow-performance gate. No
held-out test, test-depth, or compositional-test model output was inspected. The separate M1 human
review was subsequently accepted with documented limitations. This is not a held-out scientific
result; test access still requires a validated clean-workspace freeze.

Authoritative evidence:

- `artifacts/m3/bridge/report/gate_summary.json`
- `artifacts/m3/bridge/report/manifest.json`; SHA-256
  `22906d136a39e9012efddd55d7707bbc2b02bca82fe9d6a56f694cb05ca20384`
- final analysis protocol SHA-256
  `adc19528c604e117ee374fdf4e3325b52736627a5e2c35daf7ab68a40ae0267e`
- original bridge execution protocol SHA-256
  `1c144aa7e75bad242c9efb9a2e15cdfa4c2d7a6a17bb412701ee5fd6a4310809`
- gate-summary SHA-256
  `ac03efe05275c007ba52c23de0d6a93c3b02bb09d9408942bd10e8e26f6176f9`

## Evaluation summary

Each condition contains 4,800 generations: 800 validation items at each reasoning budget
`16, 32, 64, 128, 256, 512`. The final hardware-safe batch schedule was `4, 4, 4, 2, 2, 2`.

| Condition | Accuracy | Region parse | Trace parse | Complete replay | Median reasoning tokens |
|---|---:|---:|---:|---:|---:|
| C0 | 13.73% | 99.98% | 0.00% | 0.00% | 64.0 |
| C1 | 11.88% | 95.58% | 0.00% | 0.00% | 61.0 |
| C2 / R0 | 53.08% | 100.00% | 73.31% | 51.15% | 54.5 |
| C3 / R2 | 64.71% | 100.00% | 70.21% | 62.69% | 32.0 |
| C4 / R3 | 61.52% | 100.00% | 60.52% | 58.23% | 37.5 |

All conditions recorded zero reasoning-budget violations.

## Gate checks

- C2–C4 training records are finite, changed trainable parameters, completed 500 optimizer
  updates, and reproduce their saved fixtures after fresh-process adapter reload.
- C4 feasible-budget region parseability is 100%.
- Every C4 depth-2/depth-4 trace-parse cell at budgets 64–512 is 100%.
- Empty reasoning, prompt copying, and repeated-line collapse rates are each 0% in the feasible
  C4 region.
- At depths 2 and 4 and budgets 64 and 128, C4 accuracy is 100%, the strongest trained control is
  also 100%, and the paired difference of 0 percentage points satisfies the predeclared
  -1 percentage-point non-inferiority margin.
- All five condition manifests and the report manifest reproduce their recorded file hashes.
- The saved bridge execution protocol and final report protocol differ only in the seven declared
  analysis-only fields listed below; there are no behavioral changes to data, models, prompts,
  training, decoding, budgets, seeds, batching, or gates.

## Pre-freeze statistical amendment

Before freeze and without inspecting held-out data, the analysis contract advanced from
`m3-primary-v1` to `m3-primary-v2`. The bridge report was regenerated from the unchanged saved
validation artifacts and accepted these analysis-only deltas:

- `primary_comparators` adds C3 and `secondary_comparators` becomes empty, placing C4/R3 versus
  C3/R2 at budgets 64 and 128 in the same eight-test Holm family as the six central comparisons;
- `central_primary_comparators` and `h5_comparator` explicitly preserve the central C0/C1/C2
  claim set and identify the two C3 cells as the H5 structure tests; and
- `primary_density_comparator` makes C2 full-CoT QLoRA the primary density reference, avoiding an
  automatic pass caused by C0 free-form output being outside the strict trace grammar; and
- `verified_density_zero_control_rule` retains explicit positive-over-zero and both-zero states for
  diagnostic comparisons without an outcome-dependent pseudocount.

The report also recomputes trace diagnostics offline from manifest-verified validation examples,
saved reasoning text, and saved forced-transition flags. This preserves exact canonical prefix
steps before a forced incomplete final line while leaving the full trace unparseable and adverse.
No bridge generation was rerun. The regenerated gate records `test_data_inspected: false`, passes
protocol compatibility, and binds the final protocol hash. See
`M3_PROTOCOL_STATISTICAL_REVIEW.md` for the scientific rationale.

## Corrected reporting defect

The first saved-artifact report incorrectly failed protocol compatibility because the comparator
treated JSON lists loaded from evaluation artifacts as unequal to tuple-backed fields returned by
`M3Config.to_dict()`. The report displayed identical values and identical protocol hashes while
classifying the sequence-valued fields as changes.

The comparator now normalizes both payloads through JSON before field comparison, and a regression
test covers list/tuple equivalence. The original false report is preserved at
`artifacts/m3/bridge/report_protocol_type_bug_20260728`; the accepted report was regenerated from
the unchanged evaluation and training artifacts. The full test suite passes with 94 tests passed
and 3 model tests skipped.

## Freeze readiness

After the repository owner's explicit approval on 2026-08-07, `ter-m3 validate` reports the bridge
gate, protocol compatibility, review-sample binding, and M1 human review as passed, with
`ready_to_freeze: true`. The next step is to commit the exact pre-freeze workspace, create the
content-bound freeze, validate it, and only then permit held-out execution.
