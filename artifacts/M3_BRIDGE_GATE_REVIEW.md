# Milestone 3 validation-bridge gate review

**Decision:** PASS
**Evaluation completed:** 2026-07-28
**Report regenerated from saved artifacts:** 2026-07-30
**Scope:** DeepSeek-R1-Distill-Qwen-1.5B, validation split only

## Gate decision

The validation bridge passes every preregistered engineering and shallow-performance gate. No
held-out test, test-depth, or compositional-test data was inspected. This decision authorizes the
M3 protocol freeze only after the separate M1 human review is complete; it is not a held-out
scientific result.

Authoritative evidence:

- `artifacts/m3/bridge/report/gate_summary.json`
- `artifacts/m3/bridge/report/manifest.json`
- protocol SHA-256
  `1c144aa7e75bad242c9efb9a2e15cdfa4c2d7a6a17bb412701ee5fd6a4310809`
- gate-summary SHA-256
  `8ff6e78f0efb4c3a4d4ca1ac57931b5eefd7c8db28fd3d92063bc34147ce9f12`

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
- The saved bridge protocol and current report protocol have the same SHA-256 and no behavioral
  or analysis-only field differences.

## Corrected reporting defect

The first saved-artifact report incorrectly failed protocol compatibility because the comparator
treated JSON lists loaded from evaluation artifacts as unequal to tuple-backed fields returned by
`M3Config.to_dict()`. The report displayed identical values and identical protocol hashes while
classifying the sequence-valued fields as changes.

The comparator now normalizes both payloads through JSON before field comparison, and a regression
test covers list/tuple equivalence. The original false report is preserved at
`artifacts/m3/bridge/report_protocol_type_bug_20260728`; the accepted report was regenerated from
the unchanged evaluation and training artifacts. The full test suite passes with 85 tests passed
and 3 model tests skipped.

## Remaining prerequisite

`ter-m3 validate` reports:

- bridge gate passed: `true`;
- bridge protocol compatible: `true`;
- M1 human review passed: `false`; and
- ready to freeze: `false`.

A human reviewer must inspect all 34 items in `data/m1/review_sample.json`, complete
`artifacts/M1_HUMAN_REVIEW.md`, and update `artifacts/M1_HUMAN_REVIEW.json`. An agent may validate
the record afterward but must not provide the human sign-off.
