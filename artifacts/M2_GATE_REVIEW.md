# Milestone 2 gate review

**Decision:** REVISE OR HOLD

## Scope

Qwen3-0.6B smoke-test training and fixed-budget validation for C0-C4. This is an engineering and format-learnability gate, not the primary scientific result.

## Run evidence

- C0: `final-c0-validation`; manifest `59bb736373b3efe3dbcf23531ee38819c66e333a568640dff8c13ccc778eac0f`.
- C1: `final-c1-validation`; manifest `bc9a4ad86444913af6b2ff094fdfd9a11ed53235419fce964ac9ff44b3ec7260`.
- C2: `final-c2-validation`; manifest `8081a99454126c23efaf4c6d8767171f5b1b16665e325cdf2bdf1f952913e048`.
- C3: `final-c3-validation`; manifest `1e498f04f49af42a6565f27050847a51b2828b9c544c7e3c93f04ac87a96f8ed`.
- C4: `final-c4-validation`; manifest `50a518654fbbe1dfbd253ac82e95ed0b085f7be6635b542648cec1333771bb42`.

## Acceptance criteria

- C2-C4 training/reload gate: **True**.
- C4 region parseability: **100.000%** (required at least 99%).
- Post-enforcement budget violations: **0**.
- C4 structured-trace usability: **False** (trace parse 60.479%; canonical-prefix 60.396%; complete replay 58.104%).
- Shallow non-inferiority point-estimate gate: **False**.
- M1 human review complete: **False**.

## Known deviations and hold reasons

- The all-budget structured-usability calculation includes deliberately truncated low-budget generations; budget-stratified metrics remain in `aggregate.csv`.
- Failed shallow comparisons and their paired bootstrap intervals remain in `shallow_noninferiority.json`.
- The prerequisite M1 human review is unsigned.
- No M3 training configuration is selected while the advancement gate is held.

## Interpretation

A failed scientific threshold does not erase successful pipeline execution. The complete paired comparisons and bootstrap intervals are retained in `shallow_noninferiority.json`; same-item failure contrasts are retained in `failure_analysis.json`.

## Reproduction

All tables and SVG plots in the report directory are generated from saved item-level evaluation artifacts without loading a model or rerunning training.
