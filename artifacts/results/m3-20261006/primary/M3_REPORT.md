# Milestone 3 primary experiment report

**Decision:** `continue_to_generalization`

## Engineering validity

- Complete training/reload runs: **9**.
- Complete evaluation runs: **45**.
- Final generation count: **342,000**.
- Every input run passed freeze, manifest, configuration, item-pairing, summary recomputation, budget-enforcement, registry, and provenance checks.

## Statistical evidence

- Minimum viable result: **False**.
- Primary success result: **True**.
- H5 (structured R3 versus telegraphic R2) supported at one or more primary budgets: **False**.
- Primary family-wise alpha: **0.050**, corrected with **holm** over 8 hypotheses.
- `distractor_answer_sensitivity.json` repeats primary-budget accuracy comparisons after excluding arithmetic items whose irrelevant marker count equals the answer; this does not replace primary inference.

### Primary matched-budget comparisons

| Budget | Claim | Comparator | C4 accuracy | Comparator accuracy | Difference (95% CI) | Holm reject | Density ratio |
|---:|---|---|---:|---:|---:|---|---:|
| 64 | central | C0 | 51.400% | 9.783% | +41.617% [+39.617%, +43.667%] | True | ∞ (positive / zero) |
| 64 | central | C1 | 51.400% | 8.350% | +43.050% [+41.050%, +45.084%] | True | ∞ (positive / zero) |
| 64 | central | C2 | 51.400% | 38.233% | +13.167% [+11.683%, +14.667%] | True | 1.479 |
| 64 | H5 structure | C3 | 51.400% | 51.483% | -0.083% [-0.700%, +0.500%] | False | 0.749 |
| 128 | central | C0 | 75.333% | 15.333% | +60.000% [+58.083%, +61.933%] | True | ∞ (positive / zero) |
| 128 | central | C1 | 75.333% | 15.883% | +59.450% [+57.567%, +61.300%] | True | ∞ (positive / zero) |
| 128 | central | C2 | 75.333% | 63.517% | +11.817% [+10.417%, +13.217%] | True | 1.539 |
| 128 | H5 structure | C3 | 75.333% | 96.033% | -20.700% [-22.467%, -18.967%] | False | 0.785 |

## Scientific interpretation

C4 improves a fixed-budget point or clears the matched-accuracy token-reduction threshold without a material validity regression.

## Interpretation safeguards

- The complete accuracy-versus-budget frontier is the headline result; no accuracy/token scalar is substituted for it.
- Confidence intervals resample item IDs and average paired replicate-block outcomes within each item before resampling.
- C4-versus-C3 at budgets 64 and 128 directly tests H5 (structured R3 versus telegraphic R2) in the same eight-test Holm family as the six central comparisons; C3 at other budgets/splits and all held-out depth/compositional comparisons remain descriptive secondary analyses.
- A positive C4 verified density over zero C0 density is reported as an extended infinite rate ratio and passes the finite ratio threshold; if both densities are zero, the ratio is undefined and the criterion fails.
- Null, adverse, and unsuccessful attempted runs remain in the machine-readable artifacts and experiment registry.
- All report files were generated from saved artifacts without loading a model.

## Reproduction

Machine-readable aggregates, intervals, paired comparisons, task/depth breakdowns, seed variability, matched-accuracy reductions, system measurements, and qualitative paired examples are colocated with this report.
