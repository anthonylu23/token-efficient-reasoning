# Milestone 3 pre-freeze statistical amendment review

**Status:** accepted into the protocol before freeze and before held-out evaluation
**Data access:** saved validation bridge artifacts only; no held-out item or model output inspected

## Amendment

Three analysis-contract defects were corrected in `m3-primary-v2`:

1. **Primary density reference.** Verified density remains pooled verified-prefix transitions
   divided by pooled reasoning tokens. The 1.5× primary threshold now compares C4 with C2 full-CoT
   QLoRA, because both trained conditions emit controlled, verifier-readable grammars. C0 remains a
   diagnostic and cannot make the success gate automatic merely because free-form output is outside
   the strict trace grammar. Reported zero-denominator comparisons still use explicit
   `positive_treatment_over_zero_control`/`both_zero` states and no pseudocount.
2. **H5 structure test.** C4/R3 versus C3/R2 exact-answer accuracy at reasoning-token caps 64 and
   128 is now primary. These two one-sided paired sign-randomization tests join the existing six
   C4-versus-C0/C1/C2 tests in one eight-test Holm family at family-wise alpha 0.05. This preserves
   the six central comparisons as primary while paying the multiplicity cost for testing H5.
   C3 at other budgets and splits, plus task/depth breakdowns, remains descriptive.
3. **Budget-truncated prefixes.** Saved reasoning is rescored offline so a forced, incomplete final
   line cannot erase earlier exact canonical transitions. The full trace remains a parse failure and
   an adverse validity event, while verified progress before the fragment is retained.

The central primary-success rule is otherwise unchanged: C3 significance is reported separately
as H5 support and is not silently added as a new conjunct to the central success criterion.

## Implementation checks

- Protocol validation pins the analysis version, central comparator set, H5 comparator, common
  primary comparator family, and zero-control rule.
- Final reporting emits primary-claim labels, all eight Holm results, a separate H5 summary, and
  explicit finite/positive-over-zero/both-zero density states.
- Tests cover the eight-test family, C3/H5 labels, H5 reporting, and both zero-denominator boundary
  outcomes.
- Bridge compatibility classifies these fields as analysis-only. The accepted bridge report must
  be regenerated from unchanged saved validation artifacts under the amended protocol before
  freeze; no model execution is required.
