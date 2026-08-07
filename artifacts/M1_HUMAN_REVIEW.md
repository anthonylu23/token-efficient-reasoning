# Milestone 1 human review

**Status:** COMPLETE — ACCEPTED WITH DOCUMENTED LIMITATIONS

## Frozen evidence

- Review sample: `data/m1/review_sample.json`
- Review-sample SHA-256: `5bce54eb70026ad1dc375228bbcffc5131dc697d185b7131bbb19bb0fde7752d`
- Machine-readable decision: `artifacts/M1_HUMAN_REVIEW.json`
- Dataset manifest: `data/m1/manifest.json`
- Dataset-manifest SHA-256: `2bca4e9e3022f0fde902cbaea80d73e5ef71dfc8ecc45af06d807883a8005917`
- Sample size: 34 items covering every split, task family, configured depth, and R0–R3
  representation.

## Review checklist

The reviewer should inspect every selected item and confirm:

- [x] The problem statement is clear and has one intended answer.
- [x] Any distractor is explicitly irrelevant and does not alter the solution.
- [x] The recorded answer follows from the problem instance.
- [x] Every R0–R3 rendering preserves the same ordered transitions.
- [x] Arithmetic operations or labeled graph edges replay correctly.
- [x] R2 and R3 remain legible enough to serve as supervised targets.
- [ ] No item reveals an unintended split, template, or answer artifact. One answer-artifact
  exception is accepted and documented below.

Record any failed item IDs and notes below. Do not edit the generated dataset or its embedded
`pending_human_review` fields; this separate artifact is the review record.

## Exceptions and notes

Pre-sign-off advisory audit: item
`arith-test_compositional-d4-0216f06fa91b412b-aaf4344146` has an explicitly irrelevant marker
count of 22 and final answer 22. The same coincidence occurs in 103 of 6,300 arithmetic items
(1.63%): 62 train, 4 validation, 20 ordinary test, 10 compositional test, and 7 depth-test items.
The generated v1 corpus remains immutable; the M3 protocol must report a sensitivity analysis that
excludes these items. The human reviewer accepted this as a documented limitation. The preregistered M3 sensitivity
analysis excludes affected ordinary-test items without replacing full-split primary inference.

## Sign-off

- Reviewer: Anthony Lu (repository owner)
- Review date: 2026-08-07
- Decision: `ACCEPT WITH DOCUMENTED LIMITATIONS`
- Signature or explicit approval reference: Explicit user approval in Prime Agent session
  `019fd4df-b5ff-729a-91bb-39351f6c719f` on 2026-08-07.

The human approval is recorded without modifying the immutable generated dataset. The accepted
answer-artifact limitation remains bound to the M3 sensitivity analysis.
