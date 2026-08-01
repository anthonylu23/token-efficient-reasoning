# Milestone 1 human review

**Status:** PENDING HUMAN SIGN-OFF

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

- [ ] The problem statement is clear and has one intended answer.
- [ ] Any distractor is explicitly irrelevant and does not alter the solution.
- [ ] The recorded answer follows from the problem instance.
- [ ] Every R0–R3 rendering preserves the same ordered transitions.
- [ ] Arithmetic operations or labeled graph edges replay correctly.
- [ ] R2 and R3 remain legible enough to serve as supervised targets.
- [ ] No item reveals an unintended split, template, or answer artifact.

Record any failed item IDs and notes below. Do not edit the generated dataset or its embedded
`pending_human_review` fields; this separate artifact is the review record.

## Exceptions and notes

None recorded.

## Sign-off

- Reviewer:
- Review date:
- Decision: `PENDING` (`ACCEPT`, `ACCEPT WITH DOCUMENTED LIMITATIONS`, or `REJECT`)
- Signature or explicit approval reference:

The M1 gate remains conditionally complete until a human fills this section. An agent may prepare
or validate this artifact but must not substitute its own inspection for human sign-off. After
signing, the reviewer should also set the JSON record to `status: "complete"`, record an accepted
decision, identify themselves, provide the review time and approval reference, and set
`reviewed_item_count` to 34.
