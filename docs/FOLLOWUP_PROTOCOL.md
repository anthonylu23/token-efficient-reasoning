# Follow-up experiments after M3

Status: CPU-only implementations validated on 2026-09-29; follow-up training has
not started. Answer-only targets are prepared for the full 8,000/800 train/validation
corpus; a 720-item arithmetic composition development corpus and 30 PrOntoQA
development items passed their audits. See [implementation and commands](FOLLOWUP_DATA.md).
All 1,000 SVAMP items are also ready; see [external data preparation](EXTERNAL_DATA.md).
M3 computation is complete; its [primary and supplemental reports](M3_EXECUTION.md)
were recovered on 2026-10-06 without changing frozen inputs or statistical rules.
The frozen outcome is `continue_to_generalization`, but C4-over-C3 H5 is unsupported;
retain C3 as a strong follow-up control. Follow-up training has not been launched.
New data, conditions, and analyses receive separate experiment identities; none are
inserted into M3's confirmatory family.

## Questions and controls

1. Does an explicit intermediate trace improve performance beyond task supervision?
   Add an answer-only SFT condition using the same training questions, answers, item
   order, optimizer updates, LoRA configuration, and seed blocks as the trace controls.
   Its supervised target closes the empty reasoning region and emits the answer.
   Measure actual reasoning use, correctness, and latency. Do not interpret its absent
   trace as an invalid attempted proof. Retain C2, C3, and C4 as named controls and
   report target-token exposure differences.
2. Does the advantage survive unseen compositions? Generate a new versioned corpus
   with explicit train/test exclusions of adjacent arithmetic operation pairs. A first
   development design reserves `(mul, sub)` and `(div, add)` for composition testing,
   while every primitive operation appears in training. Require held-out examples to
   contain a reserved pair. Keep surface templates shared for this comparison and
   put template transfer in a separate split. Audit sequence overlap, exact integer
   division, answer distributions, cancellations, and semantic duplicate hashes.
3. Does the advantage survive branching and multiple state variables? Add a small
   two-register program family with assignments and conditional updates. Store each
   executed transition, branch predicate, and full register state; derive every
   representation from that common execution. Count executed depth separately from
   program length. Verify counterfactual inputs that change a branch. Train all matched
   conditions on this family; distinguish representation learning from zero-shot task
   exposure.

Develop generators and parsers with train/development fixtures. Before evaluation,
freeze seeds, template and composition partitions, item counts, answer rules, budgets,
and statistical contrasts. Use exact-token profiles to include both feasible and
constrained budgets. Retain full budget frontiers; feasibility-conditioned results
are supplemental because their item populations vary by representation.

The first follow-up should prioritize answer-only supervision and composition transfer.
Branching is the next scope increment. Pause-token training is conditional on evidence
of deeper failures despite sufficient representation space; it is not justified solely
by a truncation-driven accuracy gap. Keep RL and tokenizer expansion deferred.

## External evaluation

Use SVAMP for arithmetic word problems and PrOntoQA for logic. Keep the checkpoint and
tokenizer fixed and evaluate all C0–C4 conditions, not only the validation winner.
The official source revisions are pinned in
[`external_sources.json`](../configs/followup/external_sources.json), including the
SHA-256 of the SVAMP data file. These pins do not constitute a completed evaluation freeze.

- [Official SVAMP repository](https://github.com/arkilpatel/SVAMP): use the complete
  1,000-item challenge set, joining `Body` and `Question` without exposing equations or
  gold answers in prompts. Preserve upstream IDs and exact numeric answer scoring.
- [Official PrOntoQA repository](https://github.com/asaparov/prontoqa): use the original
  task from the pinned `v1` branch. Freeze the generation command, seeds, hop distribution,
  and answer vocabulary before
  model evaluation. Avoid importing another project's model outputs as input data.

Pin revisions and raw-file hashes, preserve licenses, and test conversion on artificial
fixtures. Use budgets 64, 128, 256, and 512, the existing 16-token answer allowance,
and paired decoding seeds. Include original frozen-model behavior so any loss of
general capability after task-specific SFT remains visible. Validate that the answer
allowance accommodates all labels without adding reasoning space.

External results are exploratory transfer measurements. Do not apply the synthetic
R0/R2/R3 parser to arbitrary math or logic prose and call parser failure a reasoning
error. Report exact-answer accuracy, reasoning/answer tokens, forcing, and latency;
add process verification only where an independently tested benchmark verifier exists.
Do not tune prompts on the external evaluation set. Any new prompt development uses
a disjoint development source and is recorded as a separate experimental condition.

## Interpretation and related work

The defensible target is a controlled study of executable representations under hard
budgets, including transfer limits. Shorter reasoning and depth-per-token evaluation
already have substantial prior work:

- [MUTO](https://aclanthology.org/2026.acl-long.1386/) studies token-level utility on
  the same 1.5B backbone; aggregate compression alone is not a distinctive claim.
- [ReEfBench](https://aclanthology.org/2026.acl-long.931/) evaluates logical depth and
  process efficiency. Contrast its neural parsing with our paired canonical renderings
  and deterministic replay, acknowledging the narrower grammar coverage here.
- [When Internalization Fails](https://aclanthology.org/2026.findings-acl.734/) motivates
  external transfer checks and accounting for tokens generated after the thinking region.
- [Stable Reinforcement Learning for Efficient Reasoning](https://arxiv.org/abs/2505.18086)
  motivates resolving supervised findings before introducing length rewards.

Report null and adverse results. If C3 wins, describe the result as a comparison of two
symbolic encodings. If improvements occur only when verbose targets are cut off, describe
the benefit as executing more of the learned trace within the cap. Claim broader reasoning
generalization only with held-out compositions, depths, or tasks that support it.
