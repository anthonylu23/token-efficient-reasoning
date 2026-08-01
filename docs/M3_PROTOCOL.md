# Milestone 3 primary experiment protocol

**Status:** Validation bridge passed; final test access locked pending M1 human review and freeze

Milestone 3 tests whether structured-sketch SFT shifts the accuracy-versus-reasoning-budget
frontier on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. The experiment preserves the M2 result,
uses validation only for the bridge, and prevents final held-out evaluation until the protocol
freeze validates every prerequisite.

## M2 transition

M2 remains `revise or hold`. Its result is not retroactively relabeled as passing. The
[`M2_GATE_ADJUDICATION.md`](../artifacts/M2_GATE_ADJUDICATION.md) authorizes a one-seed,
validation-only 1.5B bridge because the original all-budget trace-parse gate includes budgets
shorter than a complete canonical R3 trace. Budgets 16 and 32 remain mandatory diagnostic cells.

The final M3 freeze requires:

1. a completed human decision in `artifacts/M1_HUMAN_REVIEW.json`;
2. a passing validation bridge at `artifacts/m3/bridge/report/gate_summary.json`; and
3. valid hashes for the protocol, QLoRA configuration, M1 manifest, every split, run plan, and
   current evaluator workspace.

`ter-m3 freeze` fails closed until these requirements pass. Final evaluation commands additionally
require a valid freeze and an exact matching run ID and specification.

## Frozen scientific design

[`configs/m3/protocol.json`](../configs/m3/protocol.json) declares:

- the exact primary checkpoint and tokenizer revision inherited from M0;
- C2 full-CoT/R0, C3 telegraphic/R2, and C4 structured/R3 training;
- training seeds `314159`, `271828`, and `161803`;
- paired decoding seeds `1729`, `2718`, and `3141`;
- reasoning budgets `16, 32, 64, 128, 256, 512`;
- primary practical budgets 64 and 128;
- fixed-final-update checkpoint selection at 500 optimizer updates;
- the 1 percentage-point accuracy non-inferiority margin;
- a 1 percentage-point invalid-transition margin;
- 10,000 paired bootstrap resamples;
- item-ID bootstrap clusters, with the three paired replicate outcomes averaged within each item;
- six one-sided primary C4 comparisons on the ordinary test split (C0, C1, and C2 at
  budgets 64 and 128), controlled at family-wise alpha 0.05 with Holm's procedure;
- C3, held-out depth, compositional, family, and depth comparisons as descriptive secondary
  analyses;
- the 80% reliable-depth threshold, 30% matched-accuracy token-reduction threshold, 2
  percentage-point primary accuracy improvement, and 1.5× density threshold; and
- ordinary test, held-out depth-16, and compositional test splits.

The three replicate blocks pair one training seed and one decoding seed. Each block evaluates
C0–C4, with C0 and C1 regenerated under the block's decoding seed. This produces nine trained
adapters, 45 split-level evaluation runs, and 342,000 item generations.

Evaluation uses a deterministic hardware-safe batch schedule: batch 4 for budgets 16–64 and
batch 2 for budgets 128–512. Every
row has a persistent device RNG stream derived from its frozen item seed and shared across the
reasoning and answer phases. Delimiter detection, EOS handling, reasoning budgets, answer
allowances, and forced transitions remain independent per row. A repeated 20-generation
batch-four fixture reproduced every reasoning token, answer token, correctness outcome, trace
diagnostic, and item seed exactly. Compared with single-item evaluation, batching preserved all 20
correctness outcomes, matched 19 of 20 complete token sequences, reduced fixture wall time by
roughly 3.5×, and stayed below 4.8 GB reserved VRAM. The one cross-mode token difference is treated
as a floating-point batching effect; every condition uses the same frozen batch size at a given
budget.

## Hardware-derived QLoRA configuration

Primary-tokenizer profiling covers all 8,000 training items and yields:

| Condition | Median sequence | p95 sequence | Maximum sequence |
|---|---:|---:|---:|
| C2 / R0 | 260.5 | 559 | 559 |
| C3 / R2 | 213 | 475 | 481 |
| C4 / R3 | 224 | 499 | 499 |

The initial [`configs/m3/qlora.json`](../configs/m3/qlora.json) therefore uses a 560-token aligned
limit, NF4 double quantization, BF16 compute, LoRA rank 16/alpha 32, microbatch 1, and gradient
accumulation 16. Calibration may change hardware-derived batch and accumulation values before the
bridge. It may not change the scientific design or the effective batch without documenting the
reason.

Calibration selected microbatch 1 after it completed five updates with 4.13 GB peak allocated and
5.57 GB peak reserved memory and reproduced its reload fixture exactly. Microbatch 2 was rejected
despite modestly higher throughput because it reserved 7.34 GB and emitted a recoverable 512 MiB
allocator failure. The versionable evidence is retained in
[`M3_CALIBRATION.md`](../artifacts/M3_CALIBRATION.md).

A full C0 validation stress run later reached 7.77 GB device memory under batch 4 and emitted a
recoverable allocator failure on entering budget 512. A mixed-schedule rerun then completed C0,
but C1's 490-token few-shot prompts produced another recoverable allocation failure under batch 4
at budget 128. Batch 4 is therefore prohibited for budgets 128, 256, and 512. Batch 2 completed
five full-budget generations in a saved 20-item fixture, two simultaneous
353-input/512-reasoning-token longest-validation rows, and a validation-derived
560-token-per-row allocator-envelope check. The worst accepted fixture reserved 5.06 GB, leaving
more than 3 GB of device capacity. No held-out split was accessed for calibration.

The materialized audit is reproducible with:

```bash
uv run ter-m3 validate
uv run ter-m3 materialize --output-dir artifacts/m3/prepared
```

The current materialization confirms identical C2–C4 item IDs and invariants, 8,000 training
items, 800 validation items, and maximum sequence length 559.

## Validation bridge

The bridge uses the first declared training/decoding seed pair. It performs:

1. primary-checkpoint memory and throughput calibration;
2. a tiny overfit, save, and fresh-process reload check;
3. matched full C2–C4 training at 500 updates;
4. C0–C4 validation evaluation at every budget; and
5. saved-artifact-only bridge reporting.

The bridge gate requires finite training, changed adapter parameters, fresh-process reload,
zero budget violations, at least 99% feasible-budget C4 region and shallow trace parseability,
shallow non-inferiority at budgets 64 and 128, no collapse, and evidence that no test split was
accessed.

Example commands:

```bash
uv run ter-m3 train \
  --phase bridge \
  --condition c4 \
  --training-seed 314159 \
  --output-dir artifacts/m3/bridge/training/c4

uv run ter-m3 evaluate \
  --phase bridge \
  --condition c4 \
  --split validation \
  --decoding-seed 1729 \
  --adapter artifacts/m3/bridge/training/c4/adapter \
  --output-dir artifacts/m3/bridge/evaluation/c4
```

Bridge evaluation rejects every split except `validation`.

The complete sequential sweep and saved-artifact report can be run with:

```bash
bash scripts/run_m3_bridge_evaluations.sh
```

The script exits on the first failed condition, records lifecycle events through `ter-m3`, and
uses one loaded model at a time.

### Bridge result

The final bridge sweep completed all 24,000 C0–C4 validation generations on 2026-07-28 under the
frozen `4, 4, 4, 2, 2, 2` batch schedule. Every condition manifest verifies, all training and
format gates pass, no reasoning-budget violation occurred, and C4 satisfies the shallow
non-inferiority gate at budgets 64 and 128. The saved-artifact-only gate report passes and records
that no held-out test split was inspected.

The first report run exposed and preserved a comparator defect in which JSON lists compared
unequal to tuple-backed configuration fields despite identical serialized protocols. The
comparator now normalizes both payloads through JSON, and the accepted report was regenerated from
the unchanged saved artifacts. See
[`M3_BRIDGE_GATE_REVIEW.md`](../artifacts/M3_BRIDGE_GATE_REVIEW.md) and
[`M3_BRIDGE_ENGINEERING.md`](../artifacts/M3_BRIDGE_ENGINEERING.md).

The bridge pass does not unlock test access by itself. The remaining prerequisite is the
34-item M1 human review in `artifacts/M1_HUMAN_REVIEW.md` and
`artifacts/M1_HUMAN_REVIEW.json`.

## Final run safeguards

The freeze writes `freeze.json`, `run_plan.json`, and a recursive manifest. The snapshot embeds the
protocol, QLoRA settings, checkpoint, split hashes, prerequisite evidence, repository workspace
digest, environment, and hardware.

Primary training and evaluation:

- require the output directory name to equal a frozen run ID;
- compare condition, split, seeds, and adapter identity to the frozen run specification;
- forbid item, update, and budget overrides;
- record planned, running, complete, failed, interrupted, and excluded attempts in an append-only
  registry; and
- retain immutable item-level records and raw generations.

Evaluation supports explicit resume. A resumed run must have the identical resolved configuration,
rejects duplicate or foreign item rows, archives the prior failure record, and continues only
missing item-budget keys. Training resumes only from a `checkpoint-N` directory inside the same
frozen run directory; it retains prior failure evidence under `attempts/` and preserves the frozen
run ID. Both primary execution and final reporting compare the current repository state to the
frozen workspace digest and fail on any change.

## Reporting

The primary report will be generated from saved artifacts without loading a model. It will include
paired item-level intervals, task-family and depth breakdowns, seed variability, accuracy and
verified-density frontiers, matched-accuracy token reduction, invalid and forced-transition rates,
latency, throughput, memory, and stratified failure examples. It will report null and adverse
results and select exactly one M3 decision outcome from the roadmap.

The primary estimand is the C4-minus-comparator exact-answer difference at a fixed reasoning
budget. Outcomes are paired by replicate block and item ID. For confidence intervals, the three
replicate differences for each item are averaged first, then item IDs are sampled with replacement.
The six primary p-values use paired sign randomization over those item-level averages and Holm's
step-down correction. Per-generation accuracy and per-problem aggregate behavior are both retained;
no best-of-seed or best-of-sample selection is permitted.

Matched-accuracy token reduction uses the lowest frozen C4 budget whose pooled per-generation
accuracy is at least a comparator frontier point. The reduction is computed from pooled median
generated reasoning-token counts. Maximum reliably solved depth is the greatest frozen controlled
depth with pooled accuracy of at least 80%.

After the prerequisites pass and the freeze is created, the complete sequential, resumable primary
matrix can be executed with:

```bash
python scripts/run_m3_primary.py --stage all
```

The runner records the required uptime, memory, disk, GPU, and process preflight; rejects a busy
GPU by default; executes only one model job at a time; fresh-process verifies every adapter; resumes
interrupted training/evaluation from preserved checkpoints and item rows; and builds the final
saved-artifact-only report after all 54 planned runs finish.

The statistical contract was finalized after the validation generations but before any test
access. The bridge report therefore records both protocol hashes and a field-level compatibility
artifact. Only the declared analysis-only fields may differ; a change to any model, data, prompt,
budget, seed, decoding, batching, training, or gate field fails readiness. The bridge report must
be regenerated from its saved validation artifacts under the final protocol before freeze.
