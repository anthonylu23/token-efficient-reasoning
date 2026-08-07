# Milestone 2 smoke-test protocol

**Status:** Complete execution; gate decision `revise or hold`; validation bridge authorized

Milestone 2 exercises the complete training-to-evaluation workflow on Qwen3-0.6B. It is an
engineering and format-learnability gate, not the primary scientific experiment. The primary
comparison remains Milestone 3 on DeepSeek-R1-Distill-Qwen-1.5B.

At the M2 gate, M1's automated gates passed while its human review was pending. That review was
subsequently accepted with documented limitations. M2 engineering could proceed while
that review is completed; no M2 scientific result should be treated as final until the M1 review
artifact is signed.

## Frozen inputs

[`configs/m2/smoke_protocol.json`](../configs/m2/smoke_protocol.json) pins:

- the M1 root manifest hash and its `train` and `validation` splits;
- Qwen3-0.6B through the exact M0 checkpoint and tokenizer revisions;
- C2 as R0 full reasoning, C3 as R2 telegraphic reasoning, and C4 as R3 structured reasoning;
- one engineering training seed;
- the M0 reasoning budgets and answer allowance;
- manifest item order, assistant-target-only labels, and error-on-overflow behavior; and
- the versioned reasoning/answer target format.

The same M1 item IDs, prompts, answers, order, and example counts are used for C2-C4. Conditions
differ only in the declared reasoning rendering. Target-token exposure is measured and reported;
it is not assumed to be equal merely because optimizer updates and example counts are matched.

## Training sequence contract

Training and evaluation share the same model-native chat prefix. The prefix ends with
`<think>\n` and every prefix token receives the causal-language-model ignore label `-100`.
Supervision starts at the first reasoning token and covers:

```text
{R0|R2|R3}</think>
<answer>
{answer}</answer>
```

The prompt and target are tokenized as separate segments and then concatenated. This matches
inference, where the fully tokenized prompt already exists before the first target token is
generated. No item may be silently truncated: sequence-length calibration must choose a length
that contains every selected example or explicitly reject that run configuration.

The `token_efficient_reasoning.training` package provides:

- typed M2 protocol validation, including referenced manifest and split hashes;
- matched C2-C4 example construction;
- causal labels and attention masks;
- a condition-difference audit with an item-ID hash; and
- per-condition supervised and reasoning-token exposure summaries.

## Pinned-tokenizer profile

The complete 8,000-item training split was profiled with the pinned Qwen3-0.6B tokenizer. Counts
include the model-native prompt and the complete supervised target.

| Condition | Median sequence | p95 sequence | Maximum sequence | Supervised target tokens |
|---|---:|---:|---:|---:|
| C2 / R0 | 265.5 | 564 | 564 | 1,048,482 |
| C3 / R2 | 218 | 480 | 486 | 572,516 |
| C4 / R3 | 229 | 504 | 504 | 702,137 |

Prompt length is identical across conditions: median 136 tokens and maximum 358 tokens. A
512-token sequence limit is therefore invalid for matched M2 training because it cannot contain
every C2 example. GPU calibration must start at a sequence length of at least 564 tokens, with any
padding multiple accounted for explicitly.

C3 uses 45.4% fewer supervised target tokens than C2, and C4 uses 33.0% fewer. M2 keeps item
counts and optimizer updates matched and reports this exposure difference. It does not duplicate
compressed examples merely to equalize tokens, because doing so would change presentation
frequency and introduce a different confound.

## Calibrated QLoRA configuration

The smoke protocol freezes scientific inputs but intentionally does not freeze hardware-derived
QLoRA values before measuring them. Calibration on the RTX 3060 Ti selected the same configuration
for C2–C4:

- 4-bit NF4 with double quantization and BF16 compute;
- LoRA rank 16, alpha 32, dropout 0.05 over attention and MLP projections;
- sequence length 568, padded to a multiple of 8;
- microbatch 2 and gradient accumulation 8, for an effective batch of 16;
- paged 8-bit AdamW, learning rate `2e-4`, cosine decay, and 3% warmup; and
- one epoch, 500 optimizer steps, gradient checkpointing, and checkpoints every 100 steps.

A batch-4 calibration emitted recoverable allocator out-of-memory warnings and was rejected.
Batch 2 completed without those warnings and peaked at 3.07 GB allocated memory in the five-step
calibration. The final runs peaked between 3.23 GB and 3.54 GB.

## Execution gates

1. Materialize and audit matched C2-C4 training records from the frozen M1 manifest.
2. Profile sequence lengths under the pinned tokenizer.
3. Run a tiny overfit test that verifies masking, gradients, saving, and adapter reload.
4. Complete short C2-C4 QLoRA runs without non-finite loss or unhandled out-of-memory failure.
5. Evaluate C0-C4 across every frozen budget and validation depth.
6. Produce item-level generations, verifier diagnostics, exposure reports, frontiers, and an M2
   gate review from immutable run artifacts.

Advancement to M3 requires at least 99% C4 format parseability, zero post-enforcement budget
violations, evidence that C4 uses valid structured traces, successful fresh-process adapter reload,
and shallow-depth accuracy within the preregistered non-inferiority bound.

## Completed run evidence

The matched C2–C4 runs each trained on 8,000 items for 500 optimizer steps. All three had finite
loss, changed trainable parameters, and reproduced a fixed generation exactly after fresh-process
adapter reload. Their wall times were 31.8 minutes for C2, 28.0 minutes for C3, and 28.6 minutes
for C4. Each saved adapter is 56.3 MB.

Checkpoint resume was exercised separately by restoring C2 at step 400, continuing to step 405,
saving the resulting adapter, and verifying an identical fresh-process generation.
The experiment registry retains 14 completed attempts and one interrupted early C2 attempt rather
than silently dropping the unsuccessful run.

The full validation sweep contains 4,800 item-budget generations per condition: 800 validation
items at each reasoning budget `16, 32, 64, 128, 256, 512`. The aggregate results are:

| Condition | Accuracy | Region parse | Trace parse | Complete replay | Median reasoning tokens |
|---|---:|---:|---:|---:|---:|
| C0 frozen | 13.73% | 100.00% | 0.00% | 0.00% | 96.0 |
| C1 prompted | 10.35% | 97.04% | 0.00% | 0.00% | 96.0 |
| C2 full CoT | 51.79% | 100.00% | 73.31% | 51.12% | 54.5 |
| C3 telegraphic | 66.90% | 100.00% | 70.19% | 62.65% | 32.0 |
| C4 structured | 60.08% | 100.00% | 60.48% | 58.10% | 37.5 |

Every condition had zero post-enforcement reasoning-budget violations. C4 has no empty reasoning,
prompt copying, or repeated-line collapse. Its structured-trace parse rate rises from 0% at budget
16 to 99.875% at budgets 256 and 512; the all-budget rate is lower because deliberately truncated
rows are counted as trace failures.

## Gate decision

The M2 run is complete, but advancement is held:

- Training, adapter reload, C4 region parseability, and budget enforcement pass.
- The preregistered all-budget C4 structured-usability gate fails: trace parseability is 60.48%
  and canonical-prefix use is 60.40% across all budgets.
- Shallow non-inferiority fails at depth 2/budget 16 and depth 4/budgets 16 and 32. From budget 64
  upward, C4 and the strongest trained control are both 100% accurate at depths 2 and 4.
- At the time of the M2 gate, M1 automated criteria passed while human review was unsigned; the
  review was subsequently accepted with documented limitations.

The result is therefore `revise or hold`, not an engineering failure. Raw generations, paired
bootstrap comparisons, verifier diagnostics, SVG frontiers, and the deterministic failure sample
remain under `artifacts/m2/` and are intentionally ignored by Git. The concise, versionable exit
review is retained at `artifacts/M2_GATE_REVIEW.md`, matching the M0 and M1 convention.

The original decision is preserved. The subsequent
[`M2_GATE_ADJUDICATION.md`](../artifacts/M2_GATE_ADJUDICATION.md) authorizes only a one-seed,
validation-only 1.5B bridge. It documents why complete-trace parseability at budgets below the
canonical representation length is not a clean format-learnability measure, retains budgets 16
and 32 as mandatory diagnostics, and keeps final M3 test access locked behind the M1 human review,
the bridge gate, and a hashed M3 freeze.

## Reproduction

The main entry points are:

```bash
uv run ter-m2 validate
uv run ter-m2 materialize --output-dir artifacts/m2/prepared
uv run ter-m2 train --condition c2 --output-dir artifacts/m2/training/c2
uv run ter-m2 verify-reload artifacts/m2/training/c2
uv run ter-m2 evaluate \
  --condition c2 \
  --adapter artifacts/m2/training/c2/adapter \
  --output-dir artifacts/m2/evaluation/c2
```

Run `uv run ter-m2 --help` for the five-run report command and optional calibration overrides.
The report command loads only saved artifacts; it does not load a model or rerun training.
