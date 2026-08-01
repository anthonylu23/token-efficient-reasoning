# Milestone 3 primary-checkpoint calibration

**Date:** 2026-07-27
**Checkpoint:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
**Selected configuration:** `configs/m3/qlora.json`

## Preflight

- Host: `anthonypc`
- GPU: NVIDIA GeForce RTX 3060 Ti, 8,192 MiB
- Available system memory before calibration: approximately 9.8 GiB
- Free repository filesystem space: approximately 1.3 TiB
- Competing GPU compute processes: none

## Sequence profile

All 8,000 matched training examples fit the 560-token aligned sequence limit. Maximum sequence
lengths are 559 for C2, 481 for C3, and 499 for C4.

## Calibration comparison

Both trials used C2, 80 items, five optimizer updates, effective batch 16, NF4 double
quantization, BF16 compute, and the frozen LoRA configuration.

| Trial | Microbatch | Accumulation | Wall time | Peak allocated | Peak reserved | Result |
|---|---:|---:|---:|---:|---:|---|
| `c2-b1-5step` | 1 | 16 | 36.7 s | 4.13 GB | 5.57 GB | Selected |
| `c2-b2-5step` | 2 | 8 | 32.9 s | 5.01 GB | 7.34 GB | Rejected |

Microbatch 2 emitted a recoverable allocator failure while attempting an additional 512 MiB
allocation. Although it completed, its reserved memory left insufficient operating margin for a
long unattended run. Microbatch 1 completed without an allocator warning.

The selected microbatch-1 adapter changed trainable parameters and reproduced the saved
eight-token generation exactly after a fresh-process reload.

A separate C4 overfit check trained on 20 items for 20 updates. Logged loss fell from 0.6902 on
the first update to 0.0074 in the final interval, trainable parameters changed, peak allocated
memory was 3.94 GB, and the saved eight-token fixture reproduced exactly after a fresh-process
reload. This confirms structured-target labels, masking, optimization, saving, and adapter
application on the primary checkpoint.

## Decision

Use microbatch 1, gradient accumulation 16, and effective batch 16 for C2–C4 bridge and primary
training. Retain the rejected microbatch-2 artifacts and configuration as calibration evidence.

## Evaluation batch calibration

The original all-budget C0 validation stress run used batch 4. It completed all 4,000 rows through
budget 256, then emitted a recoverable allocator failure at budget 512 while requesting
805,306,368 bytes with only 71,565,312 bytes free. Twenty-eight budget-512 rows were retained
before the run was intentionally interrupted. This attempt is rejected and preserved as
engineering evidence.

The reduced long-budget candidate was tested without accessing held-out splits:

| Probe | Batch | Input tokens per row | Generated reasoning tokens | Peak allocated | Peak reserved | Result |
|---|---:|---:|---:|---:|---:|---|
| Saved C0 fixture, 20 validation items | 2 | up to 147 | 5 rows reached 512 | 4.04 GB | 5.06 GB | Pass |
| Two longest validation prompts | 2 | 353, 353 | 512, 512 | 4.16 GB | 4.64 GB | Pass |
| Conservative allocator envelope | 2 | 560, 560 | 411, 416 before EOS | 3.95 GB | 4.27 GB | Pass |

The longest-prompt fixture used two depth-12 graph validation items that had already forced at
budget 256. Both forced at budget 512 under the production two-phase generator. The 560-token
fixture extended validation-derived token streams to the calibrated maximum input envelope and is
used only as a memory check, not as an accuracy result.

A clean mixed-schedule bridge attempt then completed all 4,800 C0 rows, but C1 revealed a longer
prompt envelope: its few-shot prompts reach 490 tokens. At C1 budget 128, batch 4 attempted a
759,169,024-byte allocation with only 572,784,640 bytes free. The evaluator recovered, but the run
was intentionally interrupted and rejected. This is direct evidence that budget 128 must also use
batch 2; the existing 560-input-token batch-2 allocator fixture covers the larger C1 prompt.

Use batch 4 for budgets 16–64 and batch 2 for budgets 128–512. The structured evidence is in
`artifacts/m3/calibration/evaluation-memory-envelope.json`; the saved 20-item run is retained under
`artifacts/m3/calibration/evaluation-c0-b512-b2`. Both rejected bridge sweeps remain archived.
