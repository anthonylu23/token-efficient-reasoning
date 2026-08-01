# Milestone 3 validation-bridge engineering record

**Date:** 2026-07-28
**Scope:** DeepSeek-R1-Distill-Qwen-1.5B, validation data only

## Training

All selected runs use the same 8,000 items, 500 optimizer updates, seed `314159`, effective batch
16, optimizer, learning-rate schedule, LoRA settings, and primary-checkpoint revision.

| Condition | Run | Global step | Wall time | Peak allocated | Peak reserved | Adapter |
|---|---|---:|---:|---:|---:|---:|
| C2 / R0 | `c2-resume100` | 500 | 41.2 min resumed segment | 4.40 GB | 4.84 GB | 85.4 MB |
| C3 / R2 | `c3` | 500 | 46.8 min | 4.21 GB | 6.18 GB | 85.4 MB |
| C4 / R3 | `c4` | 500 | 47.4 min | 4.26 GB | 5.16 GB | 85.4 MB |

C2 was interactively interrupted after checkpoint 100 so the long run could be moved into a
detached transient service. The first attempt is retained as `interrupted`; the accepted run
restored checkpoint 100 and completed through step 500. The registry contains both attempts.

Every selected adapter changed trainable parameters and reproduced its saved eight-token fixture
exactly in a fresh process.

## Reload scheduling deviation

Transient `PathExists` units were initially used to schedule C3 and C4 reload verification after
training. Because a path remains present after creation, the units retriggered, and the first C3
and C4 verification processes briefly overlapped. Both produced exact fixtures without OOM, but
their overlap is not used as the accepted isolated evidence.

Both path units were stopped. With the GPU free, C4 and C3 were each reloaded again in isolated
foreground processes and independently reproduced all expected token IDs. Training artifacts were
not affected.

## Evaluation calibration

Single-item C4 evaluation measured 4.02 seconds per generation on a 20-generation fixture. The
same fixture under deterministic batch size 4:

- completed in roughly one quarter to one third of the wall time;
- used 4.06 GB peak allocated and 4.79 GB peak reserved VRAM;
- had zero reasoning-budget violations;
- preserved all 20 correctness outcomes;
- matched 19 of 20 complete token sequences against single-item decoding; and
- reproduced all 20 reasoning sequences, answer sequences, correctness outcomes, item seeds, and
  trace diagnostics exactly on a second batched run.

The initial full C0 sweep then demonstrated that batch 4 is unsafe at long budgets. It completed
all 4,000 rows through budget 256 and retained 28 rows at budget 512, but the allocator attempted
an additional 768 MiB allocation with only about 68 MiB free. The process recovered, and was then
cleanly interrupted rather than treating a near-capacity run as acceptable evidence. Its
`failure.json`, manifest, item rows, registry events, and system journal remain preserved.

Batch 2 subsequently completed a saved 20-item budget-512 validation fixture with five forced
512-token generations at 4.04 GB peak allocated and 5.06 GB peak reserved. A targeted replay of
the two longest observed validation prompts (353 tokens each) forced both rows at 512 reasoning
tokens while peaking at 4.16 GB allocated and 4.64 GB reserved. A validation-derived 560-token
allocator-envelope fixture also completed safely.

The first mixed-schedule rerun completed C0, then C1 exposed a wider prompt envelope from its
few-shot examples: up to 490 prompt tokens. Batch 4 at budget 128 emitted a recoverable allocator
failure while requesting about 724 MiB with about 546 MiB free. The run was cleanly interrupted
after 2,824 C1 rows and is also retained as engineering evidence.

Final bridge and primary comparisons therefore use a frozen per-budget schedule: batch 4 for
budgets 16–64 and batch 2 for budgets 128–512. Both rejected sweeps are engineering evidence only;
the accepted bridge evaluation is rerun from an empty directory under the final schedule.

The earlier C4 transient path unit was also found still retriggering failed file-exists attempts
and appending them to the registry. It did not load a model or alter adapters, but generated noisy
append-only lifecycle events. The path and service units were stopped and collected before the
final bridge rerun; their events remain preserved rather than rewriting the registry.

## Accepted bridge sweep

The accepted bridge sweep ran under the final `4, 4, 4, 2, 2, 2` batch schedule from
2026-07-28 05:01 to 13:16 CDT. It completed C0–C4 sequentially in the same systemd service:

| Condition | Rows | Completed |
|---|---:|---|
| C0 | 4,800 | 06:52 |
| C1 | 4,800 | 08:29 |
| C2 | 4,800 | 10:28 |
| C3 | 4,800 | 11:46 |
| C4 | 4,800 | 13:16 |

Every condition has a finalized summary and manifest, all recorded file hashes verify, and every
budget contains exactly 800 rows at its configured batch size. C1 passed the prior failure point
at row 2,824 under budget 128/batch 2 and completed budgets 256 and 512 without an allocator
warning. The service exited successfully after 8 hours 15 minutes.

The initial saved-artifact report exposed a protocol-comparison type bug: JSON lists from the
resolved evaluation payload compared unequal to tuple-backed `M3Config.to_dict()` fields despite
identical printed values and protocol hashes. The comparator was corrected to normalize both
payloads through JSON, covered by regression tests, and the original false report was archived at
`artifacts/m3/bridge/report_protocol_type_bug_20260728`.

The report regenerated from unchanged saved artifacts passes all bridge gates. See
`artifacts/M3_BRIDGE_GATE_REVIEW.md` for the decision and exact results. Held-out test access
remains locked pending the M1 human review and protocol freeze.
