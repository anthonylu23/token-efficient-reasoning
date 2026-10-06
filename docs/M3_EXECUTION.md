# M3 execution and supplemental analysis

## Completion and report recovery

All nine training/reload runs and all 45 evaluation runs completed on 2026-10-03
at 06:03 CDT, producing all 342,000 planned generations. The original final-report
command failed on a manifest-schema mismatch: the evaluation writer emits `run_id`,
whereas the report reader required the training writer's `artifact_id`. The dependent
supplemental service then stopped as designed. Neither failure required new model runs.

On 2026-10-06, reporting was recovered in the separate follow-up worktree. The reader
now accepts either identity field and checks every declared identity against the run
directory; conflicting identities, changed files, and incomplete file sets still fail.
The end-to-end test now uses the actual evaluation manifest writer.

The Git-tracked results are in `artifacts/results/m3-20261006/`:

- [Primary report](../artifacts/results/m3-20261006/primary/M3_REPORT.md)
- [Supplemental diagnostics](../artifacts/results/m3-20261006/budget/REPORT.md)
- [Archive metadata](../artifacts/results/m3-20261006/publication.json) and
  `reports-and-freeze.tar.gz`, containing all 107 files from the full recovery bundle
  and original freeze, verified byte-for-byte against the local originals.

The archive includes full aggregate and per-problem tables, all report manifests,
the exact reporting implementation, and recovery provenance. Smaller reports and plots
also have directly browseable copies. Raw generation runs, model weights, and development
corpora remain local and are not part of this Git archive; full input revalidation still
requires the original run artifacts. Local paths in provenance identify the original
execution environment, not paths expected to exist on a fresh clone.

The unmodified local recovery outputs remain under
`artifacts/m3/recovery/20261006-manifest-reader-v1/`:

- [Primary report](../artifacts/m3/recovery/20261006-manifest-reader-v1/primary/M3_REPORT.md)
  and 38 hash-bound report files, including the full frontier and eight primary tests.
- [Supplemental budget diagnostics](../artifacts/m3/recovery/20261006-manifest-reader-v1/budget/REPORT.md)
  with 412 descriptive analysis cells.
- `recovery.json`, the exact `report_reader.patch`, and an `implementation/` source
  snapshot documenting the reporting-only repair and input/output manifest hashes.

Recovery used the original frozen protocol, analysis seeds, hypotheses, and statistical
code. All input runs passed manifest, configuration, item-pairing, summary recomputation,
budget-enforcement, registry, provenance, and training/reload checks. Source manifests,
the registry, and the original failed report were preserved; the original checkout still
matches its frozen workspace hash. No model was loaded or GPU experiment rerun.

The frozen decision is `continue_to_generalization`. Its central `primary_success`
flag is true, but `minimum_viable_result` and `h5_supported` are false. These flags
use different frozen criteria; they are not interchangeable success labels. At budget
128, C4 accuracy is 75.33%, versus 63.52% for C2 and 96.03% for C3. H5 does not support
a structured-R3 advantage over telegraphic R2. The follow-up should retain C3 as a
strong control rather than treating C4 as the best representation.

Validation: 182 tests passed, three opt-in model tests skipped; Ruff and whitespace
checks passed. All three recovery/output manifests verified successfully. The original
systemd units retain their historical failed status; this recovery supersedes their
reporting attempts without clearing the failure evidence or restarting GPU execution.

To reproduce in a **fresh** output directory, from the follow-up worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../token-efficient-reasoning/.venv/bin/python -u scripts/recover_m3_reports.py \
  --source-root ../token-efficient-reasoning \
  --output-dir artifacts/m3/recovery/20261006-manifest-reader-v1
```

The wrapper retains the frozen-workspace check, records the exact reporting source,
builds the primary report first, and supplies its explicit path to supplemental
analysis. `budget_analysis --primary-report PATH` also accepts a recovered report
directly, but still requires a matching freeze, run plan, source roots, complete
generation/run counts, and passing training gates.

## Original execution

The frozen M3 matrix was launched on `anthonypc` on 2026-09-27 (America/Chicago).
It contains nine training runs and 45 evaluation runs, totaling 342,000 generations.
The runner performs fresh-process adapter reload checks, then builds the original
saved-artifact report. The frozen primary hypotheses, C4 treatment, and statistical
rules are unchanged.

The execution checkout is:

```text
/home/anthonylu/Documents/Projects/token-efficient-reasoning
```

It remains at commit `cfb0824a9d4d4a181eeb02f0a79d214be03ac21a`. The validated freeze
manifest SHA-256 is
`34d20ba79e56d7ec4805e104e7c74328b3227030696d02b318e5c173d416e7db`.
Keep that checkout unchanged for frozen-input validation and reproducibility.
Follow-up code and documentation live in this separate worktree:

```text
/home/anthonylu/Documents/Projects/token-efficient-reasoning-followup
```

Follow-up code, reporting fixes, and published results are maintained on the separate
`m3-results-and-followups` branch. This worktree uses the existing environment through
an explicit interpreter path and `PYTHONPATH=src`; its analysis does not alter the
frozen evaluator.

## Operations

The transient user service survives terminal disconnection but does not automatically
resume after a reboot. It stops on failures; it does not delete or automatically retry
failed artifacts. `RemainAfterExit=yes` retains its successful completion state.

```bash
systemctl --user status ter-m3-primary-20260927
tail -f /home/anthonylu/Documents/Projects/token-efficient-reasoning/artifacts/m3/operations/primary-20260927.log
systemctl --user status ter-m3-budget-analysis-20260927
```

`UV_NO_SYNC=1` preserves the verified environment and its installed training extras.
The training service has a 5 GiB host-memory limit and 512 MiB swap limit because
other desktop workloads occupy significant RAM. These are process limits, not changes
to model, precision, batches, or optimization. Before launch the GPU had no competing
compute job, approximately 5.4 GiB host RAM was available, and disk had 1.3 TiB free.
The first run reached stable optimizer updates within the limit.

The validation bridge contains approximately 8.24 hours of generation when each
batch's latency is counted once. Scaling by `3 * 3800 / 800` projects 117.3 hours
of primary generation, plus roughly seven hours of training and reporting overhead.
This is an estimate, not an ETA: depth-16 prompts, workload contention, failures,
and resume operations can change elapsed time.

If execution fails, inspect the service, registry, failure records, and latest
checkpoint before restarting. The original runner supports same-run checkpoint and
item-level resume. Do not remove failed run directories or change protocol inputs.

## Budget-feasibility analysis

The supplemental analysis verifies source manifests and dataset hashes, recomputes
trace diagnostics from saved text, and groups C2–C4 by family, depth, budget, and
whether the canonical rendering fits the reasoning allowance. It reports accuracy,
complete replay, forced transitions, parsed errors, and verified-prefix density.

```bash
PYTHONPATH=src ../token-efficient-reasoning/.venv/bin/python \
  -m token_efficient_reasoning.budget_analysis \
  --source-root ../token-efficient-reasoning \
  --phase bridge \
  --output-dir artifacts/m3/supplemental/bridge-budget
```

Use a new output directory when rerunning; reports are not overwritten. The bridge
report already exists at `artifacts/m3/supplemental/bridge-budget/REPORT.md` in this
worktree. Primary mode refuses to inspect held-out examples until the complete
primary report exists and its input summary is bound to the same freeze.
The original `ter-m3-budget-analysis-20260927` service was configured to wait for
successful M3 completion and write `artifacts/m3/supplemental/primary-budget/`.
It stopped when original reporting failed; the completed replacement is the
`budget/` directory in the recovery bundle above.

This analysis is descriptive and does not modify frozen p-values or success gates.
Canonical rendering length is not a lower bound on all valid solutions. Feasible
subsets differ across representations, and forced transitions can coexist with
complete replay when the last content token exactly fills the allowance.

## Validation finding

At budget 128, the bridge results are:

| Condition | Canonical trace fits | Accuracy when it fits | Accuracy when it does not fit |
|---|---:|---:|---:|
| C2 / R0 | 500 / 800 | 100% | 3.0% |
| C3 / R2 | 776 / 800 | 100% | 0.0% |
| C4 / R3 | 600 / 800 | 100% | 0.5% |

This supports representation length and truncation as an explanation for the
validation frontier. It does not establish causal mediation or generalization to
unseen depths. In particular, C3 and C4 both use symbolic, parseable representations;
their comparison cannot establish a general structure-versus-unstructured-text effect.
