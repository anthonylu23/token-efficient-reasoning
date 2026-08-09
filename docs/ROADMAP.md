# Experimental Roadmap and Milestone Gates

**Status:** M3 in progress — validation bridge and M1 review passed; ready to freeze

**Last updated:** August 6, 2026

This roadmap turns the research design in [`PROJECT_PROPOSAL.md`](../PROJECT_PROPOSAL.md) into an executable sequence of gated milestones. It defines what must be built, what evidence must be retained, and what must be true before the project advances. A milestone can be complete even when its scientific result is negative; completion means the experiment was run correctly and produced enough evidence for a defensible decision.

The milestones are ordered by dependency rather than calendar date. Heavy GPU work should not begin until the data, accounting, and evaluation gates that make it interpretable have passed.

## Roadmap at a glance

| Milestone | Purpose | Entry dependency | Exit decision |
|---|---|---|---|
| M0 — Reproducible baseline | Establish trustworthy inference, token accounting, and result capture | None | Baseline measurements are repeatable and auditable |
| M1 — Controlled data and verifier | Produce equivalent full and compressed traces with executable ground truth | M0 token-accounting interface | Data is valid, uncontaminated, and shorter under the real tokenizer |
| M2 — 0.6B smoke-test training | Validate the full QLoRA and evaluation workflow cheaply | M0 and M1 | The pipeline works end to end and C4 learns a usable structured format |
| M3 — Primary 1.5B experiment | Test the central hypothesis under controlled conditions | M2 | Continue, revise, investigate compute, or stop/reframe |
| M4 — Generalization | Test whether the result survives new distributions and checkpoints | Positive or informative M3 result | Scope and limits of transfer are established |
| M5 — Extensions | Test RLVR and other follow-up mechanisms one at a time | Evidence-based M3/M4 rationale | Each extension earns continuation independently |

## Gate and evidence conventions

Each milestone uses four kinds of requirements:

- **Deliverables** are code, data, configurations, or reports that must exist.
- **Acceptance criteria** are the objective conditions required to close the milestone.
- **Verification** describes how those conditions are tested.
- **Exit artifacts** are the evidence needed to reproduce or audit the decision later.

Use `planned`, `in progress`, `blocked`, or `complete` as milestone status values. Record a gate review when a milestone closes. The review should list the exact artifact revisions, passed and failed criteria, known deviations, and the decision to continue or revise. Failed scientific hypotheses must not be relabeled as failed engineering milestones when the protocol ran correctly.

Across all milestones:

1. Keep the final test split hidden from model and hyperparameter selection.
2. Record exact model and tokenizer revisions rather than mutable names alone.
3. Give compared conditions identical prompts, answer allowances, decoding settings, and item order unless a documented condition explicitly changes one of them.
4. Save item-level results and raw generations; aggregate metrics alone are insufficient.
5. Treat tokenizer counts, not words or characters, as the reasoning budget.
6. Report unsuccessful and interrupted primary runs in the experiment registry.
7. Complete the smallest relevant CPU tests before starting a substantial GPU job.

## M0 — Reproducible baseline

### Objective

Build the inference and evaluation foundation needed to compare reasoning representations fairly. This milestone establishes the original checkpoints' behavior before training and makes token-budget enforcement, answer extraction, and system measurements independently testable.

### Why this comes first

An apparent compression gain can be created by off-by-one token counts, different chat templates, hidden answer reasoning, or asymmetric truncation. Those errors would invalidate every later comparison. M0 therefore treats the evaluator as research infrastructure, not a convenience script.

### Deliverables

- A locked or reproducibly generated Python environment with PyTorch, Transformers, PEFT, quantization, and evaluation dependencies recorded.
- Model-loading support for Qwen3-0.6B and DeepSeek-R1-Distill-Qwen-1.5B, including exact model and tokenizer revisions.
- A serializable run configuration containing model revision, tokenizer revision, chat template, decoding parameters, seeds, budget, answer allowance, device, precision, and dependency versions.
- An output parser for separate `<think>` and `<answer>` regions, with explicit handling of malformed or incomplete generations.
- Reasoning-token accounting that excludes prompt tokens, delimiters, and the fixed answer allowance.
- Fixed-budget generation for budgets `16, 32, 64, 128, 256, 512`, including a logged forced transition to the answer region when the reasoning budget is exhausted.
- Item-level result storage containing the prompt or stable item ID, raw generation, parsed answer, token counts, parse status, forced-transition status, latency, and correctness.
- Baseline evaluation conditions C0 and C1 on a small development set for both the smoke-test and primary checkpoints.
- Unit and integration tests for parsing, token boundaries, truncation, answer extraction, configuration capture, and result serialization.

### Acceptance criteria

- A deterministic decoding run repeated with the same environment, configuration, and seed produces identical item-level outputs.
- If sampled decoding is supported, repeated runs with the same seed reproduce outputs; different-seed variation is retained rather than averaged away silently.
- Hand-constructed boundary cases confirm that exactly `B` reasoning tokens are allowed at every configured budget, with no answer tokens charged to the reasoning region.
- Forced-answer transitions are applied identically to every condition and are visible in item-level records.
- Every evaluated item has a raw generation, parse status, answer-extraction result, reasoning-token count, and correctness value; missing fields fail the run rather than becoming implicit nulls.
- The saved configuration is sufficient to rerun the evaluation without relying on undocumented defaults.
- Peak VRAM, wall-clock latency, and decode throughput are captured for at least one representative run on the target RTX 3060 Ti.
- All M0 tests pass on a clean checkout.

### Verification

1. Run parser unit tests over valid, missing-delimiter, duplicate-delimiter, empty-reasoning, empty-answer, and budget-exhausted examples.
2. Compare evaluator token counts against direct tokenizer calls on a small manually inspected fixture set.
3. Run the same deterministic baseline twice and compare configurations, raw generations, and item-level metrics.
4. Deliberately exhaust each reasoning budget and confirm the forced-transition flag, answer allowance, and raw output.
5. Reload a saved run configuration and reproduce the run through the normal entry point.
6. Run a short hardware profile on `anthonypc` after checking `uptime`, memory, disk, GPU state, and competing processes.

### Exit artifacts

- Environment lock and hardware snapshot.
- Frozen baseline run configurations.
- Raw C0/C1 development generations and item-level metrics.
- Test report and deterministic-repeat comparison.
- M0 gate review documenting any known parser or model-template limitations.

### Gate decision

Advance only when token accounting and forced-answer behavior are trusted. Model accuracy is not an M0 gate: the purpose is to measure the baseline correctly, including weak performance.

## M1 — Controlled dataset and executable verifier

### Objective

Create an uncontaminated synthetic benchmark whose reasoning depth and state transitions are known, then render each canonical solution in R0–R3 without changing its semantics.

### Why this milestone matters

The central claim concerns verified conceptual transitions, not short-looking prose. A formal generator and replay verifier are therefore required to distinguish genuine reasoning density from omitted steps, answer shortcuts, or tokenizer artifacts.

### Deliverables

- At least two initial task families selected from multi-operation arithmetic, algebraic substitution, state tracking, graph reasoning, Horn-clause reasoning, and conditional program execution.
- Seeded generators for depths `2, 4, 8, 12, 16`, with machine-readable initial states, canonical solution graphs, valid solution paths, and exact answers.
- R0 full-language, R1 concise, R2 telegraphic, and R3 structured renderers derived from the same canonical path.
- An executable transition verifier and final-answer solver for every included task family.
- Train, validation, ordinary test, and compositional-generalization split manifests.
- Split-audit tooling for seed, template, instance, surface-form, graph, and value overlap.
- Per-rendering tokenizer measurements using the exact tokenizer revision selected in M0.
- Data validation that rejects semantic mismatch, invalid transitions, lost negation or identity, incorrect answers, and non-compressed R2/R3 examples.
- Dataset documentation covering schema, generation, verification, splits, known biases, and regeneration.

### Acceptance criteria

- At least two task families and every target depth are represented in each appropriate split.
- At least 99.9% of accepted traces pass replay verification in a full validation sweep. Any accepted trace that fails is investigated; silent dropping during evaluation is not allowed.
- Every R0–R3 rendering for an accepted item resolves to the same ordered transition sequence and final answer.
- Train, validation, and test manifests have no generator-seed or exact-instance overlap. The compositional split also withholds declared template combinations, and the greater-depth split contains depths absent from training.
- R2 and R3 are shorter than R0 under the actual tokenizer for every accepted training example, as required by the proposal.
- The corpus report states median, p10, and p90 tokenizer reduction by task family, depth, and representation. R3 should show at least 30% median reduction from R0 overall; if it does not, revise the renderer before training rather than weakening the threshold after seeing model results.
- Counterfactual checks demonstrate that changing operands, edges, negation, or branch conditions changes the verified path or answer as expected.
- Property tests and a manually reviewed stratified sample cover every task family, depth, and representation.
- The same manifest and generator revision regenerate identical canonical examples and splits.

### Verification

1. Regenerate a fixture dataset twice from the same seeds and compare stable content hashes.
2. Replay every accepted canonical and rendered trace through the verifier.
3. Mutate representative operators, values, variables, graph edges, negations, and branches; confirm invalid traces are rejected.
4. Run the split-overlap audit and retain both its machine-readable output and a summary table.
5. Tokenize all renderings with the pinned tokenizer and produce compression distributions, not only corpus averages.
6. Manually inspect a stratified sample with extra attention to the longest depths and most compressed traces.
7. Run all generator, renderer, solver, verifier, and split property tests.

### Exit artifacts

- Versioned generator and verifier source.
- Immutable split manifests with seeds and content hashes.
- Dataset statistics and tokenizer-compression report.
- Full validation and overlap-audit results.
- Reviewed-sample record and M1 gate review.

### Gate decision

Advance when semantic equivalence, split integrity, and tokenizer-level compression are established. If compressed traces are invalid, repair the representation. If they are valid but not shorter under the real tokenizer, redesign R2/R3 before any training claim about token efficiency.

## M2 — 0.6B smoke-test training

**Status:** Complete execution; gate decision `revise or hold`

### Objective

Exercise the complete training-to-evaluation workflow on Qwen3-0.6B before spending substantial time on the primary checkpoint. The focus is engineering validity and format learnability, not final scientific conclusions.

### Deliverables

- A short memory and throughput calibration run that determines safe sequence length, microbatch, gradient accumulation, precision, and target modules on the 8 GB GPU.
- Reproducible 4-bit QLoRA configurations for C2 full-CoT, C3 telegraphic, and C4 structured-sketch training.
- Training datasets that use the same questions and example counts across C2–C4, with target-token exposure measured and reported.
- Checkpointing, resume, adapter reload, and inference support.
- Training telemetry including loss, learning rate, examples per second, peak VRAM, wall time, and checkpoint size.
- Fixed-budget C0–C4 evaluation across all configured budgets and controlled depths on validation data.
- Frontier plots plus diagnostics for parsing, budget adherence, forced transitions, invalid transitions, repetition, and trace collapse.
- A failure-analysis sample comparing conditions on the same items.

### Acceptance criteria

- C2, C3, and C4 each complete a short training run without non-finite loss or unhandled out-of-memory failure.
- Saved adapters reload in a fresh process and reproduce the original checkpoint-evaluation outputs within deterministic expectations.
- Training configurations differ only in declared condition-specific inputs or targets; an automated comparison reports all differences.
- At least 99% of evaluated C4 generations are parseable into the required reasoning and answer regions. Any lower rate requires a formatting intervention before M3.
- Budget enforcement has zero observed over-budget reasoning regions after evaluator truncation, and every forced transition is logged.
- C4 uses the structured representation rather than collapsing to empty reasoning, copying the prompt, or emitting mostly malformed syntax, as measured by verifier outcomes and manual review.
- On depths 2 and 4, C4 does not lose more than 1 absolute percentage point of exact-answer accuracy relative to the strongest comparable trained control, or the paired confidence interval rules out a material degradation under a predeclared alternative margin.
- The pipeline emits all M3-required raw and aggregate artifacts without manual editing.
- Unit, integration, and a short end-to-end GPU smoke test pass.

### Verification

1. Run a minimal overfit test on a tiny dataset to confirm labels, masking, gradients, saving, and adapter application.
2. Compare resolved C2–C4 configurations and dataset item IDs before training.
3. Reload each adapter in a new process and rerun a fixed evaluation fixture.
4. Inspect loss and gradient behavior for divergence, sudden collapse, and ineffective adapter application.
5. Replay generated structured traces through the M1 verifier and inspect a stratified failure sample.
6. Generate the entire budget-by-depth frontier and confirm each aggregate point is traceable to item-level rows.
7. Reproduce at least one plot and summary table from saved artifacts alone.

### Exit artifacts

- Hardware-calibrated QLoRA configuration.
- C2–C4 smoke-test adapters and run records.
- Raw C0–C4 validation generations and item-level results.
- Frontier, format-adherence, and failure-analysis reports.
- M2 gate review with the exact configuration selected for M3.

### Gate decision

Advance when the training and evaluation system is reliable and structured output is learnable. If the 0.6B model cannot solve even shallow tasks, distinguish a model-capacity limit from a pipeline defect before deciding whether a tightly scoped 1.5B pilot is justified.

### Optional M2-R — RLVR feasibility spike

M2-R is a non-blocking engineering slice that begins only after C4 satisfies the M2 format, replay, and shallow-accuracy gates. Its purpose is to determine whether verifier-backed online policy optimization is operationally feasible on Qwen3-0.6B and the 8 GB target GPU. It is not a scientific comparison and does not delay M3.

#### Deliverables

- A quantized PEFT policy initialized from the C4 adapter.
- Deterministic reward fixtures covering malformed output, verified prefixes, correct answers with invalid traces, and fully replayable correct traces.
- One short RLVR calibration on a single task family using training or development prompts only.
- Recorded policy-optimization method and version, group size, completion limit, sampling parameters, effective batch, rollout-token count, optimizer updates, peak VRAM, verifier time, and wall time.
- Fresh-process adapter reload and evaluation of a fixed development fixture.

#### Acceptance criteria

- Every reward fixture produces the preregistered component values and total reward.
- At least one complete generate, verify, update, save, and reload cycle runs without non-finite loss or an unhandled out-of-memory failure.
- The saved adapter has changed parameters or fixed-fixture token log-probabilities and reloads reproducibly.
- Reward components, zero-variance-group rate, rollout lengths, policy entropy, throughput, and peak VRAM are retained as structured artifacts.
- No validation or test result is used to select the feasibility curriculum or reward coefficients.

#### Gate decision

If M2-R succeeds, record a candidate local configuration for the M5 RLVR extension. If it fails after a documented safe calibration sweep, record the hardware or software boundary and continue to M3 unchanged. Do not interpret engineering infeasibility as evidence against H7 or H8.

## M3 — Primary 1.5B experiment

**Status:** Validation bridge and M1 human review passed; ready for protocol freeze

### Objective

Run the preregistered primary comparison on DeepSeek-R1-Distill-Qwen-1.5B and determine whether structured-sketch post-training improves the accuracy-versus-reasoning-budget frontier.

### Freeze point before training

Before any final test evaluation, freeze and version:

- Primary hypotheses, success thresholds, and the non-inferiority margin.
- Train, validation, test, and compositional manifests.
- Conditions C0–C4 and their training controls.
- Budgets, answer allowances, prompts, chat templates, stop behavior, decoding parameters, and seeds.
- Primary and diagnostic metrics, answer normalization, bootstrap method, and planned figures.
- The rule for checkpoint selection using validation data only.

### Deliverables

- A primary-checkpoint memory and throughput calibration on the target GPU.
- C2, C3, and C4 adapters trained with matched examples, optimizer settings, update counts, LoRA settings, and random-seed sets.
- C0–C4 evaluations at reasoning budgets `16, 32, 64, 128, 256, 512` and depths `2, 4, 8, 12, 16`.
- At least three training seeds for C2–C4 in the final primary comparison. If this proves infeasible, the reduced design must be documented before test-set inspection and the result labeled exploratory.
- Paired item-level bootstrap confidence intervals for accuracy and matched-budget differences.
- Accuracy-versus-budget and accuracy-versus-depth frontiers, verified-transition density, invalid-transition rate, forced-transition rate, latency, throughput, and peak-memory results.
- A complete experiment registry containing completed, failed, interrupted, and excluded runs with reasons.
- A primary report that separates engineering validity, statistical evidence, and scientific interpretation.

### Milestone acceptance criteria

M3 is complete when:

- The frozen protocol and held-out test set were not altered in response to final results.
- All mandatory C0–C4 cells were evaluated, or any missing cell has a documented failure that prevents a primary claim.
- C2–C4 controls use the same base checkpoint, questions, example counts, optimizer family, update count, LoRA configuration, and seed set; unavoidable target-token exposure differences are quantified.
- Every headline aggregate is reproducible from immutable item-level records and raw generations.
- Confidence intervals, task-family breakdowns, and seed variability accompany point estimates.
- Answer allowance, forced-transition behavior, and decoding settings are identical across conditions.
- The report includes null and adverse results and makes one of the decision-rule outcomes below.

These criteria measure whether the experiment is complete. They do not require the hypothesis to succeed.

### Scientific outcome thresholds

**Minimum viable result — all required:**

- At least 30% fewer median reasoning tokens at matched accuracy.
- No more than 1 percentage-point absolute accuracy loss against the strongest comparable baseline.
- Higher verified transitions per token.
- No material increase in invalid-transition rate.

**Primary success — at one or more preregistered practical budgets, such as 64 or 128 tokens:**

- At least 2 percentage points higher exact-answer accuracy than C0.
- Higher accuracy than C1 prompt-only compression and C2 full-CoT QLoRA.
- At least 1.5 times C2 full-CoT QLoRA's verified reasoning density. C2 is the primary density
  reference because both trained conditions emit verifier-readable controlled grammars; C0 density
  remains diagnostic.
- Reliable performance, using the preregistered 80% accuracy rule, at least two controlled-depth levels beyond C0.

**H5 structure test:** C4/R3 versus C3/R2 exact-answer accuracy at budgets 64 and 128 is included
with the six central C4-versus-C0/C1/C2 tests in one eight-test Holm family at family-wise alpha
0.05. C3 comparisons at other budgets/splits and C3 task/depth breakdowns remain descriptive.

**Strong result:**

- A statistically supported upward and leftward frontier shift across multiple task families, with some gain retained on out-of-distribution evaluation in M4.

### Verification

1. Hash and archive the frozen protocol, manifests, configurations, and evaluator revision before the first final test run.
2. Compare all condition configurations and training item IDs automatically.
3. Re-evaluate a deterministic subset after adapter reload to catch checkpoint or environment drift.
4. Recompute all metrics from item-level records with a separate reporting path or independent consistency checks.
5. Bootstrap paired item-level differences and report intervals beside effect sizes.
6. Audit examples on which C4 gains, loses, becomes invalid, or is forcibly transitioned, stratified by task family and depth.
7. Confirm the final report can be regenerated from the experiment registry and saved artifacts without rerunning training.

### Exit artifacts

- Preregistered protocol snapshot and content hashes.
- C2–C4 adapters, configurations, logs, and seed records.
- Complete C0–C4 raw generations and item-level results.
- Reproducible tables, figures, statistical output, and primary report.
- M3 gate review and one explicit decision outcome.

### Decision rules

- **Continue to M4:** C4 improves fixed-budget accuracy or achieves a clear matched-accuracy token reduction without degrading transition validity.
- **Revise the representation:** traces are shorter but transition validity declines.
- **Investigate pause or latent compute:** traces preserve correct information at shallow depth but fail increasingly at high depth, consistent with insufficient sequential computation.
- **Stop or reframe:** C2 explains all gains, C1 performs equally well, or gains disappear on uncontaminated compositional splits.

## M4 — Generalization and robustness

### Objective

Determine whether the best M3 result transfers beyond the training distribution and whether it depends on one model family, task renderer, or synthetic benchmark.

### Entry gate

Enter M4 only when M3 produces either a positive result or a specific failure mode that an out-of-distribution test can distinguish. Generalization evaluation should not be used to search broadly for a benchmark on which an otherwise negative result looks favorable.

### Core deliverables

- Evaluation on the frozen compositional-generalization and greater-depth synthetic splits.
- Evaluation on at least one external math benchmark and one external logic or symbolic-reasoning benchmark selected before observing condition-level results.
- Benchmark-specific answer extraction and contamination caveats.
- Matched C0, C1, C2, and best compressed-condition evaluation with the same budget controls used in M3.
- Task-family and difficulty breakdowns showing where gains transfer, vanish, or reverse.
- Latency, reasoning-token, parsing, forced-transition, and transition-validity measurements where verification is supported.
- A robustness report distinguishing in-distribution compression, compositional transfer, depth extrapolation, and external-benchmark transfer.

### Optional replication deliverable

If compute permits and M3 justifies it, replicate the best condition and its essential controls on Qwen3-1.7B. Treat this as a within-checkpoint comparison; do not use raw cross-family accuracy as evidence for the representation.

### Acceptance criteria

- External and compositional datasets, prompts, budgets, answer normalization, and metrics are frozen before condition-level evaluation.
- Every selected dataset is evaluated for all declared comparison conditions; partial benchmark reporting is labeled exploratory.
- Synthetic compositional and greater-depth results are reported separately from potentially contaminated public benchmarks.
- Transfer claims include paired uncertainty estimates and do not rely only on aggregate accuracy.
- Any verifier that cannot validate free-form external traces is identified, and final-answer accuracy is not mislabeled as verified reasoning density.
- Failure analysis covers at least one transferred task family and one task family where the effect weakens or reverses.
- The report states the supported scope of the claim, including model, task, depth, and budget limitations.

### Verification

1. Hash evaluation manifests and configurations before running each condition.
2. Audit benchmark answer extraction with manually checked fixtures.
3. Recompute results by benchmark, task family, depth, and budget from item-level rows.
4. Inspect paired examples for transfer, formatting adaptation, shortcut behavior, and truncation effects.
5. If replicating on Qwen3-1.7B, rerun tokenizer compression analysis because representation length may differ by tokenizer revision.

### Exit artifacts

- Frozen external and compositional evaluation manifests.
- Raw generations, item-level results, and uncertainty estimates.
- Generalization and robustness report.
- Optional replication adapters and matched-control results.
- M4 gate review defining the defensible contribution boundary.

### Gate decision

A positive M3 result does not need to transfer everywhere. Continue only with claims supported by the observed task families, depths, budgets, and checkpoints. If gains are synthetic-only, report that boundary directly and prioritize dataset or representation analysis over broader claims.

## M5 — Evidence-driven extensions

### Objective

Test mechanisms suggested by M3 and M4 after the discrete structured-reasoning hypothesis has received a clean evaluation. M5 is a portfolio of separately gated follow-up experiments, not a requirement to implement every idea.

### Candidate extensions

| Extension | Evidence that justifies it | Required comparison |
|---|---|---|
| Mixed budget-conditioned training (C5) | One representation works, but fixed compression is inefficient across difficulty levels | Best fixed representation versus C5 at every budget |
| Answer-only training (C6) | Need to test whether visible transitions add value beyond task supervision | C2, C4, and C6 with matched questions and optimization |
| Added reasoning vocabulary (C7) | Valid sketches tokenize poorly despite good semantic behavior | Unchanged-tokenizer C4 versus C7, including latency and compute |
| Utility-weighted pruning (C8) | Failures cluster around low-value repetition or removable trace regions | C4 versus a preregistered pruning method |
| Verifier-backed RLVR (C9–C11) | C4 is replayable and accurate enough to provide a stable cold start | C4 versus correctness-only C9 versus efficiency-aware C10; optional C11 |
| Compact traces plus pause tokens | Compact traces remain valid but accuracy falls at greater depths | Equal visible content with controlled extra sequential compute |
| Continuous latent reasoning | Discrete traces hit a demonstrated representational or compute ceiling | Strong discrete baseline and architecture-aware compute accounting |
| Coding and debugging tasks | Gains generalize across initial symbolic families | Executable MBPP-style or generated program-state verification |

### RLVR extension protocol

RLVR remains downstream of the primary supervised comparison. C9 and C10 may proceed only when C4 produces reliably parseable, replayable traces and either M2-R demonstrates a viable local configuration or an alternative compute environment is documented. C11 is optional.

| Condition | Initialization | Reward | Purpose |
|---|---|---|---|
| C4 | Structured-sketch SFT | None | No-RL baseline |
| C9 | C4 adapter | Exact-answer correctness | Isolate generic correctness RLVR |
| C10 | C4 adapter | Correctness-gated verified efficiency | Test direct optimization of semantic throughput |
| C11, optional | C2 adapter | C10 reward | Test whether the structured cold start is necessary |

C10 and C11 must use a tiered reward with this ordering:

```text
malformed or unverifiable trace              -> 0
verified proper prefix without full success  -> 0.2 * progress_fraction
correct answer with a non-replayable trace   -> 0.25
fully replayable and correct                  -> 1.0 + 0.2 * efficiency_bonus
```

The initial definitions are:

```text
progress_fraction = verified prefix transitions / required transitions
efficiency_bonus = max(0, 1 - reasoning_tokens / assigned_budget)
```

Each rollout must receive an assigned reasoning budget from a preregistered training-budget set and use the evaluator's token-native reasoning-region accounting and forced-answer semantics. The answer allowance and reasoning delimiters do not contribute to `reasoning_tokens`.

Coefficients may be tuned on training and development data, but their ordering must not change and final values must be frozen before the compared runs. Efficiency reward is available only to exact-correct traces that replay completely. No learned reward model is permitted in this extension.

The policy-optimization loss must not introduce an uncontrolled response-length preference. Use and pin a group-relative formulation designed to reduce or eliminate response-length bias, such as DAPO or Dr-GRPO, and treat any residual preference as a potential confound. Compare C9 and C10 using matched policy initialization, prompts, rollout-token budget, optimizer updates, sampling configuration, and hardware wherever practical. Report discrepancies rather than silently equalizing them after results are known.

Training prompt selection may emphasize family-depth strata with mixed rollout outcomes, because groups with identical rewards supply no group-relative signal. Selection must use training and development data only. Log the zero-variance-group rate and retain the unfiltered prompt distribution so the curriculum is auditable.

In addition to the standard M3 metrics and artifacts, retain:

- Each raw rollout and every reward component.
- Rollout and optimizer token counts.
- Verifier calls, failures, and wall time.
- Reward distributions and zero-variance groups by family and depth.
- Policy entropy, completion diversity, and completion lengths.
- Answer-only, skipped-transition, malformed-output, repetition, and parser-exploit diagnostics.
- Training reward alongside the untouched validation accuracy/token frontier.

H7 is supported only if C9 improves over C4. H8 is supported only if C10 improves over both C4 and C9 without a material increase in invalid transitions, malformed traces, forced transitions, or reward-hacking indicators. Shorter output without a frontier improvement is not success.

### Requirements for each extension

- Write a short experiment card before implementation stating the motivating evidence, hypothesis, changed variable, controls, datasets, metrics, success threshold, compute budget, and stop rule.
- Change one primary mechanism at a time whenever practical.
- Reuse frozen M3 evaluation items only for final confirmation, not iterative extension tuning.
- Include the strongest relevant baseline and the simplest alternative explanation.
- Measure accuracy, tokenizer use, latency, memory, and transition validity when the mechanism affects any of them.
- Add the extension to the experiment registry, including failed and abandoned runs.

### Acceptance criteria

An individual extension is complete when its preregistered matrix has run, its results are reproducible from item-level artifacts, and a continue or stop decision is documented. An extension earns further work only when it improves a preregistered outcome over the strongest relevant M3/M4 condition without creating an unreported regression in validity, budget adherence, latency, or memory.

### Verification

Use the M3 protocol as the default verification template: frozen configuration, matched items and seeds, raw generations, paired statistics, condition-difference audit, failure analysis, and reproducible reporting. Add mechanism-specific checks—for example, embedding resize and decode throughput for C7 or explicit extra-compute accounting for pause-token and latent methods.

### Exit artifacts

- One experiment card, run group, report, and decision record per extension.
- Updated experiment registry and artifact index.
- A final synthesis separating confirmed findings, negative findings, and open questions.

## Cross-milestone artifact checklist

The repository should ultimately preserve:

- Locked dependencies and exact model/tokenizer revisions.
- Hardware, driver, CUDA, precision, and peak-memory records.
- Dataset generators, solver/verifier source, seeds, manifests, and hashes.
- Trace renderers and exact tokenizer counts.
- Training and evaluation configurations for every condition and seed.
- LoRA adapters where redistribution permits.
- Raw generations and item-level metrics for every reported run.
- An append-only experiment registry including failed runs.
- Scripts that regenerate every table and figure.
- Gate reviews documenting protocol deviations and decisions.

Large generated datasets, model adapters, and raw outputs may live in external artifact storage, but the repository must retain content-addressed manifests and documented retrieval instructions. Secrets, access tokens, and model-cache paths must never be embedded in saved configurations.

## Project completion definition

The initial study is complete after M3 when the primary comparison has produced a defensible positive, null, or negative answer. M4 strengthens the contribution by establishing transfer boundaries. M5 is optional and should not delay publication of a clean initial result.

The project is ready for an open-source research release when a new contributor can recreate the environment, regenerate or retrieve the data, run a small end-to-end smoke test, reproduce reported tables from saved artifacts, and understand every deviation from the frozen protocol using repository documentation alone.
