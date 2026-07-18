# Learning High-Density Reasoning Representations in Small Language Models

**Project proposal and experimental specification**  
**Status:** Draft v0.1  
**Date:** July 18, 2026

## 1. Executive summary

This project will investigate whether post-training can teach a small language model to reason more efficiently by expressing intermediate reasoning in a compact, information-dense representation.

The central goal is stronger than producing shorter answers. The project asks whether a post-trained model can complete **more useful conceptual reasoning steps within the same generated-token budget**, and therefore outperform the original checkpoint at fixed reasoning budgets.

The primary experiment will compare an unchanged small reasoning model with parameter-efficient variants trained on compact, structure-preserving reasoning traces. Evaluation will hold the model, prompts, decoding configuration, answer allowance, and reasoning-token budget as constant wherever possible. The principal outcome will be an accuracy-versus-budget Pareto frontier, supported by measurements of verified reasoning transitions per token, latency, and memory use.

The work is designed to run locally on an NVIDIA RTX 3060 Ti with 8 GB of VRAM. Initial experiments will use 4-bit QLoRA on models between 0.6B and 1.7B parameters. Supervised fine-tuning will be the first training method; online reinforcement learning and latent-space reasoning are deferred until the discrete-reasoning hypothesis has been tested cleanly.

## 2. Motivation

Chain-of-thought reasoning improves model performance on difficult tasks but spends many autoregressive decoding steps producing natural-language glue: articles, repeated context, hedging, transitions, restatements, and explanatory prose. Those tokens increase latency and consume a fixed generation budget without necessarily advancing the solution.

A human working on scratch paper typically uses denser notation:

```text
Verbose:
The request reaches the handler, but the email field is null. The validator
assumes the field is present, which causes the exception. The input should be
rejected before the handler executes.

Compact:
POST /users -> email=null -> validator crash -> reject pre-handler
```

The compact trace contains the same causal transitions with less linguistic overhead. If a model can learn to use such representations reliably, a fixed budget could accommodate more reasoning transitions, deeper searches, or additional verification.

This is not guaranteed. Generated tokens also provide sequential transformer computation. Removing tokens can remove useful compute, obscure causal relationships, or produce out-of-distribution syntax that tokenizes inefficiently. The project will therefore measure actual tokenizer output and verified reasoning progress rather than assuming that fewer words imply better reasoning.

## 3. Core research question

> Under an identical generated reasoning-token budget, can a small language model post-trained on compact, structure-preserving reasoning traces complete more verified logical transitions and solve problems of greater reasoning depth than the original frozen checkpoint?

The intended contribution is an empirical study of **semantic reasoning throughput** rather than concise response generation alone.

## 4. Hypotheses

### H1 — Representation compression

Post-training on compact reasoning traces will reduce the number of tokenizer tokens required to represent a fixed sequence of logical transitions.

### H2 — Fixed-budget performance

At the same maximum reasoning-token budget, the compressed-reasoning model will achieve higher exact-answer accuracy than the frozen checkpoint and prompt-only compression baseline.

### H3 — Increased reasoning depth

At a fixed token budget, the compressed-reasoning model will reliably solve tasks requiring more sequential state transitions than the baselines.

### H4 — Training effect beyond prompting

Post-training will improve the accuracy/token frontier beyond what the frozen model achieves when merely prompted or shown few-shot examples of compressed reasoning.

### H5 — Structure matters

Structured sketches that preserve operators, variables, references, negation, and causal relationships will outperform unstructured telegraphic English at comparable token counts.

### H6 — Adaptive compression

A model conditioned on explicit budget tags will learn to use short traces on simple problems and reserve additional reasoning for harder problems, producing a better aggregate frontier than a single fixed compression style.

## 5. Definitions

### Reasoning token

A token generated inside the model's designated reasoning region, excluding the user prompt, chat-template tokens, reasoning delimiters, and final-answer allowance.

### Conceptual reasoning step

A verifiable transition from one problem state to another. Examples include applying an arithmetic operator, substituting a value, following a graph edge, invoking a logical rule, branching on a condition, or checking a candidate solution.

Sentence count, line count, and punctuation will not be treated as conceptual-step counts.

### Verified reasoning density

For tasks with executable or formally checkable traces:

```text
verified reasoning density = valid state transitions / reasoning tokens
```

This is a diagnostic metric, not a replacement for final-answer accuracy.

### Fixed-budget performance

Exact-answer accuracy when every condition receives the same maximum number of generated reasoning tokens and the same separate final-answer allowance.

### Semantic reasoning throughput

The rate at which a generated trace advances through valid problem states, measured through verified transitions per token and the maximum problem depth solved reliably at a fixed budget.

## 6. Scope

### In scope

- Discrete, inspectable compressed reasoning traces.
- Existing-tokenizer compression as the main experiment.
- Arithmetic, algebraic, graph, and symbolic logic tasks with automatic verification.
- A small open-weight model trainable through QLoRA on an RTX 3060 Ti.
- Comparisons against the exact original checkpoint.
- Fixed-budget accuracy, reasoning-depth, latency, and memory measurements.
- Budget-conditioned reasoning formats.

### Out of scope for the initial study

- Training a language model from scratch.
- Claims about unobservable human-like internal reasoning.
- General-purpose chat quality optimization.
- Large-scale online reinforcement learning.
- Hidden-state or continuous latent reasoning in the primary experiment.
- Treating shorter visible text as sufficient evidence of better reasoning.
- Comparisons between unrelated model families as the main result.

## 7. Related work

The project is most closely related to the following lines of research:

- [Chain of Draft](https://arxiv.org/abs/2502.18600) elicits minimal but informative intermediate notes and reports substantial token reductions while preserving performance in several settings.
- [Sketch-of-Thought](https://aclanthology.org/2025.emnlp-main.1236/) uses cognitively inspired sketches, symbolic chunks, and domain-specific representations to reduce reasoning verbosity.
- [TokenSkip](https://arxiv.org/abs/2502.12067) identifies and removes lower-importance reasoning tokens, then fine-tunes models on compressed traces.
- [How Well Do LLMs Compress Their Own Chain-of-Thought?](https://arxiv.org/abs/2503.01141) studies the accuracy/compression tradeoff and motivates adaptive, problem-dependent budgets.
- [Think Better, Not Longer / MUTO](https://aclanthology.org/2026.acl-long.1386/) estimates token-level marginal utility and directly evaluates a 1.5B reasoning-model backbone.
- [Coconut](https://arxiv.org/abs/2412.06769), [Compressed Chain-of-Thought](https://arxiv.org/abs/2412.13171), and [SoftCoT](https://aclanthology.org/2025.acl-long.1137/) explore continuous or soft latent reasoning representations. These motivate a potential second phase but require a different inference architecture.
- [Think Before You Speak](https://openreview.net/forum?id=ph04CRkPdC) studies learned pause tokens, illustrating that the number of visible language tokens and the amount of sequential model computation are related but separable design choices.

The proposed study differs in its primary emphasis: it will directly test whether compact post-training increases **verified conceptual transitions and task accuracy at the same token budget**, rather than focusing only on preserving accuracy after shortening outputs.

## 8. Model selection

### Smoke-test checkpoint

**Qwen3-0.6B**

Purpose:

- Validate data formatting, token accounting, training, evaluation, and checkpoint handling quickly.
- Detect obvious failure modes before committing to longer 1.5B–1.7B runs.

### Primary checkpoint

**DeepSeek-R1-Distill-Qwen-1.5B**

Rationale:

- Small enough for 4-bit adapter training on the target GPU.
- Already trained to emit explicit reasoning, providing a meaningful verbose baseline.
- Closely aligned with existing efficient-reasoning research, including MUTO.
- Open weights and a permissive license.

### Alternative or replication checkpoint

**Qwen3-1.7B**

Its native thinking and non-thinking modes make it useful for testing whether the findings generalize beyond one distilled reasoning checkpoint.

The primary scientific comparisons will always be within a checkpoint family. Cross-model comparisons will be secondary.

## 9. Reasoning representations

Each canonical solution will be rendered in multiple forms while preserving the same underlying state-transition graph.

### R0 — Full natural-language reasoning

```text
Alice begins with 12 apples. After giving away 4, she has 8. Dividing the
remaining 8 apples equally between 2 people gives 4 apples per person.
```

### R1 — Concise natural language

```text
Start 12; give 4 -> 8 left; split 8 between 2 -> 4 each.
```

### R2 — Telegraphic reasoning

```text
12 apples; -4 => 8; /2 => 4 each
```

### R3 — Structured reasoning sketch

```text
qty(12) -> sub(4)=8 -> div(2)=4
```

### R4 — Added reasoning vocabulary, optional ablation

```text
qty 12 <SUB> 4 = 8 <DIV> 2 = 4
```

R4 changes the tokenizer vocabulary and will not be mixed into the primary unchanged-tokenizer claim. It will be reported as a separate representation-learning experiment with latency and compute measurements, not only token counts.

## 10. Dataset strategy

### 10.1 Synthetic controlled-depth dataset

The central dataset will be generated programmatically and paired with an exact solver. It will include task families such as:

- Multi-operation arithmetic with irrelevant facts.
- Linear equation manipulation and substitution.
- Function composition and state tracking.
- Directed-graph reachability and path selection.
- Horn-clause or implication-chain reasoning.
- Conditional execution over small program states.

Each example will contain:

- A natural-language problem.
- A machine-readable initial state.
- A canonical solution graph.
- One or more valid solution paths.
- The minimum required transition depth.
- An exact final answer.
- Renderings R0–R3 of the same reasoning path.
- Exact token counts under the target tokenizer.

Controlled depths will initially be 2, 4, 8, 12, and 16 transitions. Training and test sets must differ in surface forms, values, graph instances, and templates. A dedicated compositional-generalization split will contain unseen template combinations and greater depths than the main training distribution.

### 10.2 Standard external benchmarks

The initial external suite should include:

- GSM8K.
- MATH-500.
- SVAMP or ASDiv.
- ProntoQA or ProsQA.

These benchmarks provide comparability with prior work but may have appeared in model pretraining. They will not replace the uncontaminated synthetic evaluation.

### 10.3 Later coding extension

After the core result is established, the representation can be tested on code reasoning with executable verification:

- MBPP or MBPP+.
- Small generated debugging tasks.
- Program-state prediction.
- Fault-localization traces followed by unit-test execution.

## 11. Data-quality requirements

A training example is accepted only when:

1. The canonical solver verifies the final answer.
2. Every structured transition is executable or rule-valid.
3. All renderings encode the same transition sequence.
4. The compressed rendering is shorter under the actual model tokenizer.
5. Compression preserves numbers, operators, negation, variable identity, and branch conditions.
6. The example does not cross train, validation, or test-generation seeds.

The data pipeline will record both character/word counts and tokenizer counts, but only tokenizer counts will determine the reasoning budget.

## 12. Training conditions and controls

The minimum experiment matrix contains five conditions:

| ID | Model condition | Training target | Purpose |
|---|---|---|---|
| C0 | Frozen checkpoint | None | Original reasoning baseline |
| C1 | Frozen checkpoint + concise prompt/few-shot examples | None | Tests prompting alone |
| C2 | Full-CoT QLoRA | R0 | Controls for additional task supervision |
| C3 | Telegraphic QLoRA | R2 | Tests unstructured language compression |
| C4 | Structured-sketch QLoRA | R3 | Primary proposed method |

Optional conditions:

- C5: Mixed R0–R3 budget-conditioned QLoRA.
- C6: Answer-only QLoRA to test whether visible reasoning is necessary.
- C7: Added-vocabulary structured sketches.
- C8: A utility-pruned or MUTO-inspired dataset variant.

All trained controls must use, as closely as practical:

- The same base checkpoint.
- The same training questions.
- The same number of examples.
- The same optimizer and update count.
- The same LoRA rank and target modules.
- The same random-seed set.
- Comparable target-token exposure, reported explicitly when exact matching is impossible.

The full-CoT control is mandatory. Without it, accuracy gains could be attributed to additional supervised problem exposure rather than reasoning compression.

## 13. Budget conditioning

The model will be trained with explicit reasoning-budget tags:

```text
<B32>   target approximately 32 reasoning tokens
<B64>   target approximately 64 reasoning tokens
<B128>  target approximately 128 reasoning tokens
<B256>  target approximately 256 reasoning tokens
<FULL>  unconstrained explanatory reasoning
```

Training examples will be rendered at one or more achievable compression levels. The model should learn that the tag controls representation density, not final-answer verbosity.

Budget adherence will be measured as its own outcome. A model that ignores the tag or frequently fails to reach an answer within the budget has not succeeded even if its unconstrained generations are accurate.

## 14. Training configuration

Initial configuration for the 1.5B–1.7B run:

- Quantization: 4-bit NF4 with double quantization where supported.
- Method: QLoRA.
- LoRA rank: begin with 16; test 32 only if justified.
- LoRA alpha: 32 for rank 16 as an initial setting.
- Target modules: attention projections plus MLP projections, subject to memory profiling.
- Sequence length: begin at 1,024 tokens.
- Microbatch: 1.
- Gradient accumulation: tune to an effective batch that is stable within VRAM.
- Gradient checkpointing: enabled.
- Precision: BF16 if stable on the installed stack; otherwise FP16.
- Optimizer: memory-efficient AdamW variant.
- Epochs: begin with 1–3, selected from validation behavior rather than test results.
- Checkpoint selection: validation accuracy/token frontier, not loss alone.

The first run on each model will be a short memory and throughput calibration. Exact batch, sequence, and accumulation settings will be recorded rather than assumed.

Online RL is excluded from the first milestone because rollout generation and multi-model memory requirements are disproportionate on an 8 GB GPU. If supervised results are promising, offline preference optimization or token-utility reweighting can be evaluated before full RL.

## 15. Inference and budget enforcement

Outputs will use separate reasoning and answer regions:

```text
<think>
reasoning trace
</think>
<answer>
final answer
</answer>
```

The evaluator will count only generated tokens inside `<think>` as reasoning tokens. Delimiters and final-answer tokens will be accounted for separately.

Primary evaluation uses a maximum reasoning budget `B`. If the model finishes reasoning early, it may transition to its answer. If it reaches `B` without closing the reasoning region, the evaluator will force the answer transition and provide the same small answer allowance to every condition. Forced transitions will be logged.

Budgets will initially be:

```text
16, 32, 64, 128, 256, 512 reasoning tokens
```

The answer allowance will be fixed per task family and kept small enough to prevent additional hidden reasoning in the answer field.

Generation parameters, chat templates, stop conditions, and random seeds will be frozen before final evaluation. If the checkpoint's authors recommend sampling rather than greedy decoding, all conditions will use the same recommended scheme and multiple fixed seeds.

## 16. Evaluation metrics

### Primary metrics

1. **Exact-answer accuracy at each fixed budget.**
2. **Accuracy versus minimum required reasoning depth at each budget.**
3. **Maximum reliably solved depth**, defined in advance, such as the greatest depth achieving at least 80% accuracy.
4. **Token reduction at matched baseline accuracy.**
5. **Accuracy improvement at matched token budget.**

### Reasoning-process diagnostics

- Valid state transitions per generated reasoning token.
- Number and fraction of invalid transitions.
- Gold-path coverage.
- Maximum verified progress before the first invalid transition.
- Repetition, reversal, and unproductive-branch counts.
- Budget-tag adherence.
- Forced-answer-transition rate.

### Systems metrics

- Mean, median, and p95 reasoning-token count.
- End-to-end wall-clock latency.
- Decode tokens per second.
- Peak GPU memory.
- Training time and examples per second.
- Adapter size and total checkpoint storage.

No single ratio such as `accuracy / tokens` will serve as the headline metric. The primary result will be the complete accuracy-versus-budget frontier.

## 17. Statistical protocol

- Freeze a held-out test set before tuning.
- Use at least three training seeds for the final primary comparison if runtime permits.
- Use identical evaluation items and decoding seeds across conditions.
- Report paired accuracy differences.
- Compute item-level bootstrap confidence intervals for accuracy and matched-budget differences.
- Predeclare the non-inferiority margin and success thresholds before the final run.
- Report all attempted primary configurations, including unsuccessful ones.

For sampled decoding, report both per-generation accuracy and per-problem aggregate results. Do not select the best sample after observing the answer unless pass@k is explicitly the metric for every condition.

## 18. Success criteria

### Minimum viable result

The structured-sketch model satisfies all of the following on the uncontaminated synthetic test set:

- At least 30% fewer median reasoning tokens at matched accuracy.
- No more than a 1 percentage-point absolute accuracy loss against the strongest comparable baseline.
- Higher verified transitions per token.
- No material increase in invalid-transition rate.

### Primary success result

At one or more practically relevant budgets, such as 64 or 128 tokens:

- At least 2 percentage points higher exact-answer accuracy than the frozen checkpoint.
- Higher accuracy than the prompt-only compression and full-CoT SFT controls.
- At least 1.5× verified reasoning density.
- Reliably solves problems at least two controlled-depth levels beyond the frozen baseline.

### Strong result

The structured or budget-conditioned model produces a statistically supported upward and leftward shift of the accuracy/token frontier across multiple task families and retains part of the gain on standard out-of-distribution benchmarks.

## 19. Failure criteria and interpretation

The central hypothesis is not supported if apparent gains arise only from:

- Shorter traces with fewer verified conceptual transitions.
- Additional task supervision shared by all fine-tuned models.
- A stronger prompt or favorable answer-extraction behavior.
- Different answer allowances or decoding settings.
- Truncating verbose baselines before they can emit an answer without giving them the same forced-answer mechanism.
- Tokenizer changes that pack operations into custom tokens but do not improve latency or compute efficiency.
- Memorized benchmark items without gains on generated held-out tasks.

A null result remains useful if it identifies the depth or compression ratio at which accuracy breaks down.

## 20. Risks and mitigations

### Compact traces lose essential relationships

**Risk:** Negation, scope, variable identity, or causal direction disappears.

**Mitigation:** Generate traces from a formal solution graph; validate every transition; prefer structured sketches over arbitrary word deletion.

### Fewer words do not mean fewer tokens

**Risk:** Abbreviations or symbols fragment under the existing tokenizer.

**Mitigation:** Accept training renderings only after exact tokenizer measurement; optimize renderers against the target tokenizer.

### The model learns answer shortcuts

**Risk:** It predicts answers from superficial patterns without performing the represented transitions.

**Mitigation:** Use compositional splits, unseen templates, greater test depth, distractors, counterfactual values, and trace verification.

### Fine-tuning improves the task but not the representation

**Risk:** Any SFT condition would improve accuracy.

**Mitigation:** Include a full-CoT SFT control with identical questions and comparable optimization.

### Compression removes useful sequential computation

**Risk:** Denser traces contain the information but provide too few transformer steps to process it.

**Mitigation:** Evaluate a range of budgets and compression levels; later compare compact content tokens plus learned pause tokens.

### Small-model capacity ceiling

**Risk:** A 0.6B or 1.5B model cannot reliably learn the compact language or the underlying tasks.

**Mitigation:** Validate on shallow controlled tasks first, scale depth gradually, and replicate promising settings on the 1.7B checkpoint.

### Hardware limits

**Risk:** Long sequences or preference training exceed 8 GB VRAM.

**Mitigation:** Use 4-bit QLoRA, gradient checkpointing, microbatch 1, short sequences, and staged experiments; keep RL out of the initial scope.

## 21. Milestones

### Milestone 0 — Reproducible baseline

- Implement model loading and deterministic configuration capture.
- Implement separate reasoning/answer token accounting.
- Evaluate the frozen smoke-test and primary checkpoints across fixed budgets.
- Save raw generations and item-level metrics.

**Gate:** Repeated runs reproduce metrics within expected sampling variance.

### Milestone 1 — Controlled dataset and verifier

- Implement at least two task families.
- Generate canonical solution graphs at controlled depths.
- Render R0–R3 traces.
- Verify every trace and record tokenizer lengths.

**Gate:** At least 99.9% of accepted traces pass replay verification, and compressed renderings achieve meaningful tokenizer-level reduction.

### Milestone 2 — Smoke-test training

- Train C2–C4 on Qwen3-0.6B.
- Validate controls, budget enforcement, and frontier plotting.
- Diagnose trace collapse and formatting failures.

**Gate:** The structured model learns its format and does not materially degrade shallow-task accuracy.

### Milestone 3 — Primary 1.5B experiment

- Run calibrated QLoRA training on DeepSeek-R1-Distill-Qwen-1.5B.
- Evaluate C0–C4 across all budgets and depths.
- Run final seeds for promising settings.

**Gate:** Determine whether the primary success criteria are satisfied.

### Milestone 4 — Generalization

- Evaluate standard external math and logic benchmarks.
- Add unseen-task-family or deeper compositional splits.
- Replicate the best condition on Qwen3-1.7B if compute permits.

### Milestone 5 — Extensions

Pursue only if earlier evidence is positive:

- Mixed budget-conditioned training.
- Added reasoning vocabulary.
- Utility-weighted trace pruning.
- Compact traces combined with pause tokens.
- Continuous latent reasoning.
- Coding and debugging tasks.

## 22. Reproducibility and artifacts

The repository should preserve:

- Locked environment and dependency versions.
- Model and tokenizer identifiers plus exact revisions.
- LoRA configuration and training arguments.
- Dataset-generator source, seeds, and split manifests.
- Canonical solution graphs and trace renderers.
- Raw generations for every evaluated condition.
- Item-level token counts and verifier results.
- GPU model, driver, CUDA version, peak memory, and wall time.
- Scripts that regenerate every table and figure.
- A machine-readable experiment registry recording failed and completed runs.

Model weights should remain adapters where possible so the original checkpoint and post-trained differences remain explicit.

## 23. Expected figures

The final report should prioritize:

1. Accuracy versus reasoning-token budget.
2. Accuracy versus required problem depth at fixed budgets.
3. Verified transitions per token by condition.
4. Token distribution and forced-transition rate.
5. Accuracy versus wall-clock latency.
6. Qualitative paired traces showing successful compression and characteristic failures.

## 24. Decision rules

After Milestone 3:

- **Continue to generalization** if C4 improves fixed-budget accuracy or produces a clear matched-accuracy token reduction without degrading transition validity.
- **Revise the representation** if traces are shorter but transition validity falls.
- **Investigate pause or latent computation** if compact traces retain information but performance drops primarily at high reasoning depths, suggesting insufficient sequential compute.
- **Stop or reframe** if full-CoT SFT matches all gains, prompt-only compression performs equally well, or improvements disappear on uncontaminated compositional splits.

## 25. Proposed contribution statement

If successful, the project will demonstrate that post-training a small language model on compact, structure-preserving reasoning sketches can increase semantic reasoning throughput: the model completes more verified conceptual transitions and achieves greater task accuracy within the same generated-token budget. The accompanying controlled-depth benchmark and evaluation protocol will separate genuine reasoning-density improvements from verbosity reduction, task memorization, and answer-format artifacts.
