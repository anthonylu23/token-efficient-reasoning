# External benchmark preparation

## SVAMP

The CPU-only converter prepares the complete 1,000-item challenge set from the
[official repository](https://github.com/arkilpatel/SVAMP) at revision
`78e727689e1c1bebfc4be39c446898e8e10b0518`. The source revision, data SHA-256, and
expected item count are pinned in `configs/followup/external_sources.json`.

On 2026-09-29, all 1,000 items were converted and validated in the separate
follow-up worktree. The prepared artifact is:

```text
artifacts/runs/external/svamp-v1/
```

The directory contains evaluator-compatible `items.jsonl`, the original raw JSON,
the upstream license, converter/source provenance, and a file-hash manifest.
These generated artifacts are Git-ignored; retain the directory when archiving
experiment inputs. Conversion does not load a model or use the GPU.

## Conversion contract

- Keep the upstream item IDs and order.
- Construct each prompt from `Body.strip()`, one newline, and `Question.strip()`.
  Equation, answer, and question-type fields are not included in the prompt.
- Preserve the reference answer numerically using decimal parsing; reject invalid
  or non-finite answers, duplicate IDs, and empty prompt fields. Use the evaluator's
  decimal answer type so fractional references are not scored as integers.
- Verify the source hash and expected item count before creating the output directory.
  Keep the full set; do not select items based on model behavior.
- Refuse to overwrite an existing output directory. No upstream equation is executed
  and no reference answer is silently corrected.

## Reproduction

Download `SVAMP.json` and `LICENSE` from the pinned official revision. The repository's
[source README](https://github.com/arkilpatel/SVAMP/blob/78e727689e1c1bebfc4be39c446898e8e10b0518/README.md)
specifies that the full challenge set is used for evaluation. Then, from the follow-up
worktree, run:

```bash
PYTHONPATH=src ../token-efficient-reasoning/.venv/bin/python \
  -m token_efficient_reasoning.external_data \
  --raw-data artifacts/runs/external/raw/SVAMP.json \
  --license artifacts/runs/external/raw/SVAMP-LICENSE \
  --output-dir artifacts/runs/external/svamp-v1
```

Use a fresh output path if the prepared artifact already exists. Unit tests use
artificial examples to cover field leakage, fractional/negative answers, malformed
data, source-hash changes, duplicate IDs, output protection, and evaluator round trips.

## Remaining work

Data preparation is complete; model evaluation has not started. Freeze external
prompt/decoding settings and the C0–C4 adapter mapping separately after M3. Public
benchmark contamination and upstream annotation errors remain possible. Do not use
this challenge set for prompt development or interpret the controlled synthetic
trace parser's failures as errors on arbitrary benchmark reasoning.

## PrOntoQA v1

The CPU-only wrapper uses the [official v1 generator](https://github.com/asaparov/prontoqa/tree/066f73fc70bfaa92ed2476fd3c41b4951ad6e63a)
at revision `066f73fc70bfaa92ed2476fd3c41b4951ad6e63a`. Before execution it compares
every required source file and the license with the pinned Git object. It copies
those verified files into a fresh artifact and runs only `--model-name json`.

The development configuration, `configs/followup/prontoqa_dev.json`, generates
10 items at each of 1, 3, and 5 hops using seed 20260929, fictional ontologies,
postorder premises, Modus Ponens, and zero demonstrations. The seed is reset for
each hop, following upstream behavior; exact duplicate question/query pairs are
rejected across all files. These 30 items are development fixtures, not a final
external evaluation set.

Install `uv` and obtain a checkout of the pinned revision. From the follow-up
worktree:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  ../token-efficient-reasoning/.venv/bin/python \
  -m token_efficient_reasoning.prontoqa \
  --checkout artifacts/runs/external/prontoqa-upstream \
  --output-dir artifacts/runs/external/prontoqa-dev-v1
```

The wrapper runs `uv --isolated --no-project` with pinned NumPy 2.4.6 and
SciPy 1.16.3. It does not install dependencies into M3's environment. CUDA is
hidden, numerical-library threads are limited to one, Python hash randomization
is fixed, and inherited `PYTHONPATH` is removed for the upstream subprocess.
Each hop has a five-minute subprocess timeout; failures retain logs and a failure
record without a success manifest. The first run may need network access for
isolated dependency downloads. No model weights or external model APIs are used.

Conversion includes only `test_example.question` and `test_example.query` in
prompts, joined with a newline. Demonstrations and gold chains of thought are
excluded. References must be exactly `True` or `False` and use the existing
case-insensitive exact scorer. IDs include upstream hop, seed, and example ID.
Malformed records, model-output fields, missing counts, and duplicate prompts
are rejected. Original JSON, source files, license, command lines, environment
settings, dependency versions, source hashes, label distributions, and per-item
hop metadata are retained in the hash-bound artifact.

The generator was run twice on 2026-09-29: both runs produced the same raw data
and evaluator-item hashes. The second artifact is
`artifacts/runs/external/prontoqa-dev-v1-repeat/`. Upstream ontology-rejection
warnings are retained in the logs; all requested item counts were fulfilled.

Gold proofs are preserved but not independently verified by this converter.
Upstream hop counts are not synthetic-verifier transition counts. Before model
evaluation, freeze a separate evaluation seed set, hop distribution, counts,
prompts, budgets, and adapter mapping. Do not reuse development fixtures for
confirmatory transfer claims. See [the follow-up protocol](FOLLOWUP_PROTOCOL.md).
