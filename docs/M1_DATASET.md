# Milestone 1 controlled dataset

Milestone 1 provides a deterministic synthetic benchmark with executable state transitions at
depths 2, 4, 8, 12, and 16. Every problem, exact answer, and R0–R3 reasoning trace is derived from
one canonical transition path. Rendered traces are parsed back into typed transitions and replayed;
stored metadata alone cannot certify a trace.

## Task families

`multi_operation_arithmetic` applies an ordered sequence of addition, subtraction,
multiplication, and exact integer-division operations. Every intermediate value is an integer and
the prompt includes one explicitly irrelevant fact.

`directed_graph_traversal` follows an ordered label sequence through a branching directed graph.
Outgoing labels are unique at each node, so every requested transition has at most one valid
destination. Unused branches act as distractors.

For both families, depth is the minimum number of primitive transitions accepted by the replay
verifier. The verifier rejects skipped operations or edges even when a result could be obtained by
an algebraic shortcut or other external method.

## Representations

- R0 is controlled full natural language.
- R1 is concise natural language.
- R2 is telegraphic notation.
- R3 is a strict structured sketch.

The representation grammars are family-specific. Canonical data must satisfy
`parse(render(path)) == path` for every representation. The parsers intentionally accept only the
canonical generated grammar; tolerance for imperfect model generations belongs to the M2
evaluation layer.

## Configuration and splits

[`configs/m1.json`](../configs/m1.json) freezes the generator version, master seed, split sizes,
task families, depth allocation, compression threshold, and the two tokenizer checkpoint IDs
selected in M0.

| Split | Depths | Items | Purpose |
|---|---|---:|---|
| `train` | 2, 4, 8, 12 | 8,000 | Training distribution |
| `validation` | 2, 4, 8, 12 | 800 | Model and representation selection |
| `test` | 2, 4, 8, 12 | 2,000 | Ordinary held-out evaluation |
| `test_compositional` | 2, 4, 8, 12 | 800 | Withheld prompt-template combinations |
| `test_depth` | 16 | 1,000 | Depth absent from training |

Seeds are derived from the master seed plus split, family, depth, and attempt index. Generation
rejects repeated exact instances and family-specific value signatures across the entire corpus.
The audit also checks seed, prompt, instance, template, and depth overlap.

## Workflow

Validate the frozen configuration:

```bash
uv run ter-data validate-config configs/m1.json
```

Generate the complete corpus. The destination must not already exist; generation builds in a
temporary sibling directory and publishes it atomically.

```bash
uv run ter-data generate --config configs/m1.json
```

Replay and audit an existing corpus:

```bash
uv run ter-data validate-dataset data/m1
```

Regenerate independently and compare every manifested file:

```bash
uv run ter-data regenerate-check --config configs/m1.json --dataset data/m1
```

The `to_evaluation_item` adapter in `controlled/adapters.py` exposes a controlled example to the
M0 fixed-budget evaluator without discarding the richer source record.

## Artifact schema

Each split JSONL record contains:

- schema, item, family, split, seed, depth, template, and surface-form identifiers;
- natural-language prompt, machine-readable initial state, and canonical solution graph;
- ordered typed transitions, exact answer, and stable semantic instance hash;
- R0–R3 text; and
- per-representation token counts under each pinned tokenizer revision.

Each split manifest records the data-file hash and, for every item, its seed, semantic hash, and
complete record hash. The root manifest covers all generated files.

Generated evidence includes `validation.json`, `split_audit.json` and `.md`, tokenizer compression
reports in JSON and Markdown, all rejected generation attempts, and a stratified review sample.
Rejected attempts are never silently substituted during evaluation: accepted item IDs are frozen
in the split manifests.

## Current generated corpus

The v1 corpus contains 12,600 accepted examples. All accepted canonical and rendered traces pass
replay verification, including independent solver comparison and 25,200 mutation checks. There
are no audited cross-split collisions. R3 achieves 32.0% median token reduction from R0 under both
pinned tokenizers, exceeding the preregistered 30% gate. The 258 rejected attempts remain recorded:
253 repeated value signatures and five exact semantic instances.

The generated review sample covers every split, family, depth, and representation. Its status is
left as `pending_human_review`; the milestone gate must not be reported as final until a human has
reviewed and signed off on that sample.

## Known limitations

- Both families use templated language and linear canonical paths; they do not represent the full
  diversity of arithmetic or graph reasoning.
- Arithmetic answers can sometimes be computed with algebraic shortcuts even though the formal
  trace system requires one transition per listed operation.
- Graph problems test labeled traversal rather than unrestricted reachability proofs.
- Token-based acceptance filters the benchmark toward examples whose R2 and R3 traces are shorter
  for both target tokenizers. Rejection counts are reported to make this selection visible.
- Negation, branching program conditions, algebra, and Horn-clause reasoning are not included in
  this initial two-family corpus.
