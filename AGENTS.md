# AGENTS.md

## Project

Research on whether post-training can teach small LMs (0.6B–1.7B) to complete more verified reasoning steps within a fixed token budget. See `PROJECT_PROPOSAL.md` for the full experimental spec.

## Tech stack

- Python
- Hugging Face / PEFT (4-bit QLoRA)
- PyTorch + CUDA
- Local training/eval on RTX 3060 Ti (8 GB)

## Working rules

- Build and run tests as you go; add coverage when thin.
- Ask when ambiguity affects behavior, safety, or scope; otherwise proceed with clear assumptions.
- Suggest refactors when useful; wait for approval before large ones.

## Documentation

- Put documentation in `/docs`. Create the directory if it does not exist.
- Update docs as you go: keep them concise, accurate, and current with the tech stack, architecture, usage, and workflow.
- Write for an eventual open-source audience — clear, professional prose; no scratchpad tone.
- Keep the root `README.md` as the entry point; detailed material belongs in `/docs`.

## Git

- Do not commit unless the human explicitly asks.
- Prefer scoped commits that preserve useful history (no bulk dumps by default).
- Never force-push `main` / rewrite history unless explicitly requested.
- Inspect existing git changes before editing.

## Repo structure

```text
.
├── AGENTS.md              # agent instructions (this file)
├── CLAUDE.md              # symlink → AGENTS.md
├── README.md
├── PROJECT_PROPOSAL.md    # research questions, protocol, milestones
├── docs/                  # project documentation
├── src/                   # training, data, eval code (as added)
└── tests/                 # tests (as added)
```

## Remote compute

Primary machine: Fedora desktop with RTX 3060 Ti (8 GB), 16 GB RAM, Ryzen 7 5800X.

```bash
ssh anthonylu@anthonypc
```

Fallback (Tailscale IP): `ssh anthonylu@100.86.45.68`

Before heavy jobs: check `uptime`, `free -h`, `df -h`, `nvidia-smi`, and running processes. Size models/batches for 8 GB VRAM; prefer one substantial GPU job at a time.
