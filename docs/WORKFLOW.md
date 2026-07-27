# How we work — co-researching with Claude

This project is run as a **researcher + AI co-researcher** collaboration. If you're a
teammate adopting this style, this file is the whole method.

## Roles
- **Researcher** — decides *what* we work on, nudges direction, and generates ideas
  (collaboratively with Claude).
- **Claude (co-researcher)** — runs the research process *autonomously*: designs
  experiments, builds and unit-tests the code, runs it, and analyzes results — then
  **documents, saves, and reports at every step**. Honest, baseline-relative reporting;
  negative results are first-class.

## Where everything lives (ADR-0008 — no Google Drive)
| Artifact | Home |
|---|---|
| Code + notebooks | **git** (this repo; Colab↔GitHub open/save, outputs stripped) |
| Decisions / rationale | **git** — `docs/adr/`, `CONTEXT.md`, `notes/handoff-*` |
| Datasets + run artifacts | **Hugging Face** — `efeyenice/pc-completion-data` |
| Live metrics / curves | **Weights & Biases** |
| Demo | **HF Space** `efeyenice/pc-completion-runs` |
| Scratch | **Colab local disk** (ephemeral) |

## The loop (every unit of work)
1. **Decide / nudge** (researcher).
2. **Grill + plan** (Claude) — stress-test the plan before building; record the decision.
3. **Build** — logic in tested Python modules under `tools/`; notebooks stay thin.
4. **Run** — on Colab, **resumably** (see below).
5. **Document** — an ADR (`docs/adr/`) for hard-to-reverse choices; new domain terms in
   `CONTEXT.md`; a session log in `notes/handoff-<date>.md`.
6. **Report** — numbers **and** visuals, framed against the baseline.

## Conventions
- **Decisions → ADRs** (`docs/adr/NNNN-*.md`), written only when a choice is hard to
  reverse, surprising without context, and a real trade-off.
- **Domain language → `CONTEXT.md`** (a glossary, nothing else).
- **Agent handoff → `CLAUDE.md`** (canonical; `AGENTS.md` points to it).
- **Resilience is mandatory** for long Colab jobs: commit **per unit** to HF and
  **resume idempotently** (skip units already on HF). "Done" = bytes on HF, never in RAM.
  See `tools/pc_resilient.py` (data-gen) and `pcnTraining` T7/T9 (training).

## Running headless (google-colab-cli)
Colab disconnects are unavoidable; we beat them two ways — resumable code (above) *and*
the official CLI's keep-alive + headless execution:
```
colab login                              # once — Google OAuth
colab new --gpu T4                       # provision a runtime
colab exec -f notebooks/pcnTraining.ipynb   # run headless; built-in keep-alive
```
Secrets (`HF_TOKEN`, `WANDB_API_KEY`) live in **Colab → Secrets**; the notebooks read
them at runtime. The CLI cannot set secrets — that stays a one-time UI step.

## Adopting this on a new project
1. Clone the repo; read `CLAUDE.md` (state) and this file (method).
2. Point your agent at `CLAUDE.md` and start the loop from step 1.
3. Keep `CLAUDE.md`, ADRs, and `CONTEXT.md` current as you go — they *are* the memory.
