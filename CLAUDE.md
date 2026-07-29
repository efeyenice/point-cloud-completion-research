# CLAUDE.md — Point Cloud Completion Research

> **Canonical handoff.** New agent or person: read this top to bottom first. `AGENTS.md` points here.
> Last updated: 2026-07-29.

## What this is
A summer-2026 research project on **3D point cloud completion** — predict the complete shape from a
partial scan. A **student + advisor**; the AI is the **co-researcher who manages the day-to-day** (designs
experiments, builds, analyzes, keeps the workflow clean) while the researcher nudges direction. Guiding
question: **"Does adding per-point RGB/color improve completion, and under what conditions?"** — answered
through a controlled, baseline-first, honestly-reported empirical study.

## Where we are (2026-07-29)
- **The generative phase (s7) is OPEN (ADR-0009):** next spine = **from-scratch conditional point
  diffusion** — completion recast as sampling from an explicit learned shape prior, keeping the
  controlled A/C(/B) arms. Eval = **two-row protocol** (one-sample CD honesty row, s6-comparable +
  best-of-10 min-CD / TMD distribution rows). Phasing: unconditional airplane DDPM (gated) → A-diff →
  C-diff (**the diffusion A→C delta = headline**). Curve widening/extension continues as background
  data-gen (feeds curve *and* diffusion). Advisor-meeting crib sheet (decodes her diffusion / grid /
  priors fragments into 4 questions): `notes/handoff-2026-07-29.md`. Glossary: `CONTEXT.md` §s7.
- **The scale curve RAN and is read (ADR-0007 outcome):** 600 fresh bundles + 15 runs
  (n{020,060,150} × A/B/C, 3 seeds at top rung) all on HF. **The color-input harm shrinks with
  data but does not flip: A→C = +6.6% (N=20) → +4.8% (N=60) → +4.0% (N=150, sign-consistent
  across seeds).** Output head stays free (B ≈ C, B even edges it). Learned color beats NN-copy
  (0.073 vs 0.082) but **loses to mean-copy (0.070)**; oracle floor 0.020. Continuity check:
  sign replicated, magnitude not (+6.6% vs s5's +17%) → **s5's −17% is superseded**;
  cross-pipeline magnitudes are non-comparable (see ADR-0007 Outcome).
- **The full write-up exists:** `reports/2026-07-28-scale-curve/report.pdf` — the first of the
  **weekly LaTeX reports** (TL;DR, design, ops story, results, honest read, next-step menu,
  provenance, process log). Rules: numbers are macros generated from the HF artifacts
  (`make figures && make`); claims conservative; PDF committed.
- **Arms (s5, ADR-0006):** from-scratch PyTorch PCN — `A_geo` (xyz→xyz), `C_colin`
  (xyzrgb→xyz), `B_joint` (xyzrgb→xyzrgb), decoupled color head, oracle-calibrated gates.
- **Ops (ADR-0008):** resilient HF-native workflow proven end-to-end this week — per-model
  atomic data-gen, resumable curve driver (survived VM recycles), W&B live metrics,
  `google-colab-cli` for headless. Method in `docs/WORKFLOW.md`.

## Where everything lives (ADR-0008 — Drive is retired)
| Artifact | Home |
|---|---|
| Code + notebooks | **git** (this repo; edited via Colab↔GitHub open/save, outputs stripped) |
| Decisions / rationale | **git** — `docs/adr/`, `CONTEXT.md`, `notes/handoff-*` |
| Generated datasets | **HF dataset** `efeyenice/pc-completion-data` — one `{model_id}.npz`, synced by `CommitScheduler` |
| Run checkpoints / results | **HF** — same dataset repo under `runs/` |
| Live metrics / curves | **Weights & Biases** (+ `metrics.csv` → HF as a self-owned backup) |
| Weekly reports | **git** — `reports/<date>-<slug>/` (LaTeX + scripted figures + committed PDF) |
| Demo of results | **HF Space** `efeyenice/pc-completion-runs` (Gradio; built once a model exists) |
| Source ShapeNetCore | **HF** (`snapshot_download`) |
| Scratch / intermediates | **Colab local disk** (ephemeral by design) |

**Resilience rule for every long Colab job:** per-unit **atomic commit to HF** + **idempotent resume**
(skip units already on HF) + `CommitScheduler` live-sync + anti-idle. "Done" = bytes on HF, never in RAM.

## Repo layout
- `notebooks/` — `dataGeneration` (s1, colored sampling), `partialGeneration` (s4, PCN-style partials + GT),
  `pcnTraining` (s5, the model + arms + scale curve). **Source of truth for all code.**
- `docs/adr/` — architecture decisions 0001–0009. `docs/WORKFLOW.md` — **how we co-research** (for adopters).
  `docs/READING-LIST.md` — the paper list. `CONTEXT.md` — domain glossary.
- `notes/` — dated handoff journal + `00-KEY-FINDING-rgb-helps-completion.md`.
- `reports/` — **weekly research reports** (shared LaTeX template + one dir per week; numbers/figures
  regenerated from HF by `make_figures.py`; PDF committed). See `reports/README.md`.
- `tools/` — resilient modules `pc_resilient.py` (data-gen) + `pc_hf_data.py` (training loader) with tests,
  and vendored `mesh-sampling/`. `archive/` — reading-phase + source materials (decks, proposal).

## Research lineage (context)
Backbones: PointNet → PointNet++ → DGCNN → Point Transformer. Completion: PCN → FoldingNet →
PoinTr/AdaPoinTr → SnowflakeNet. Cross-modal "does color help" line: ViPC → XMFnet → CSDN → **EGIInet**
(SOTA, open code). **Our angle:** *per-point* RGB (not a separate guidance image) + *outputting* color +
the controlled "when does color help?" study.

## How to work
- **Memory** at `~/.claude/projects/-Users-macbookpro-conductor-repos-point-cloud-completion-research/memory/`
  (index `MEMORY.md`) — recalled each session; keep it current.
- Decisions → ADRs (sparingly: hard-to-reverse + surprising + a real trade-off). Per-paper reads →
  `notes/<paper>.md`. Reporting is honest and baseline-relative; visualize.
- The researcher (Efe) has **delegated day-to-day management** to the AI: take ownership, act with a clear
  recommendation, grill only on genuine forks. He wants to understand, and dislikes clutter/confusion.

## Immediate next step
1. **Efe presents** `reports/2026-07-28-scale-curve/report.pdf` to the advisor and asks the **4
   questions** in `notes/handoff-2026-07-29.md` (#3 — which sense of "priors" — is the must-ask).
   Meeting shape: report → proposal ("open the generative phase") → questions.
2. **Claude builds Phase 1** (DDPM tools module + unit tests + s7 notebook, unconditional airplane
   prior); **Efe runs it** on Colab. Background job in parallel: widen categories + extend N
   (airplane prioritized). Tier-2 deep-reads (DDPM → PVD → PDR) start alongside.
3. If the meeting redirects (grid diffusion, foundation priors), amend ADR-0009's outcome; otherwise
   proceed. New week → new `reports/<date>-<slug>/` from the template (recipe in `reports/README.md`).
