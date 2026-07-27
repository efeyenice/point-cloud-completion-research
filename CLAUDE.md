# CLAUDE.md — Point Cloud Completion Research

> **Canonical handoff.** New agent or person: read this top to bottom first. `AGENTS.md` points here.
> Last updated: 2026-07-27.

## What this is
A summer-2026 research project on **3D point cloud completion** — predict the complete shape from a
partial scan. A **student + advisor**; the AI is the **co-researcher who manages the day-to-day** (designs
experiments, builds, analyzes, keeps the workflow clean) while the researcher nudges direction. Guiding
question: **"Does adding per-point RGB/color improve completion, and under what conditions?"** — answered
through a controlled, baseline-first, honestly-reported empirical study.

## Where we are (2026-07-27)
- **Baseline done (s5, ADR-0006):** a from-scratch PyTorch PCN with three arms — `A_geo` (xyz→xyz),
  `C_colin` (xyzrgb→xyz), `B_joint` (xyzrgb→xyzrgb). First result: **color INPUT hurt geometry, A→C = −17%**
  at ~15 shapes/category — but that is a **small-data artifact** (color acts as a memorization fingerprint),
  the opposite of the cross-modal literature (EGIInet ≈ −58%); the **color OUTPUT head is free** (B−C ≈ noise).
  **Data scale is the binding constraint.**
- **Current experiment (ADR-0007):** the **scale curve** — train A/B/C across {20, 60, 150} models/category
  against a *fixed* held-out test set, and plot the A→C gap vs N to find *whether/where* the color-harm flips.
  Harness is merged to `main` (GPU-FPS, fixed test split + growable nested train pool, NN/mean/median
  color-copy baselines, early-stopping, curve driver + report).
- **Now (ADR-0008):** moving to a **resilient, HF-native workflow** (below), because Colab disconnects kept
  nuking the scaled data-gen run.

## Where everything lives (ADR-0008 — Drive is retired)
| Artifact | Home |
|---|---|
| Code + notebooks | **git** (this repo; edited via Colab↔GitHub open/save, outputs stripped) |
| Decisions / rationale | **git** — `docs/adr/`, `CONTEXT.md`, `notes/handoff-*` |
| Generated datasets | **HF dataset** `efeyenice/pc-completion-data` — one `{model_id}.npz`, synced by `CommitScheduler` |
| Run checkpoints / results | **HF** — same dataset repo under `runs/` |
| Live metrics / curves | **Weights & Biases** (+ `metrics.csv` → HF as a self-owned backup) |
| Demo of results | **HF Space** `efeyenice/pc-completion-runs` (Gradio; built once a model exists) |
| Source ShapeNetCore | **HF** (`snapshot_download`) |
| Scratch / intermediates | **Colab local disk** (ephemeral by design) |

**Resilience rule for every long Colab job:** per-unit **atomic commit to HF** + **idempotent resume**
(skip units already on HF) + `CommitScheduler` live-sync + anti-idle. "Done" = bytes on HF, never in RAM.

## Repo layout
- `notebooks/` — `dataGeneration` (s1, colored sampling), `partialGeneration` (s4, PCN-style partials + GT),
  `pcnTraining` (s5, the model + arms + scale curve). **Source of truth for all code.**
- `docs/adr/` — architecture decisions 0001–0008. `CONTEXT.md` — domain glossary.
- `notes/` — dated handoff journal + `00-KEY-FINDING-rgb-helps-completion.md`.
- `tools/mesh-sampling/` — vendored EPFL sampling. `papers/`, `READING-LIST.md`, `W1/W2` decks — background.

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
Finish the **HF-native resilience rewrite** of `partialGeneration` (per-model `npz` + `CommitScheduler` +
idempotent driver + anti-idle), regenerate ~200/category into `pc-completion-data`, then run the scale
curve (gates → T9 → T10) with W&B logging. Tracking in `docs/adr/0008` and the memory workflow note.
