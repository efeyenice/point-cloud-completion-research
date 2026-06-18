# CLAUDE.md — Point Cloud Completion Research Project

> **Handoff document.** If you are a new agent or person picking this up, read this file top to bottom first.
> It gives you the full scope, the current state, what's been decided, what's open, and what to do next.
> Last updated: 2026-06-18.

---

## 0. What this project is, in three sentences

This is a **summer 2026 academic research project on 3D point cloud completion** — given a partial 3D
scan of an object, predict the complete shape. The researcher is a **student** working under an **advisor**;
the role of the AI here is **research advisor / co-researcher** (explain concepts, read papers together,
design experiments, help build the baseline — not just answer one-off questions). The guiding research
question is **"Does adding RGB / color / texture information improve point cloud completion, and under
what conditions?"** — to be answered through a controlled, baseline-first empirical study.

**Working directory:** `~/Desktop/point-cloud` (not a git repo).
**Researcher:** efeyenice004@gmail.com (likely Efe Yenice).

---

## 1. Research area: point cloud completion

A **point cloud** is an unordered set of 3D points (x,y,z), optionally with extra per-point channels
(color, intensity, normals), produced by LiDAR or photogrammetry. **Completion** = recovering the full
object/scene from an incomplete cloud (incomplete due to self-occlusion, limited sensor viewpoints, etc.).

The deep-learning lineage that matters here:
- **Backbones:** PointNet (permutation invariance via max-pool) → PointNet++ (hierarchical local features,
  FPS + kNN grouping) → DGCNN (EdgeConv on dynamic graphs) → Point Transformer (attention over neighborhoods).
- **Completion methods:** PCN (encoder → coarse → folding decoder) → FoldingNet (2D-grid folding decoder) →
  **PoinTr / AdaPoinTr** (completion as set-to-set translation with a geometry-aware transformer over
  "point proxies") → SnowflakeNet (point-splitting coarse-to-fine).
- **PoinTr/AdaPoinTr is the architecture this project builds its baseline on** (advisor: "try with pointers"
  = PoinTr family). Official code: https://github.com/yuxumin/PoinTr (serves both papers).

---

## 2. The research question and how it evolved — READ THIS

There are **three layers** to the framing. Do not confuse them; the project operates at layer 3.

**Layer 1 — the original written proposal** (`W1 - Project Proposal.docx`, ambitious version):
Build a *unified, category-agnostic framework* that jointly recovers geometry **and** texture (per-point RGB)
as one multimodal signal, where geometry and color mutually reinforce — contrasted against sequential
pipelines (ComPC, PointDreamer) and geometry-only methods (ComPose). Grand, end-goal vision.

**Layer 2 — the W2 deck experiment proposals** (generative extensions of AdaPoinTr):
generative query generator, proxy-space diffusion, diffusion refinement of PoinTr's coarse output. These
are *candidate later mechanisms*, currently **deprioritized**.

**Layer 3 — the advisor's actual directive (THIS is the current operating plan).**
From the advisor meeting, the project deliberately **starts slow and controlled**, treating the RGB question
as an open *exploration*, not a committed thesis:
- _"Can RGB or textures improve our predictions? The answer may not be definitive — our goal is exploration,
  to approach it in a controlled manner. Under what conditions is it useful, under what conditions is it not?"_
- **First, establish a geometry-only (x,y,z) baseline.** Everything is evaluated and written relative to whether
  each iteration improves on that baseline.
- Then run **controlled iterations / ablations**: what happens if we add noise? add shape + color? change the
  architecture? Experiment everything, take notes, **visualize**, report honestly.
- **Overfit sanity check** after the baseline (can the model really learn on a tiny sample?).
- Explicit steps the advisor gave: **(1) read the slides and papers, (2) extract the baseline.**

> **Bottom line for any agent:** do not jump to the grand multimodal/generative architecture. The job right now
> is (a) finish building the knowledge base, then (b) stand up a geometry-only PoinTr baseline that reproduces a
> known number, then (c) run controlled experiments adding RGB and measure the delta. The "unified xyz+rgb" and
> "generative" ideas are future iterations, kept on the shelf until the baseline exists.

---

## 3. KEY FINDING — the hypothesis already has strong prior art

A literature search (2026-06-12; full write-up in `notes/00-KEY-FINDING-rgb-helps-completion.md`) established that
**"does extra RGB improve completion?" is already answered YES** by the **view-guided / cross-modal completion**
subfield. This reframes the project's contribution.

**The evidence — EGIInet (ECCV 2024) Table 1, ShapeNet-ViPC, Mean Chamfer Distance ×10³ (lower is better):**

| Method | Input | Mean CD ↓ |
|---|---|---|
| FoldingNet | geometry only | 6.271 |
| PCN | geometry only | 5.619 |
| GRNet | geometry only | 3.171 |
| **PoinTr** | **geometry only** | **2.851** ← our baseline target |
| Seedformer | geometry only | 2.902 |
| ViPC | + RGB image | 3.308 |
| CSDN | + RGB image | 2.570 |
| XMFnet | + RGB image | 1.443 |
| **EGIInet** | **+ RGB image** | **1.211** ← current SOTA |

- Adding one RGB image takes CD from PoinTr's **2.851** to EGIInet's **1.211** — a **~58% error reduction**.
- **Crucial nuance (this IS the advisor's "under what conditions" question):** early ViPC (3.308) is *worse*
  than geometry-only PoinTr (2.851). Color only helped once fusion got good. ⇒ **color helps iff fused well;
  naive fusion can hurt.** A systematic study of that conditionality is a genuine contribution.

**The cross-modal completion lineage (Tier 1A in the reading list):**
- **ViPC** (CVPR 2021) — https://arxiv.org/abs/2104.05666 — origin; built the **ShapeNet-ViPC** dataset.
- **XMFnet** (ECCV 2022) — https://arxiv.org/abs/2209.09552 — cross-attention fusion.
- **CSDN** (TVCG 2023) — cross-modal dual refinement.
- **EGIInet** (ECCV 2024) — https://arxiv.org/abs/2407.02887 — SOTA, **open code + pretrained + dataset**:
  https://github.com/WHU-USI3DV/EGIInet

**Where our differentiation / open gap is** (the ViPC line uses a *separate 2D image* and outputs *geometry only*):
1. **Per-point RGB** (x,y,z,r,g,b as one signal attached to the points) instead of a separate guidance image.
2. **Outputting color too** (joint geometry+texture completion), not just consuming color to fix geometry.
3. The **controlled "when does color help?" study** itself.
Per-point colored-completion literature is thin (scene-level Joint-Color-Semantic
[arXiv:2210.05891](https://arxiv.org/abs/2210.05891); GPN [arXiv:2404.08312](https://arxiv.org/abs/2404.08312)),
which is what makes the gap real.

---

## 4. Concrete project specifics (dataset, baseline, environment)

- **Primary dataset: ShapeNet-ViPC** — 38,328 objects, 13 categories, built on ShapeNet/ShapeNetRendering;
  **24 rendered image views per object**; pairs partial point cloud + complete point cloud + multi-view RGB
  images (~143 GB). Download + loaders via the EGIInet repo. Experiments commonly use 31,650 objects (8 cats),
  80/20 train/test.
  - ⚠️ It pairs point clouds with **images**, not per-point-colored point clouds. So **image-guided**
    experiments are plug-and-play, but **per-point-RGB** experiments still need color *sampled onto points*
    from textured ShapeNet meshes (an unresolved data task — see open questions).
  - For pure geometry baselines you may also use the standard **PCN benchmark** and **ShapeNet-55/34**
    (both geometry-only).
- **Baseline model:** PoinTr (or AdaPoinTr) **geometry-only**, from https://github.com/yuxumin/PoinTr.
  **Target: reproduce ≈ 2.851 CD** on ShapeNet-ViPC (PoinTr is a published row in that benchmark).
- **SOTA cross-modal reference:** EGIInet (CD 1.211), code available for comparison.
- **Compute:** GPU via AWS or Google Cloud (per proposal). Expect the usual PoinTr setup pain: custom CUDA
  extensions (Chamfer distance, `pointnet2_ops`), specific PyTorch/CUDA versions. EGIInet needs Python 3.6+,
  PyTorch 1.8+, CUDA 11.8, plus open3d/torch-scatter/timm.
- **Coordination:** there is a Slack group for the project (per proposal notes).

---

## 5. The milestone ladder (advisor's plan, operationalized)

The week catch-up plan is in `WEEK-PLAN.md`. The baseline-and-beyond ladder (the real research arc) is:

1. **Knowledge base** — read the slides + Tier 0/1 papers (in progress).
2. **Baseline** — PoinTr geometry-only on ShapeNet-ViPC; reproduce ≈2.851 CD. Get **inference + visualization**
   working on a few samples first, then full eval.
3. **Overfit sanity check** — can the model memorize 1–10 samples? (Confirms the training loop actually learns.)
4. **Iteration 1 — image-guided RGB:** add image guidance, measure delta vs baseline (reference: EGIInet).
5. **Iteration 2 — per-point RGB:** attach sampled color to input points; does in-representation color beat a
   separate image?
6. **Iteration 3 — output color:** predict rgb alongside xyz (joint completion) — the proposal's real novelty.
7. **The study** — ablate fusion type / noise / category to map *when* color helps. Visualize everything.

> Each step is judged against the geometry-only baseline. Honest reporting of negative results is expected and
> valued — "color didn't help under condition X" is a legitimate finding here.

---

## 6. Repository map

```
~/Desktop/point-cloud/
├── CLAUDE.md                  ← you are here (handoff/context)
├── W1 - Point Cloud Project.pptx          ← W1 deck: broad field survey (backbones + 4 task families)
├── W2 - PoinTr Detailed Review and Related Work.pptx  ← W2 deck: PoinTr/AdaPoinTr step-by-step + experiment ideas
├── W1 - Project Proposal.docx             ← the ambitious written proposal (layer 1 framing)
├── READING-LIST.md            ← prioritized paper list, tiers, per-paper deep-read rubric
├── WEEK-PLAN.md               ← day-by-day catch-up schedule (~June 13–19)
├── notes/
│   └── 00-KEY-FINDING-rgb-helps-completion.md  ← the literature-search result (section 3 above, in full)
│   └── <paper>.md             ← per-paper notes accumulate here as papers are read
└── papers/                    ← 24 source PDFs downloaded from arXiv (see READING-LIST.md for the map)
```

**What the source materials contain:**
- **W1 deck** — breadth/field map: point cloud basics; backbones (PointNet → PointNet++ → Point Transformer);
  then four task families — **registration** (FCGF, Predator, GeoTransformer, DCP), **completion** (PCN, PoinTr,
  AdaPoinTr, SnowflakeNet — *our lane*), **scene flow** (FlowNet3D, Neural Scene Flow Prior, ICP-Flow),
  **non-rigid/deformation** (Lepard, Neural Deformation Pyramid/Graphs). Only completion is the focus; the rest
  is context.
- **W2 deck** — depth: a careful walkthrough of the **PoinTr** pipeline (point proxies via FPS+DGCNN →
  geometry-aware transformer encoder → query generator → decoder → FoldingNet generation → Chamfer loss),
  a Transformer/FoldingNet refresher, then **AdaPoinTr** (adaptive query bank, denoising task), and the
  generative experiment proposals (deprioritized) + diffusion pointers (PVD, PDR).
- **Proposal** — the layer-1 vision (section 2). **Note its ComPC citation is wrong — see section 8.**

---

## 7. Open questions / watch items (raise with advisor)

1. **Scope decision:** focus on the mature **image-guided** direction (dataset + code ready, faster results) or
   push the harder, more novel **per-point joint xyz+rgb** direction? Both share the same first step (the
   geometry baseline), so this can be deferred until the baseline runs.
2. **Per-point RGB data:** ShapeNet-ViPC gives images, not colored points. Need to sample vertex/texture colors
   onto points from textured ShapeNet meshes — which ShapeNet subset has reliable textures? This blocks the
   per-point-RGB phase.
3. **Contribution type:** is a controlled *"when does RGB help"* empirical study an acceptable contribution on
   its own, or is a novel architecture expected?
4. **Read-only competitors:** GenPC has no code; ComPose's repo is empty (as of June 2026) → comparison by
   paper only.

---

## 8. Citation correction (fix before sharing the proposal)

The proposal cites **ComPC** as `arXiv:2307.14726` — **that ID is the wrong paper** (it's **P2C**, ICCV 2023,
self-supervised completion — coincidentally still relevant). The real **ComPC** (ICLR 2025) is
**`arXiv:2404.06814`**, "Completing a 3D Point Cloud with 2D Diffusion Priors"
(code: https://github.com/Tianxinhuang/ComPC). Verified-correct links for the other proposal papers:
GenPC `2502.19896`, ComPose `2605.25553`, PointDreamer `2406.15811`.

---

## 9. How to work on this project (conventions)

- **Persistent memory** lives at
  `/Users/macbookpro/.claude/projects/-Users-macbookpro-Desktop-point-cloud/memory/` — index in `MEMORY.md`,
  with `project-point-cloud-completion-research.md`, `user-research-collaboration.md`, `reference-paper-links.md`.
  Recalled each session; keep it updated as the project evolves.
- **Reading sessions:** per paper, produce a guided walkthrough (problem → method → the math that matters → the
  one table that proves the claim → limits → **RGB lens**: where would per-point color enter this architecture,
  and what would a controlled comparison vs the geometry baseline look like?). The researcher reads the PDF
  alongside; quiz at the end; save notes to `notes/<paper>.md`.
- **Researcher's working style:** student catching up fast (was ~2 weeks behind); prefers structured, prioritized
  guidance with a clear recommendation over exhaustive option-dumps. Wants to genuinely understand, not just be
  handed answers.
- **Tone for results:** honest and baseline-relative. Report negative results plainly. Visualize wherever it aids
  understanding (advisor explicitly asked for visualization).

---

## 10. Immediate next step

Per the advisor's two directives ("read the slides and papers" → "extract the baseline"), the project is in the
**reading phase**, working through Tier 0 of `READING-LIST.md` (PointNet → PointNet++ → DGCNN → PCN → FoldingNet
→ PoinTr → AdaPoinTr) on the `WEEK-PLAN.md` schedule. The next concrete action is to **continue the guided paper
reads**, and in parallel begin **recon of the `yuxumin/PoinTr` repo** so that by the end of the reading week the
geometry-only baseline can be stood up and reproduce ≈2.851 CD on ShapeNet-ViPC.
