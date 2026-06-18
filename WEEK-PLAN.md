# Catch-up Week Plan — June 13–19, 2026

**Goal by end of week (advisor's milestones):**
1. ✅ Knowledge base built — slides + papers read
2. ✅ Baseline plan written + extraction started (geometry-only xyz, PoinTr family)

**Guiding frame (from advisor meeting):** exploratory, controlled study. Every later iteration (RGB, noise, architecture) gets judged against the geometry-only baseline. Read every paper through this lens: *where would RGB enter this architecture, and what would a controlled comparison look like?*

---

## Day-by-day

### Tonight (Thu 12) — optional warm-up, ~1h
- Re-skim the W1 deck end to end; star anything that feels shaky → bring to sessions.

### Day 1 — Fri 13: Foundations (the vocabulary)
- **PointNet** (deep): permutation invariance, shared MLP + max-pool, T-Net, critical points.
- **PointNet++** (medium): FPS, grouping (ball query/kNN), set abstraction, MSG.
- **DGCNN** (medium): EdgeConv, dynamic graph — read because PoinTr uses it for proxies.
- 🔑 Note while reading: PointNet-family input is N×(3+D) — extra per-point channels (normals... **RGB**) are *by design* pluggable. This is the architectural hook for our whole project.
- ✓ Self-test: explain why max-pool ⇒ permutation invariance; what EdgeConv adds over PointNet++ grouping.

### Day 2 — Sat 14: Completion problem + evaluation
- **PCN** (deep): encoder → coarse FC decoder → folding detail decoder; the benchmark protocol (8 viewpoints, partial 2048 → complete 16384).
- **FoldingNet** (medium): codeword, 2D-grid double folding (PoinTr's output head).
- **Metrics** (deep — this is the baseline's measuring stick): Chamfer Distance L1 vs L2, EMD, F-score@1%. Work through CD by hand on a toy example.
- ✓ Self-test: why is X not necessarily a subset of Y in PCN's formulation? When does CD mislead?

### Day 3 — Sun 15: PoinTr (the architecture we build on)
- **PoinTr** (deep, W2 deck side by side): FPS+DGCNN → point proxies → geometry-aware transformer encoder → query generator → decoder → FoldingNet head; two-stage CD loss.
- ✓ Self-test: redraw the full pipeline *from memory* with tensor shapes at every arrow.

### Day 4 — Mon 16: AdaPoinTr + the actual code
- **AdaPoinTr** (deep): adaptive query bank (Q_I + Q_O), scoring module, denoising queries + attention mask, why 15× faster training.
- **Code recon** ([yuxumin/PoinTr](https://github.com/yuxumin/PoinTr)): map paper → code (models/, cfgs/, dataset loaders); list env requirements (CUDA extensions: chamfer, pointnet2_ops — the usual pain points); note pretrained checkpoints available.
- ✓ Output: a short "what extracting the baseline concretely means" note.

### Day 5 — Tue 17: Zero-shot frontier + the RGB question
- **ComPC** (medium-deep): 3DGS init, zero-shot fractal completion, SDS from 2D diffusion.
- **GenPC** (medium): depth prompting, geometry-preserving fusion.
- **ComPose** (medium): unified completion+pose, keypoint progressive completion (AdaPoinTr lineage).
- **PointDreamer** (skim): the sequential project→inpaint→unproject texture pipeline.
- **P2C** (skim): self-supervised completion from partials only.
- ✓ Output: positioning map + open question list — esp. **where does colored point cloud data come from?** (PCN/ShapeNet-55 benchmarks are geometry-only.)

### Day 6 — Wed 18: Breadth + synthesis
- **SnowflakeNet** (medium skim): point-splitting coarse-to-fine — the main rival decoder design.
- **Point Transformer** (skim). W1 context papers (registration / scene flow / non-rigid) at one-paragraph level via the W1 deck.
- ✓ Output: 1–2 page synthesis note (field map + what we know + open questions for advisor).

### Day 7 — Thu 19: Baseline plan + first extraction steps
Write the baseline spec, then start executing:
1. Model: AdaPoinTr (or PoinTr config) from official repo, **geometry-only**.
2. Data: PCN benchmark first (ShapeNet-55 later).
3. Env: GPU choice (AWS/Google per proposal), CUDA deps.
4. Milestone ladder: (a) pretrained inference + **visualization** on a few samples → (b) reproduce paper eval numbers → (c) **overfit sanity check** on 1–10 samples (advisor's "can the model really learn?") → (d) short training run.
5. Ablation skeleton for later: +noise, +RGB channels, architecture variants.

---

## How sessions work (with Claude)

Per paper: say **"start <paper> session"** → I produce a guided walkthrough (problem → method → math that matters → key table → limits → RGB lens), you read the PDF alongside (all in `papers/`), ask me anything, then I quiz you. Notes land in `notes/<paper>.md`. Roughly 2 deep sessions/day + skims.

## Watch items (raise with advisor at week's end)
- **RGB data source**: standard completion benchmarks carry no color → options: sample point colors from ShapeNet *textured* meshes, RGB-D real scans (ScanNet/Redwood), or rendered data. Decides feasibility of the whole RGB phase.
- ComPose repo empty, GenPC no code → read-only comparisons.
- Proposal's ComPC arXiv link is wrong (fix before sharing): correct is 2404.06814.
