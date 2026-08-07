# Adopt the collaborator's part-based RePaint color line as s7's opening move

Decided 2026-08-07, after the advisor meeting endorsed both the s6 scale-curve report and
**Pelin Kılıç's RePaint notebooks** (`RePaint_part.ipynb` / `RePaint_part_reel.ipynb`, shared
Drive, 2026-07-31). Her method: geometry completion by a **frozen pretrained PoinTr** (her fork);
the open problem is **coloring the filled-in points**. A DDPM over the **per-point RGB field** is
trained *unconditionally* (occlusion mask never seen) and conditioned only at **sampling time** via
RePaint (Lugmayr et al., CVPR 2022): known colors re-injected at the correct noise level each
reverse step + time-travel resampling. "Part-based" three ways: part-pooled denoiser features (a
*learned* part-mean), explicit anchorless-part accounting (a fully occluded part is filled from the
prior, not a neighbor's color), adaptive resampling budget. Her results: on synthetic part colors
the machine **works** (ΔE 22.5 vs ~36 for NN-copy/part-mean, 9/10 wins); on real texture her
**4-model × 1-seed first pass loses** (48.8 vs ~24.5) with a collapsed part-segmenter — explicitly
not a result yet. The mandate (Efe): *replicate her work — more samples, more data — under
production-grade workflow.*

**Decision:** adopt the RePaint color line as the **concrete opening move of the generative phase
(s7)** and productionize it (this repo: `tools/pc_repaint.py` + `tools/pc_partlabels.py` +
`notebooks/repaintTraining.ipynb`). ADR-0009 stands unrewritten; its from-scratch **geometry**
diffusion re-sequences *after* this line. The structural fit is exact: her D3+D4 is ADR-0009's
Phase 1 (an unconditional prior) plus the one conditioning branch that ADR left unwritten —
**sampling-time conditioning** — applied to the color half of the study, which is precisely where
our regression arms lost to mean-copy.

| choice | what | why not the alternative |
|---|---|---|
| **Faithful replication, formal experiment** | same method, same ΔE(Lab)-on-missing-region protocol, same six table rows; scaled to the full test split, ≥3–5 seeds, and the **25/50/75% occlusion sweep** her intro promises but never ran (occlusion = PoinTr's `seprate_point_cloud` viewpoint crop, per her note) | "fix the method first" repeats the mistake our house rules exist to prevent — her real-texture loss is a 4-model first pass; the port *is* the instrument that makes the true verdict cheap. Method fixes (part-seg collapse, capacity, T) go on the next-step menu. |
| **Two-row analog for color** | per-model **mean-of-seeds ΔE** (honesty row) + **best-of-k min-ΔE** (does the prior cover the truth) + **diversity** (mean pairwise ΔE between samples) — computed from the same samples, ~zero extra compute | one-sample-only is structurally rigged against samplers (ADR-0009's argument, unchanged); a full generative suite doesn't sharpen the color delta. |
| **Self-sufficient data** | labeled real-texture GT **built by us**: our HF airplane bundles (same recipe as her `build_gt.py`: textured-mesh sampling → EPFL dual-face cleanup → FPS 8192) + ShapeNet-Part labels fused by NN transfer with frame QA (`tools/pc_partlabels.py`) → `data_labeled/02691156/` on HF, her exact npz contract (137 models = bundles ∩ annotations; 107 chairs + 51 cars label-capable later) | her `labeled_s3` zip + build scripts exist only in her personal Drive/Colab (verified: not in the shared folder, not in the fork). Efe: "I don't want this dependent on anything." Her zip, if it arrives, becomes a **continuity cross-check row** — never critical path. |
| **Every external artifact mirrored on first touch** | her PoinTr fork forked into `efeyenice/Pelin_Efe_PoinTr` and **pinned** (`3f98676`); the ShapeNet55 PoinTr weights and the ShapeNet-Part h5 benchmark mirror into our HF dataset on first download | gdown links, personal forks, and third-party HF mirrors are outside our control and have burned runs before; ADR-0008's rule is self-owned resilience. |
| **Ops = ADR-0008 unchanged** | config-**signed** checkpoints (her Drive idea, made HF-native: signature covers data source/category/sizes/split/arch/epochs; mismatch → retrain), per-row resumable eval CSV, CommitScheduler sync, W&B curves, tested modules before any Colab run | her Drive checkpointing dies with the Drive-retirement; unsigned checkpoints silently load a synthetic-trained DDPM into a real-texture run (her own warning). |

## Considered options / the honest tradeoffs

- **Treat it as a service job (port only, roadmap unchanged).** Rejected: the line *is* the
  generative phase's color half; pretending otherwise duplicates work next week.
- **Replace ADR-0009 wholesale.** Rejected: the geometry-diffusion spine and its two-row eval are
  untouched by this adoption; they re-sequence, not die. Amending outcome > rewriting decision.
- **Headline on her zip (faithful data continuity).** Rejected for the dependency (above); the
  fused set follows her recipe end-to-end, and the parity gate is the synthetic stage, which is
  fully reproducible from public sources.
- **Fix the part-segmenter this week.** Deferred: the `oracle seg` rows already isolate segmenter
  error, so the formal eval cleanly separates "is the idea good" from "is the segmenter good
  enough" without new engineering.

## Consequences

- **Build:** `tools/pc_repaint.py` (DDPM schedule + RePaint jump schedule + part-conditional
  denoiser + sampler + config-signed ckpts + resumable eval rows; 13 tests) and
  `tools/pc_partlabels.py` (h5 label loading + NN fusion with 48-frame QA + atomic labeled-GT
  writer + resilient driver; 5 tests); `notebooks/repaintTraining.ipynb` (18 cells, T-cell style,
  both stages via `DATA_SOURCE`). Stage 0 ("auto") is the zip-free smoke gate: reproduce her
  synthetic-stage ordering before any real-texture compute.
- **The week's headline quantity:** the full-scale real-texture verdict — does the diffusion color
  prior beat part-mean/NN-copy once eval is no longer 4 models × 1 seed — reported honestly either
  way, with the occlusion sweep and the diversity row as the new information.
- **Credit:** method & original notebooks are Pelin's; port, scaling, fused dataset, and eval
  discipline are ours. The weekly report and this ADR say so explicitly. The HF dataset is public,
  so she can pull everything without tokens; no tooling change is forced on her.
- **Risk, logged:** 137 fused airplanes < her 200 (bundle-limited); the background data-gen job can
  regenerate bundles restricted to PartAnnotation ids if the advisor wants more. Fusion QA refuses
  models whose best-frame NN transfer stays poor — count reported, never silently written.
- **Next fork (not this week):** if the scaled eval still loses on real texture, the menu is
  part-seg repair, capacity/epochs, T/jump tuning, and the geometry-diffusion spine (ADR-0009
  Phase 1) picking up the RePaint conditioning mechanism for shape.
