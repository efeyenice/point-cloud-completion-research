# From-scratch PCN with configurable channels; three arms for the color question

s4 delivered the training pairs (`Ortak/data_s4`: 8192-pt colored GTs + 8 partials × 3 methods per
model, 60 models, zero errors). The next stage is the first **training** pipeline: "try the PCN way
but with color", produce a baseline, and prove the pipeline learns. The word **baseline** was
overloaded and is now split: the **s5 baseline** is a geometry-only PCN trained on *our* s4 data —
the controlled twin every color experiment is judged against; the **published PoinTr baseline**
(≈2.851 CD ×10³ on ShapeNet-ViPC) remains a separate, later reproduction milestone.

**Decision:** one **from-scratch PyTorch PCN** (encoder → coarse decoder → folding decoder) whose
input/output channel counts are configuration, trained as **three arms** on literally identical
data, split, seeds, and schedule — so any difference between arms is caused by channels, not setup:

| arm | input → output | what it isolates |
|---|---|---|
| `A_geo` | `xyz → xyz` | the **s5 baseline** |
| `B_joint` | `xyzrgb → xyzrgb` | joint geometry+color completion (the proposal's novelty) |
| `C_colin` | `xyzrgb → xyz` | does color **input** improve geometry? |

A→C measures the input-color effect; C→B measures the cost/benefit of jointly predicting color.
Running only A vs B would confound the two.

The supporting choices, all recorded in `notebooks/pcnTraining.ipynb` (T0 config):

- **Partials: `meshray_nn` only** this week — the reference method (ADR-0005). The tex/ptproj
  comparison stays banked as a ready-made ablation (data already on Drive).
- **Sizing:** input subsampled to 2048; coarse 1024 (PCN-canonical); **2×4 folding grid → fine
  8192 = GT size**, so every prediction↔GT number is density-matched. (Rectangular grids are legal
  folding; the alternative — canonical 4×4 → 16384 vs an 8192 GT — muddies interpretation.)
- **Chamfer is pure torch** (nearest-neighbour indices found under `no_grad`, distances recomputed
  on gathered pairs — same subgradient, no 268 MB matrix in the autograd graph). No CUDA
  extensions to compile on Colab.
- **Metric conventions**, stated once so future numbers are unambiguous (both ×10³):
  `CD-L1 = (mean_a min_b ‖a−b‖ + mean_b min_a ‖b−a‖)/2`;
  `CD-L2 = same with squared norms`. Plus F-score@1%. Color error = per-channel rgb MSE (and PSNR)
  at the **xyz**-nearest-neighbour correspondence, both directions — matching is geometric; a 6D
  Chamfer that mixes meters with rgb units is a banked ablation, not v1.
- **Color-ok subset everywhere:** models whose GT color std ≤ 0.01 (textureless) are excluded from
  **all** arms — including geometry-only A — so the comparison is never confounded by data.
- **Verification ladder as hard gates:** overfit-1 → overfit-10 (memorize fixed pairs; pass = fine
  CD-L2 ×10³ < 0.3 and visually identical panels, color MSE < 1e-3 for B) → only then the full runs
  with a **model-level** 16/4-per-category split (8 views of one model are near-duplicates; a
  pair-level split would leak).
- Runs write `config.json`, `metrics.csv`, best/latest checkpoints (resumable), and PNG/HTML panels
  to `Ortak/runs_s5/<run>/` — advisor-browsable without executing anything.

## Considered options / the honest tradeoffs

- **Adapt an existing PyTorch PCN repo (e.g. qinglew/PCN-PyTorch).** Rejected: it assumes its own
  dataset layout and compiled CUDA Chamfer extensions (classic Colab friction), and the color arms
  touch every layer anyway. Writing the ~200 lines ourselves keeps channel handling clean and — per
  the project's working style — the researcher understands every line. The from-scratch bug risk is
  exactly what the overfit gates exist to catch; the model/loss/loop cells were additionally
  smoke-tested locally (shapes for all three arms, gradient flow incl. the zero-distance edge case,
  a memorization trend, checkpoint-resume round-trip) before ever reaching Colab.
- **Official PCN code.** TensorFlow 1.x; effectively dead.
- **A + B only.** Rejected — confounded (input channels *and* output task change together).
- **Train on all three partial methods now.** Deferred; a finding to measure later, not assumed.

## Consequences

- With ~58 usable models, full-run numbers are **pipeline validation + controlled A/B/C deltas**,
  not literature-comparable performance. Honest framing in every report.
- Scaling the dataset (more models/categories) is a separate later step and is **blocked on the FPS
  bottleneck** observed in s4 (pure-numpy FPS ≈ 8 min/model on 1M-pt clouds; vectorize or GPU-FPS
  before regenerating at scale).
- The color-loss weight λ (default 1.0) has no principled prior; it is tuned on the overfit gates
  and both loss terms are logged separately so color can never silently trade against geometry.
- `runs_s5/comparison/` (table + side-by-side panels on identical val samples) is the week's
  deliverable to the advisor.
