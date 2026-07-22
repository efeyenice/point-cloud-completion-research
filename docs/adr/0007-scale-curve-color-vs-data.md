# A data-scale curve (fixed test set) to test whether per-point color helps *at scale*

s5 produced its first real result (`runs_s5`, 2026-07-18): with our 3-category, ~15-model/category
data, **color *input* hurt geometry — A→C = −17%**, consistent across all categories, color arms
overfitting ~3× sooner; while the **color *output* head was free — B−C = +0.59 ≈ noise**. The −17%
is the *opposite* of the cross-modal literature the project anchors on (EGIInet: color ≈ −58% CD),
and our own reading is that at ~15 shapes/category color acts as a **memorization fingerprint**
("the one with the red stripe") rather than a shape cue — i.e. almost certainly a **small-data
artifact**, not evidence color is useless. Everything overfits after ~epoch 40; **dataset size is the
binding constraint**. The advisor's 2026-07-22 menu (color as NN/mean/median/Gaussian; PCN vs
Snowflake; densification; more categories / full training; report CD numerically **and** visually;
diffusion later) is broad, but one question dominates the whole RGB thesis: **does the −17% flip as
data grows?** Nothing else is worth building until that is answered.

**Decision:** the week's deliverable is a **data-scale curve** — the A→C color-hurts gap plotted
against training-set size N — built under a **fixed held-out test set**, on the existing 3 categories:

| choice | what | why not the alternative |
|---|---|---|
| **Curve, not a single bigger re-run** | train A/B/C at **{20, 60, 150} models/cat** | a curve turns "did it flip?" into "here is *how* color's effect scales and where it crosses over" — the direct answer to "under what conditions?". A monotone trend is evidence even at 1 seed/point. |
| **Fixed test set, grow only train** | reserve ~30–40 models/cat as a never-trained test set + a val set for checkpoint/early-stop; sample train from the rest per rung | same eval shapes at every N ⇒ training-size is the *sole* variable; also dodges the tiny-test-set problem at small N that a per-N re-split creates. |
| **Deepen, don't broaden** | hold the 3 categories fixed; add *shapes*, not categories | broadening varies category *and* scale at once, confounding the very effect we're isolating. Broadening is a separate later axis. |
| **Seeds only at the top rung** | 1 seed per curve point; re-run **150/cat with 2 extra seeds** (3 total) | the curve's many (N, gap) points already average noise; spend the expensive seed multiplier only on the decisive top point that carries the headline claim. |
| **NN/mean/median color-copy baselines now; Gaussian deferred** | 3 training-free predictors (copy nearest / mean-of-kNN / median input-point color) reported beside the arms at every N | makes the otherwise-uninterpretable "0.077" color number mean "beats/ties/loses to trivial copying" — the cheap, honest half of the advisor's color note. The Gaussian probabilistic head is a real modeling change, orthogonal to the scale question, and gets its own week. |

Baked in regardless (not separate decisions, just prerequisites): **GPU/torch FPS** in data-gen
(the s4 pure-numpy FPS ≈ 8 min/model is what blocked scaling; GPU drops it to ~5 s/model, no CUDA to
compile); **early-stopping on val** (s5 val bottoms out ~epoch 40 then overfits — 300 fixed epochs is
waste, and best-on-val is what we report anyway); **meshray_nn** as the sole partial method (ADR-0005,
the report's recommendation). Curve metrics: **CD-L1/CD-L2 ×10³ + F-score@1%** (headline; advisor
emphasized CD), color MSE/PSNR vs the **oracle floor** and vs the copy-baselines, plus a per-N
**uniformity descriptor** (the s5 §7 NN-spacing number) as a cheap nod to "densification" and
Chamfer's blindness to point spread — but **no uniformity/EMD/repulsion *loss*** this week.

## Considered options / the honest tradeoffs

- **Single bigger re-run (one larger N).** Rejected: answers only yes/no at one scale; a curve costs
  little more (the small rungs are cheap) and yields a *trend*, which is the actual contribution.
- **Per-N random re-split.** Rejected: eval shapes differ per N (noisier curve) and test sets shrink
  to a few models at the small rungs — exactly the "12-model val" fragility s5 flagged.
- **3 seeds at every point.** Rejected for a one-week budget: it buys tight error bars at the cost of
  curve resolution; the top-rung-only seeding keeps both the trend and a defensible endpoint.
- **Add the Gaussian head / SnowflakeNet / a uniformity loss / more categories now.** All deferred:
  each is a real build competing with the scale question, and each is only *interpretable* once we
  know whether the core color-vs-geometry phenomenon survives scaling. They are the next spines, not
  this one.
- **Reduce views 8→~4 for the scaled runs.** Accepted as a feasibility lever, not a finding: shape
  diversity now comes from *models*; deltas are within-rung so view count cancels. Drop to 2 if Colab
  time bites.

## Consequences

- The headline the advisor sees becomes a **curve**, not a point: whether −17% shrinks/flips with N,
  with a seeded error bar on the top rung. If it hasn't fully flipped by 150/cat, "the harm shrinks
  monotonically toward zero" is still a real, publishable trend and motivates a bigger next run —
  honest either way.
- **Continuity check:** the N≈20/cat rung must roughly reproduce s5's −17%, or the scaled pipeline
  diverged from `runs_s5` and the curve can't be trusted.
- Compute: data-gen ≈ a weekend after the FPS fix; the curve ≈ ~15–30 GPU-hrs (resumable) — the
  top-rung seeds are ~half of it.
- Still **not literature-comparable** (3 categories); this measures *deltas and their scaling*, not
  absolute SOTA. Framing stays honest, as in ADR-0006.
- Reuses the s5 harness wholesale — the arms, decoupled color heads, Fourier PE, oracle floor, and
  gates are unchanged; only the split (fixed test + growable train), early-stop, copy-baselines, and
  a curve-driver cell are added.
