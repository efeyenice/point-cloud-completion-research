# Open the generative phase (s7): from-scratch conditional point diffusion, controlled color arms preserved

Decided 2026-07-29 in a grilled session, ahead of this week's advisor meeting. The s6 curve is done
and read (ADR-0007): the color-input harm **shrinks but does not flip** (+6.6% → +4.0% at N=20→150/cat),
and the learned color head **beats NN-copy but not mean-copy**. Both residuals are
**regression-to-the-mean symptoms**: completion is ambiguous (many valid backs for one front), and a
Chamfer/L2-trained regressor hedges across the plausible answers — over-smooth geometry (our
uniformity issue), mean-ish color (losing to a trivial local color prior). The advisor's fragments
this week — *"treat like diffusion?"*, *"point clouds are continuous / fix a grid??"*, *"priors"* —
plus her original sequencing (Tier 2 reading "deprioritized — baseline first; revisit in the
generative phase") read as opening the door she always planned. Baselines now exist; time is not a
constraint (Efe, 2026-07-29).

**Decision:** open **s7 — the generative turn**. Recast completion as **conditional sampling from an
explicit, learned shape prior** (a from-scratch DDPM over point sets), while keeping the project's
identity: the controlled A/C(/B) deltas, honest baseline-relative reporting, and the resilient
HF-native ops. The question the phase answers: **does the color story change when the model samples
instead of regresses?**

| choice | what | why not the alternative |
|---|---|---|
| **Hybrid spine** | generative build in the foreground; curve **widening/extension as background data-gen** (more categories, N>150) | deepen-only is more of what we know, and the flip question needs more data anyway — which the background job pursues at near-zero attention cost; generative-only wastes cheap curve-firming and idle data-gen capacity. One background job feeds two customers (curve + diffusion, which is data-hungrier). |
| **Shape-first entry** | build the shape prior / completion sampler first; the color-head fight folds in later (B-diff outputs color natively) | color-first (the Gaussian head) partly misfires on our metric: under an L2-type score the optimal *point* prediction is the conditional mean, so a distributional head does not beat mean-copy *by sampling* — beating it needs either better conditional-mean estimation (a regression tweak) or the metrics turn this ADR makes. Build the machinery once, at the shape level. |
| **Fixed-N point diffusion** (PVD/PDR lineage) | diffuse the N×3 coordinates; per-point ε-net, FiLM-style timestep conditioning, partial-encoder features injected per point | a **voxel grid** demotes per-point color to voxel color (changes the research object), caps resolution (thin structures die at 64³), and breaks s6 comparability; **latent diffusion** (LION) is two-stage and muddies where color enters (A/B/C attribution). PVCNN-style *internal* voxelization stays available as gate-driven escalation — the grid *inside* the network, points at the interface. |
| **Two-row eval protocol** | every sampled result reports an **honesty row** (one-sample CD, same fixed test set / CD formula / point counts as s6 → directly comparable to the PCN arms) **and distribution rows** (best-of-10 minimum CD = does the prior cover the truth; TMD = does it commit to diverse answers) | one-sample-only is structurally rigged against samplers — Chamfer rewards hedging, so a better model can measure worse; the full generative suite (COV/MMD/1-NNA) targets *unconditional* generation and does not sharpen our delta. Deltas stay within-protocol (ADR-0007's honesty rule). House rule (Efe): any protocol is fine **as long as the methodology is fully reported**. |
| **Earned conditioning** | Phase 1 = **unconditional** DDPM on complete clouds, single category (airplane, the richest), 2048-pt pilot; gates: memorize-1 → memorize-10 → full-set samples pass a visual panel + best-of-k CD to held-out shapes | straight-to-conditional is faster to the headline but couples diffusion-machinery bugs (schedule, ε-target, sampler) with conditioning bugs — the overfit-ladder lesson. The unconditional sample panel is also the priors conversation made visible: *"airplanes sampled from our shape prior."* |
| **Phases 2–3 on the s6 top rung** | **A-diff** (xyz conditioning), then **C-diff** (xyzrgb conditioning), then B-diff if healthy — trained on the same 3-cat N=150 data, scored on the same fixed test set, at native 8192 fine points | any other data/test choice forfeits the free comparability that makes the honesty row meaningful; the **diffusion A→C delta under both rows is the phase's headline**. DDIM (~50–100 steps) makes best-of-10 sampling affordable at eval. |

## Considered options / the honest tradeoffs

- **Deepen-only (widen + seed + extend the curve).** Rejected as the spine — kept, in full, as the
  background job. It answers "is the trend category-general / does it flip," which stays open and
  cheap; it just no longer deserves the foreground.
- **Color-first (the advisor's Gaussian output head).** Deferred with a sharpened rationale (above):
  distributionality pays off only under distribution-aware metrics, so the metrics turn comes first;
  a learned head that beats mean-copy on the *current* metric is a locality/attention regression fix
  (a learned soft-NN copy), which is a fine later thread but not the phase-opener.
- **Voxel/SDF-grid diffusion (DiffComplete-style).** Rejected for the research-object reason.
  Reopened only if the advisor meant it specifically — advisor question #2.
- **Foundation-model priors (ComPC/GenPC zero-shot; pretrained 2D diffusion as the prior).** A
  *different project* — no training of our own prior, different claims. Parked pending advisor
  question #3 ("which sense of 'priors'?"), the one genuine ambiguity in her fragments.
- **Skip the unconditional pilot.** Rejected: coupled debugging, and it forfeits the one artifact
  that makes the phase legible to a non-implementer.

## Consequences

- **Build:** a new s7 notebook + a tested `tools/` module (noise schedule, ε-net, DDIM sampler,
  conditioning blocks), local unit tests before any Colab run, per-unit atomic HF commits + resume,
  W&B curves — ADR-0008 ops unchanged.
- **The headline quantity changes shape:** from "A→C gap vs N" (s6) to **"A→C gap, sampler vs
  regressor, at matched data"** — s6 becomes the regression anchor row in every s7 table.
- **Expected and accepted:** diffusion arms will likely *lose* the one-sample-CD honesty row to the
  PCN arms at first (samplers pay the hedging premium in reverse). That disagreement between rows is
  itself the measured regression-to-the-mean effect — it gets reported, not hidden.
- **Risk, logged:** diffusion is data-hungrier than PCN; at N=150/cat the prior may be weak. The
  claim is the *delta* (not absolute SOTA), and the background data-gen grows N continuously; if
  unconditional samples fail gates, escalate **data before architecture**.
- **The advisor meeting can redirect:** question #2 could reopen the grid, #3 could pivot toward
  foundation priors. This ADR stands unless the meeting says otherwise; the crib sheet is
  `notes/handoff-2026-07-29.md`.
