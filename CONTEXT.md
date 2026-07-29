# Point Cloud Completion — Data Generation Context

Turning ShapeNet meshes into colored point clouds (XYZRGB) as the raw material for
the colored point-cloud completion study. This file is a glossary only — decisions
live in `docs/adr/`.

## Language

**Colored point cloud (XYZRGB)**:
An unordered set of points, each carrying a position (x,y,z) and a color (r,g,b),
sampled from a mesh surface. The data product this task generates.
_Avoid_: RGB cloud, textured point cloud.

**Mesh sampling**:
Scattering points over a mesh's triangle surface to produce a point cloud, giving
each point the color of the face it landed on (per-vertex color or texture lookup).
_Avoid_: meshing (that is the inverse operation), point extraction.

**Dual-face problem**:
ShapeNet meshes frequently store the same triangle twice with opposite normals and
sometimes different colors. Renderers hide the inward copy via back-face culling, but
surface sampling hits both copies, scattering two colors over one surface as
salt-and-pepper noise.
_Avoid_: double-sided faces, normal flipping.

**Naive sampling**:
Sampling color + geometry directly with CloudCompare WITHOUT removing duplicate inner
faces, so the output exhibits the dual-face speckle. The deliberate starting point for
this task (see ADR-0001).
_Avoid_: direct sampling, unfiltered sampling.

**Internal-face removal (ambient-occlusion fix)**:
The EPFL method of deleting inward-facing duplicate faces — detected via ambient
occlusion — before sampling, to eliminate the speckle. Deferred to a later iteration.
_Avoid_: face culling, dedup.

**Ambient occlusion (AO)**:
A per-face exposure score (fraction of surrounding viewpoints from which the face is
visible). The fix uses it to distinguish exterior faces from interior duplicates.

### Partial / complete generation (the s4 stage)

**Complete point cloud (ground truth, GT)**:
The full-shape target the network must predict — here the colored cloud FPS-downsampled
from the dense cloud to a fixed size. The "answer" half of a training pair.
_Avoid_: full cloud, target (ambiguous), label.

**Partial point cloud (partial)**:
The incomplete, single-viewpoint input — the surface visible from one camera, with the
occluded back/interior missing. The "question" half of a training pair.
_Avoid_: input cloud, occluded cloud, observation.

**Partial–complete pair**:
One (partial, GT) for the same model, in the same coordinate frame — a single training
example. Critically, the partial is NOT a subset of the GT (PCN: "X is not a subset of Y").
_Avoid_: sample, datum.

**Back-projected depth (depth back-projection)**:
PCN's partial mechanism: render a 2.5D depth image from a virtual camera viewing the
**mesh**, then unproject its pixels into 3D points. Gives true occlusion and perspective
density. Our `meshray_*` methods do this with an Open3D raycaster.
_Avoid_: depth render (ok loosely), unprojection.

**Viewpoint (view)**:
A virtual camera pose (a point on a sphere around the object, looking at its center) from
which one partial is generated. 8 per model, seeded and shared across all partial methods.
_Avoid_: angle, camera (the device), shot.

**Point-projection partial (ptproj)**:
The alternative partial method — project the dense colored cloud through the pinhole camera
and z-buffer to the nearest point per pixel; color comes directly from the surviving point.
A more sensor-faithful cousin of Hidden Point Removal.
_Avoid_: HPR (a different, hull-based algorithm), splatting.

**See-through (leakage)**:
The artifact where a point-cloud-based partial (ptproj) lets back-surface points show
through the gaps between front points, because points — unlike a mesh — don't fully
occlude. Worst on thin structures.
_Avoid_: bleed, ghosting.

### Training / evaluation (the s5 stage)

**s5 baseline**:
The geometry-only completion model trained on our own s4 pairs — the controlled twin
that every color experiment is measured against. One of two "baselines" in this project.
_Avoid_: "the baseline" unqualified (ambiguous with the published PoinTr baseline).

**Published PoinTr baseline**:
The literature reference number (PoinTr ≈2.851 CD ×10³ on ShapeNet-ViPC) that the project
eventually reproduces as a separate milestone. Not trained by us in s5.
_Avoid_: conflating it with the s5 baseline.

**Arm**:
One training configuration of the shared model, differing only in which channels it
consumes and predicts: A (xyz→xyz), B (xyzrgb→xyzrgb), C (xyzrgb→xyz). A→C isolates the
effect of color input; C→B isolates the effect of jointly predicting color.
_Avoid_: variant, experiment (both overloaded).

**Overfit ladder**:
The gate sequence — memorize one pair, then ten, then train for real — that proves the
pipeline can learn before any result is trusted. The advisor's sanity check.
_Avoid_: smoke test (that term belongs to data generation).

**Coarse cloud / fine cloud**:
The model's two predictions: a low-resolution global shape first (coarse), then the
full-resolution completion unfolded from it (fine). Each is supervised against a GT of
matching size.
_Avoid_: draft/final, low-res output.

**Folding grid**:
The small 2D patch of points each coarse point unfolds into around itself to produce the
fine cloud.
_Avoid_: upsampling kernel, tile.

**Model-level split**:
Partitioning train/val by model, never by pair — the 8 views of one model are
near-duplicates, so a pair-level split would leak shapes into val.
_Avoid_: random split (underspecified).

**Color error at NN correspondence**:
The color metric: rgb error between each predicted point and its geometrically nearest
GT point (and symmetrically), so color is judged only where geometry already matches.
_Avoid_: color chamfer (we do not mix rgb into the distance used for matching).

**Decoupled color head**:
Predicting color in a separate branch that reads only detached geometry features, so
color gradients cannot alter the predicted shape — color paints the geometry, never
steers it. Makes arm B's geometry the same optimization problem as arm C's by
construction.
_Avoid_: two-stage model (it is one network, one forward pass), frozen backbone
(geometry still trains — just not from color).

### The scale-curve stage (s6)

**Data-scale curve**:
The experiment that trains the three arms at several training-set sizes N and plots a
metric against N — here the color-hurts gap vs N — to show *how* an effect scales rather
than measuring it at one point. Answers the advisor's "under what conditions?".
_Avoid_: ablation (a single on/off comparison), sweep (underspecified).

**Color-hurts gap (A→C)**:
The geometry penalty from feeding color *in* — fine-CD of arm C minus arm A, as a % — the
curve's primary quantity. Positive = color input hurts geometry. s5 measured +17% at
~15 models/cat; the s6 curve answered the scaling question: the gap *shrinks* with N
(+6.6% → +4.8% → +4.0% at N = 20/60/150 per cat) but does not flip in the tested range.
_Avoid_: "the color effect" (ambiguous with the output tax).

**Color-output tax (C→B)**:
The geometry cost of *also predicting* color — fine-CD of arm B minus arm C. Expected ≈0
by construction (decoupled color head), so it doubles as a noise estimate. s5: +0.59;
s6: ≈0 at every rung (B even edges out C slightly, within seed noise).
_Avoid_: conflating with the A→C gap.

**Color-generalization gap**:
How far predicted color on *unseen* shapes sits above the oracle NN-painter floor (the
best any painter could do given the geometry). s5: 0.077 vs a ~0.016 floor; s6: 0.073 vs
a 0.020 floor at N=150 — better than NN-copy but still behind mean-copy. Tracks whether
color learns a transferable rule or just memorizes.
_Avoid_: raw color MSE (meaningless without the floor).

**Color-copy baseline (NN / mean / median)**:
Three training-free color predictors — give each output point the color of its nearest
input point, or the mean/median of its k nearest input points. The floor a *learned* color
head must beat; instantiates the advisor's "NN – mean – median" note.
_Avoid_: "the baseline" unqualified (already overloaded — see s5 baseline).

**Fixed held-out test set**:
A constant set of models never used in training, reserved once and scored at every curve
point, so N (train size) is the only thing that varies across the curve. Distinct from the
val set, which selects checkpoints / triggers early-stopping.
_Avoid_: conflating test with val; per-N re-splitting (changes the eval shapes).

**Uniformity descriptor**:
A *reported* number (spread of within-cloud nearest-neighbour spacing, from s5 §7) that
quantifies how evenly a predicted cloud covers the surface — Chamfer is blind to it. A
descriptor only this stage; turning it into a training loss is the deferred densification
thread.
_Avoid_: density (predicted points are actually *closer* on average — the issue is even-ness).

**Early-stopping on val**:
Halting a run once validation error stops improving for a patience window, rather than
running a fixed 300 epochs — s5 val bottoms out ~epoch 40 then overfits. Cheaper and
matches the best-on-val number we already report.
_Avoid_: "converged" (val improving stopped; train loss keeps falling).

**Continuity check**:
Re-measuring a previously observed regime *inside* a changed pipeline before trusting the
new pipeline's trend — the s6 curve's smallest rung re-measures s5's harm regime. Sign
agreement validates the effect; a magnitude difference (s5 +17% vs s6 +6.6%) quantifies
how pipeline-dependent the measurement is, and marks cross-pipeline numbers as
non-comparable.
_Avoid_: replication (implies an identical pipeline), sanity check (weaker claim).

**Residual color harm**:
The color-hurts gap remaining at the largest tested N — the curve's current endpoint
(s6: +4.0% at N=150/cat, sign-consistent across seeds). "Residual" flags that it may keep
shrinking beyond the tested range; it is a frontier, not a converged asymptote.
_Avoid_: "the final gap" (nothing final about the tested range).

### The generative phase (s7)

**Shape prior**:
What a completion model believes complete shapes look like *before* seeing the partial.
The occluded region is unobserved, so every completion = prior + evidence; models differ
only in whether the prior is hand-built (symmetry, mean shape), implicit (regressor
weights), or explicit and sampleable (a generative model).
_Avoid_: "prior" unqualified when color is in scope (see color prior).

**Color prior**:
The same notion for appearance — what colors are plausible given a geometry. Mean-copy
instantiates a trivial *local* color prior, and it is the bar a learned color model must
beat.
_Avoid_: conflating with the color-copy baseline mechanics (that's the estimator; this is
the concept).

**Sampler vs. regressor (regression-to-the-mean)**:
The fork the generative phase exists to study. Completion is ambiguous (many valid backs
for one front); a regressor trained on a symmetric loss hedges between the plausible
answers — over-smooth, mean-ish output — while a sampler draws one committed answer from
the learned distribution. Our uniformity issue and the mean-copy-unbeaten color head are
regression-to-the-mean symptoms.
_Avoid_: "generative" as vague praise; sharpness (an effect, not the mechanism).

**Generative phase (s7)**:
The project stage that recasts completion as conditional sampling from an explicit shape
prior, opened once the baseline-first phase (s5–s6) produced trustworthy regression
anchors — the sequencing the advisor set ("baseline first; revisit in the generative
phase").
_Avoid_: "the diffusion pivot" (it extends the controlled study, not replaces it).

**Two-row protocol (honesty row / distribution rows)**:
The generative-phase reporting rule: every sampled result carries an *honesty row* —
one-sample CD on the same fixed test set, directly comparable to the regression arms —
and *distribution rows* — best-of-k minimum CD and TMD. Exists because Chamfer rewards
hedging: a sampler can be genuinely better and still measure worse on one sample.
_Avoid_: "the new metric" (it is a protocol, not a metric).

**Best-of-k minimum CD**:
Draw k completions for one partial, report the minimum CD to GT — measures whether the
learned prior *covers* the true mode, independent of which sample committed to it.
_Avoid_: MMD (a set-level unconditional-generation metric).

**Total mutual difference (TMD)**:
The mean pairwise CD among the k samples for one partial — measures diversity /
commitment. Near-zero TMD = the sampler collapsed to a single answer (a regressor in
disguise).
_Avoid_: variance (underspecified).
