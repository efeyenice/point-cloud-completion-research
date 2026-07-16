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
