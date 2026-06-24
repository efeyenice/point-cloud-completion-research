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
