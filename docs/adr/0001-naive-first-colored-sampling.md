# Naive-first colored sampling (defer the dual-face fix)

For the first data-generation iteration we generate colored point clouds by sampling
ShapeNet meshes (airplane / car / chair) directly with CloudCompare, WITHOUT the
ambient-occlusion internal-face removal that fixes the dual-face color-corruption
problem. The output will therefore contain salt-and-pepper color noise — this is
accepted and expected, not a bug.

Why: today's goal is a minimal end-to-end pipeline (one CloudCompare sampling call per
mesh). This keeps moving parts low, de-risks getting CloudCompare to run headless on
Google Colab before adding pymeshlab on top, and lets us *visualize the dual-face
speckle on our own data* before fixing it. The fix (EPFL ambient-occlusion dedup —
roughly 250 of the 300 lines in `mesh_sampling_geo_color_shapenet.py`) is a later,
separately-measured iteration. A teammate consumes the colored clouds downstream
(normalization / FPS / partial–complete pairs).
