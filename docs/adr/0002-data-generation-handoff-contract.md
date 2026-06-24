# Data-generation handoff contract (raw colored clouds + manifest)

This data-generation step delivers, per model, a raw colored point cloud and stops
there: one PLY (XYZRGB, uint8, ~100k points) in the model's NATIVE ShapeNet
coordinates — no unit-sphere normalization, no FPS downsampling, no partial/complete
pair generation, no colour-space conversion. Output layout is
`colored_pc/<synset>/<model_id>.ply` plus a `manifest.csv`
(synset, category, model_id, source_obj_path, output_ply_path, n_points, color_ok) and
an `errors.log`.

Why: it draws a clean division of labour with the teammate, who owns all downstream
processing (normalization, FPS, partial/complete pairs, RGB→Lab). Recording the
boundary — especially the explicit "we do NOT normalize or downsample" — stops either
side from duplicating or assuming the other did that work. The `manifest.csv` and its
`color_ok` flag (set by the automated colour-variance check) are the contract for which
models are actually usable, given that broken textures fail silently (see ADR-0001).
