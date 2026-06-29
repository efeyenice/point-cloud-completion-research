# PCN-style partial generation by three methods (compare, don't pick yet)

We need **partial** inputs paired with the colored **complete (GT)** clouds, to train completion.
PCN makes partials by **rendering a 2.5D depth image from the mesh and back-projecting it to 3D**, from
8 random viewpoints, *deliberately not* as subsets of the complete cloud ("X is not a subset of Y", PCN §3/§5.1).
Our project extends PCN to **colored** completion, so the partial must also carry `rgb` — a channel PCN never
produces. That extra channel is the only thing PCN leaves undefined for us, and it forced a choice.

**Decision:** generate partials by **three** methods and ship **all** of them for side-by-side comparison
(supervisor + teammate evaluate), rather than committing to one now. This is an *exploration* step — the
project's whole ethos is "under what conditions does color help?", so we keep the options open and **report
the tradeoff of each, in each cell**.

| method | geometry | color | fidelity to PCN |
|---|---|---|---|
| `meshray_nn` *(reference)* | mesh raycast → back-project (Open3D) | nearest-neighbour from the cleaned dense EPFL cloud | **faithful** geometry; color is sub-mm-exact transfer |
| `meshray_tex` | *same* mesh raycast geometry | direct texture lookup at the ray-hit (barycentric→UV→texture) | faithful geometry; literally-direct color; **brittle**, falls back to NN |
| `ptproj` | project the dense colored cloud through the camera + z-buffer | direct, from the surviving point | point-based occlusion (leaks); partial ⊂ dense cloud |

All three use the **same 8 seeded viewpoints** per model, so a comparison isolates the *method*, never the
camera: `meshray_nn` vs `meshray_tex` is a pure **color** comparison (identical geometry); `meshray_*` vs
`ptproj` is a pure **geometry** comparison.

## Considered options / the honest tradeoffs

- **`meshray_nn` (our default).** Most PCN-faithful overall. The only compromise is colouring by
  nearest-neighbour — but at ~1M dense points the neighbour is sub-millimetre away, so the colour *equals*
  what texture-sampling would give, only more robust and already cleaned of the dual-face speckle by the EPFL
  step. _(Efe flagged NN as "feels sketchy" — recorded here. The reframe: it is not a fuzzy guess, it is
  reading a colour we already computed correctly onto the faithful geometry.)_
- **`meshray_tex`.** Colour comes literally from the texture at the exact hit — the "most direct" answer — but
  UV/material handling is fragile on Colab and re-opens the dual-face colour bug. Kept because the teammate may
  want to see it; wrapped in `try/except` with a **fallback to NN** so a failure never aborts the batch.
- **`ptproj` (the teammate's "shoot rays at the colored points" idea, made rigorous).** Simplest, and colour is
  direct. But a point cloud is not a solid surface, so the partial **sees through gaps** to the back surface
  (worst on thin parts — legs, wings), and the partial becomes a **subset of the dense cloud**, so it can share
  exact points with the GT — a step away from PCN's independent sampling.

## Consequences

- Partials are **`xyzrgb`** (a superset of `xyz`): the geometry-only PoinTr baseline simply ignores `rgb`; the
  later colored experiments use it. One dataset serves every milestone — no second data-gen run.
- Partials are **raw / variable size** (set by the render resolution, default 512²), faithful to PCN ("can have
  different sizes"); the PoinTr dataloader resamples to its fixed input (2048) at train time.
- GT is FPS-downsampled from the dense cloud (reusing s3's `color_aware_sample`, default 16384 — confirm
  against the PoinTr config); both halves live in the native `model_normalized` frame and share it automatically.
- We **defer** picking a single method on purpose. "Method X's partials look better / train better" is a finding
  to be measured, not assumed — consistent with the naive-vs-EPFL compare habit (ADR-0001).
