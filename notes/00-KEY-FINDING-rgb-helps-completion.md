# Key Finding — Prior art that RGB improves point cloud completion

_Literature search, 2026-06-12. This directly addresses the project's core hypothesis._

## TL;DR
"Does extra RGB/appearance info improve geometric point cloud completion?" is **already answered YES** by the
**view-guided / cross-modal completion** subfield (ViPC → XMFnet → CSDN → EGIInet). We should *build on* this,
not re-discover it. The genuinely open space is (a) **per-point RGB** instead of a separate image, and
(b) **outputting color** (joint geometry+texture), plus (c) a controlled **"when does color help?"** study.

## The evidence — EGIInet Table 1 (ShapeNet-ViPC, Mean Chamfer Distance ×10³, lower = better)

Geometry-only (partial point cloud input only):
- FoldingNet 6.271 · PCN 5.619 · GRNet 3.171 · **PoinTr 2.851** · PointAttN 2.853 · Seedformer 2.902

Cross-modal (partial point cloud + single RGB image):
- ViPC 3.308 · CSDN 2.570 · XMFnet 1.443 · **EGIInet 1.211**

**Headline:** best geometry-only (PoinTr 2.851) → best cross-modal (EGIInet 1.211) = **~58% CD reduction** from one RGB image.

**Nuance (the advisor's question, already in the data):** ViPC 3.308 is *worse* than geometry-only PoinTr 2.851.
Adding an image only helped once fusion got good (XMFnet/EGIInet). ⇒ *Color helps iff fused well; naive fusion hurts.*
A systematic study of this conditionality is a real contribution.

## The papers
| Paper | Venue | Link | Note |
|---|---|---|---|
| ViPC | CVPR 2021 | [2104.05666](https://arxiv.org/abs/2104.05666) | origin; ShapeNet-ViPC dataset |
| XMFnet | ECCV 2022 | [2209.09552](https://arxiv.org/abs/2209.09552) | cross-attention fusion |
| CSDN | TVCG 2023 | — | dual refinement |
| EGIInet | ECCV 2024 | [2407.02887](https://arxiv.org/abs/2407.02887) | SOTA + [code](https://github.com/WHU-USI3DV/EGIInet) |
| CLIP-guided | 2024 | [2412.08271](https://arxiv.org/abs/2412.08271) | text/image priors |
| Joint Color+Semantic (scene, RGB-D) | 2022 | [2210.05891](https://arxiv.org/abs/2210.05891) | per-point color, scene-level |
| GPN (Generative Point NeRF) | 2024 | [2404.08312](https://arxiv.org/abs/2404.08312) | completes colored clouds |

## ShapeNet-ViPC dataset (from ViPC paper, verified)
- 38,328 objects, **13 categories**, built on ShapeNet / ShapeNetRendering.
- **24 rendered image views per object** (same viewpoint setting as 3D-R2N2 / Pixel2Mesh).
- Experiments commonly use 31,650 objects (8 categories), 80/20 train/test.
- Pairs **partial point cloud + complete point cloud + multi-view RGB images** (143 GB; EGIInet repo links download).
- ⚠️ Provides paired **images**, NOT per-point colored point clouds. Image-guided experiments = plug-and-play.
  Per-point-RGB experiments still need color sampled onto points from textured ShapeNet meshes.

## How this maps onto our plan (advisor's "baseline → controlled iterations")
1. **Baseline** = PoinTr geometry-only on ShapeNet-ViPC; target reproduce ≈ 2.851 CD. (PoinTr is in the published table.)
2. **Iteration 1 (image-guided)** = add RGB image guidance, measure delta vs baseline. Reference SOTA = EGIInet (code available).
3. **Iteration 2 (per-point RGB)** = attach sampled color to points; does in-representation color help vs separate image?
4. **Iteration 3 (output color)** = predict rgb alongside xyz (joint completion) — the proposal's real novelty; thin prior art.
5. **The study** = ablate fusion type / noise / category to map *when* color helps (ViPC-hurts vs EGIInet-helps is the seed).

## Open questions for advisor
- Scope to **image-guided** (mature, dataset+code ready, faster results) or push the harder **per-point joint xyz+rgb** novelty first?
- Per-point RGB data: sample vertex colors from ShapeNet textured meshes? Which subset has reliable textures?
- Is "controlled study of when RGB helps" itself an acceptable contribution, or is a new architecture expected?
