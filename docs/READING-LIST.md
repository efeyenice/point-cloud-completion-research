# Reading List — Point Cloud Completion Project

> Prioritized catch-up plan. Tiers ordered by how directly each paper feeds the proposal.
> Status: ⬜ not read · 🟨 skimmed · ✅ deep-read

---

## Tier 0 — The spine (your method builds directly on these)

| Status | Paper | Venue | Link | Why it matters here |
|---|---|---|---|---|
| ⬜ | PCN: Point Completion Network | 3DV 2018 | [arXiv:1808.00671](https://arxiv.org/abs/1808.00671) | The original learned completion pipeline: encoder → coarse → fine. Everything since is a reaction to it. |
| ⬜ | FoldingNet | CVPR 2018 | [arXiv:1712.07262](https://arxiv.org/abs/1712.07262) | The folding decoder PoinTr uses to turn proxies into points (W2 slides 21–25). |
| ⬜ | Attention Is All You Need | NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Transformer fundamentals (W2 slides 9–15). Skim if already solid. |
| ⬜ | DGCNN (EdgeConv) | TOG 2019 | [arXiv:1801.07829](https://arxiv.org/abs/1801.07829) | PoinTr's feature extractor for point proxies (W2 slide 5). |
| ⬜ | **PoinTr** | ICCV 2021 oral | [arXiv:2108.08839](https://arxiv.org/abs/2108.08839) | The core architecture: completion as set-to-set translation with point proxies. W2 deck is a walkthrough of this paper. |
| ⬜ | **AdaPoinTr** | TPAMI 2023 | [arXiv:2301.04545](https://arxiv.org/abs/2301.04545) | Adaptive queries + denoising task. The W2 experiment proposals modify *this* model. Code: [yuxumin/PoinTr](https://github.com/yuxumin/PoinTr) (same repo serves both). |

## ⭐ Tier 1A — Cross-modal / view-guided completion (THE prior art for "does RGB help?" — added 2026-06-12)

> This subfield = direct evidence that extra RGB improves geometric completion. EGIInet Table 1 (ShapeNet-ViPC, CD×10³): geometry-only PoinTr 2.851 → +image EGIInet 1.211 (~58% lower). BUT early ViPC (3.308) is *worse* than PoinTr — fusion quality decides. This is the "under what conditions does color help?" study, already seeded.
> Distinction from our proposal: this line uses a SEPARATE 2D image and outputs geometry only. Our open gap = per-point RGB (x,y,z,r,g,b) + outputting color too.

| Status | Paper | Venue | Link | Why it matters here |
|---|---|---|---|---|
| ⬜ | **ViPC** (View-Guided PC Completion) | CVPR 2021 | [arXiv:2104.05666](https://arxiv.org/abs/2104.05666) | Origin of the task; built **ShapeNet-ViPC** dataset (38,328 objs, 13 cats, 24 views). Our data + baseline starting point. |
| ⬜ | XMFnet (Cross-modal Learning) | ECCV 2022 | [arXiv:2209.09552](https://arxiv.org/abs/2209.09552) | Cross-attention image↔point fusion; CD 1.443. |
| ⬜ | CSDN (Cross-modal Shape-transfer Dual-refine) | TVCG 2023 | — | CD 2.570. |
| ⬜ | **EGIInet** | ECCV 2024 | [arXiv:2407.02887](https://arxiv.org/abs/2407.02887) | Current SOTA (CD 1.211). **Open-source + pretrained + dataset**: [WHU-USI3DV/EGIInet](https://github.com/WHU-USI3DV/EGIInet). Our cross-modal reference baseline. |
| ⬜ | Position-aware Guided Completion w/ CLIP | arXiv 2024 | [arXiv:2412.08271](https://arxiv.org/abs/2412.08271) | Uses CLIP/text-image priors for guidance. |

Per-point colored completion (closest to proposal's xyz+rgb output — thin literature = the gap):
- Joint Color & Semantic Scene Completion (RGB-D, scene-level) — [arXiv:2210.05891](https://arxiv.org/abs/2210.05891)
- GPN: Generative Point-based NeRF (completes colored clouds) — [arXiv:2404.08312](https://arxiv.org/abs/2404.08312)

## Tier 1 — The problem setting (zero-shot / category-agnostic completion — where the proposal lives)

| Status | Paper | Venue | Link | Why it matters here |
|---|---|---|---|---|
| ⬜ | **ComPC** | ICLR 2025 | [arXiv:2404.06814](https://arxiv.org/abs/2404.06814) | Zero-shot completion via 3D Gaussian Splatting + 2D diffusion inpainting + SDS. ⚠️ Proposal cites the wrong arXiv ID (2307.14726 = P2C). Code: [Tianxinhuang/ComPC](https://github.com/Tianxinhuang/ComPC) |
| ⬜ | **GenPC** | CVPR 2025 | [arXiv:2502.19896](https://arxiv.org/abs/2502.19896) | Zero-shot completion via 3D generative priors + depth prompting. No code released. |
| ⬜ | **ComPose** | CVPR 2026 oral | [arXiv:2605.25553](https://arxiv.org/abs/2605.25553) | Unified completion + pose estimation, category-agnostic, keypoint-based progressive completion inspired by AdaPoinTr. Best-paper candidate. Repo empty as of June 2026. |
| ⬜ | PointDreamer | arXiv 2024 | [arXiv:2406.15811](https://arxiv.org/abs/2406.15811) | The *sequential* texture baseline the proposal argues against (project → 2D inpaint → unproject). |
| ⬜ | P2C (optional) | ICCV 2023 | [arXiv:2307.14726](https://arxiv.org/abs/2307.14726) | Self-supervised completion from single partial clouds. The mis-cited paper — but actually relevant for training without complete GT. |

## Tier 2 — Generative machinery (▶ ACTIVE as of 2026-07-29 — the generative phase opened, ADR-0009; deep-read order: DDPM → PVD → PDR)

| Status | Paper | Venue | Link | Why it matters here |
|---|---|---|---|---|
| ⬜ | DDPM (background) | NeurIPS 2020 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | Diffusion fundamentals — prerequisite for PVD/PDR and the proxy-diffusion idea. |
| ⬜ | PVD: Point-Voxel Diffusion | ICCV 2021 | [arXiv:2104.03670](https://arxiv.org/abs/2104.03670) | Diffusion for 3D shape generation & completion (W2 slides 39–40). |
| ⬜ | PDR: Point Diffusion-Refinement | ICLR 2022 | [arXiv:2112.03530](https://arxiv.org/abs/2112.03530) | Conditional diffusion + refinement for completion (W2 slides 37–38). Closest prior art to "diffusion guided by a coarse completion". |

## Tier 3 — Backbones & completion alternatives

| Status | Paper | Venue | Link | Why it matters here |
|---|---|---|---|---|
| ⬜ | PointNet | CVPR 2017 | [arXiv:1612.00593](https://arxiv.org/abs/1612.00593) | Permutation invariance via max-pool; the field's starting point. |
| ⬜ | PointNet++ | NeurIPS 2017 | [arXiv:1706.02413](https://arxiv.org/abs/1706.02413) | Hierarchical local features, set abstraction; FPS+kNN grouping pattern reused everywhere (incl. PoinTr). |
| ⬜ | Point Transformer | ICCV 2021 | [arXiv:2012.09164](https://arxiv.org/abs/2012.09164) | Attention over local neighborhoods. |
| ⬜ | SnowflakeNet | ICCV 2021 | [arXiv:2108.04444](https://arxiv.org/abs/2108.04444) | Coarse-to-fine by point splitting; main non-transformer completion rival. |

## Tier 4 — Context from W1 (skim level — one-line takeaway each)

Registration: FCGF (ICCV 2019, [code](https://github.com/chrischoy/FCGF)) · Deep Closest Point ([arXiv:1905.03304](https://arxiv.org/abs/1905.03304)) · PREDATOR ([arXiv:2011.13005](https://arxiv.org/abs/2011.13005)) · GeoTransformer ([arXiv:2202.06688](https://arxiv.org/abs/2202.06688))

Scene flow: FlowNet3D ([arXiv:1806.01411](https://arxiv.org/abs/1806.01411)) · Neural Scene Flow Prior ([arXiv:2111.01253](https://arxiv.org/abs/2111.01253)) · ICP-Flow (CVPR 2024 — verify link on first read)

Non-rigid: Lepard ([arXiv:2111.12591](https://arxiv.org/abs/2111.12591)) · Neural Deformation Pyramid (NeurIPS 2022) · Neural Deformation Graphs (CVPR 2021) · DeGO (W1 slide 42 — identify this paper)

---

## Per-paper deep-read rubric

For every Tier 0–2 paper, answer:
1. **Problem** — what gap, in one sentence?
2. **Method** — the pipeline, drawn end to end; which design choice does the heavy lifting?
3. **Training signal** — loss(es), supervision source, datasets/benchmarks (PCN, ShapeNet-55/34, KITTI, Projected ShapeNet).
4. **Results** — which table proves the claim; by how much over what baseline?
5. **Limits** — what the paper admits + what it doesn't.
6. **For us** — what does this change about the proposal (joint xyz+rgb, category-agnostic, generative queries / proxy diffusion)?
7. **RGB lens (advisor's frame)** — where would per-point color enter this architecture? Would it plausibly help, hurt, or do nothing — and under what conditions? What would the controlled comparison against the geometry-only baseline look like?
