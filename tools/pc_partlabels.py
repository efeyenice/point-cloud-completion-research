"""Fuse ShapeNet-Part labels onto our textured HF bundles → labeled real-texture GT.

Self-sufficient replacement for the collaborator's `build_labeled_gt.py` step (which lives
only on her machine): part labels come from the ShapeNet-Part HDF5 benchmark mirror
(`larryshaw0079/ShapeNetPart` — 2048-pt clouds + per-point `seg` + `*_id2file.json` row→id
maps), transferred onto our 8192-pt textured bundle clouds by nearest neighbor after
matched normalization. Output npz uses HER exact labeled_s3 contract — keys `xyz (N,3)
f32`, `rgb (N,3) f32`, `part (N,) i16`, per-category label indices — so the training
notebook loads ours and her zip identically.

Robustness: ShapeNet-Part was annotated on ShapeNetCore v1; if our bundle's mesh frame
disagrees (v2 axis conventions differ for some models), a plain NN transfer silently
produces garbage labels. `fuse_labels` therefore QA-checks the mean NN distance and, when
it is poor, searches the 48 signed axis permutations and keeps the best — the chosen frame
and residual land in the returned diagnostics, and the driver refuses models that stay bad.

Everything heavy (HF download, bundle loading) is injected; the module unit-tests CPU-only
with tiny synthetic clouds (`python3 tools/test_pc_partlabels.py`).
"""

import glob
import json
import os

import numpy as np

# ShapeNet-Part standard: categories alphabetical (hdf5 `label` = index into this list),
# each category's parts contiguous in the global 0..49 range. Same tables as her notebook.
SEG_CLASSES = {"Airplane": [0, 1, 2, 3], "Bag": [4, 5], "Cap": [6, 7],
               "Car": [8, 9, 10, 11], "Chair": [12, 13, 14, 15],
               "Earphone": [16, 17, 18], "Guitar": [19, 20, 21], "Knife": [22, 23],
               "Lamp": [24, 25, 26, 27], "Laptop": [28, 29],
               "Motorbike": [30, 31, 32, 33, 34, 35], "Mug": [36, 37],
               "Pistol": [38, 39, 40], "Rocket": [41, 42, 43],
               "Skateboard": [44, 45, 46], "Table": [47, 48, 49]}

PART_NAMES_BY_CAT = {"Airplane": ["body", "wing", "tail", "engine"],
                     "Chair": ["back", "seat", "leg", "arm"]}   # display only

CAT_TO_SYNSET = {"Airplane": "02691156", "Bag": "02773838", "Cap": "02954340",
                 "Car": "02958343", "Chair": "03001627", "Earphone": "03261776",
                 "Guitar": "03467517", "Knife": "03624134", "Lamp": "03636649",
                 "Laptop": "03642806", "Motorbike": "03790512", "Mug": "03797390",
                 "Pistol": "03948459", "Rocket": "04099429", "Skateboard": "04225987",
                 "Table": "04379243"}
SYNSET_TO_CAT = {v: k for k, v in CAT_TO_SYNSET.items()}

H5_SPLITS = ["train0", "train1", "train2", "train3", "train4", "train5",
             "val0", "test0", "test1"]


def normalize_unit_sphere(xyz):
    xyz = np.asarray(xyz, np.float32)
    c = xyz.mean(0)
    return (xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-9)


# ---------------------------------------------------------------------------
# Label source: the ShapeNet-Part HDF5 benchmark (+ id2file row maps)
# ---------------------------------------------------------------------------

def load_h5_labeled_clouds(h5_dir, synset, splits=None):
    """Read every (h5, id2file.json) pair in h5_dir → {model_id: (xyz (2048,3), part (2048,))}.

    `part` is remapped from the global 0..49 range to the category's 0..P-1 (her
    convention: labeled_s3 and the synthetic source share part indices).
    """
    import h5py
    cat = SYNSET_TO_CAT[synset]
    off = SEG_CLASSES[cat][0]
    out = {}
    for split in (splits or H5_SPLITS):
        h5p = os.path.join(h5_dir, f"{split}.h5")
        idp = os.path.join(h5_dir, f"{split}_id2file.json")
        if not (os.path.exists(h5p) and os.path.exists(idp)):
            continue
        files = json.load(open(idp))
        with h5py.File(h5p, "r") as f:
            data, seg = f["data"][:], f["seg"][:]
        assert len(files) == len(data), f"{split}: id2file rows != h5 rows"
        for i, rel in enumerate(files):
            if not rel.startswith(synset + "/"):
                continue
            mid = os.path.splitext(os.path.basename(rel))[0]
            out[mid] = (data[i].astype(np.float32), (seg[i].astype(np.int64) - off))
    return out


# ---------------------------------------------------------------------------
# Fusion (NN label transfer with frame QA)
# ---------------------------------------------------------------------------

def _nn_transfer(gt_n, anno_n, anno_part):
    from scipy.spatial import cKDTree
    d, j = cKDTree(anno_n).query(gt_n, k=1)
    return anno_part[j], float(d.mean())


def fuse_labels(gt_xyz, anno_xyz, anno_part, dist_ok=0.05, frame_search=True):
    """NN-transfer part labels from the annotation cloud onto our GT cloud.

    Both clouds are unit-sphere normalized first. If the identity frame's mean NN
    distance exceeds `dist_ok`, all 48 signed axis permutations of the annotation cloud
    are tried and the best kept (v1/v2 mesh-frame mismatches). Returns
    (labels (N,) int64, diag) with diag = {mean_dist, frame, searched}.
    """
    gt_n = normalize_unit_sphere(np.asarray(gt_xyz, np.float32)[:, :3])
    anno_n = normalize_unit_sphere(anno_xyz)
    anno_part = np.asarray(anno_part, np.int64)
    labels, dist = _nn_transfer(gt_n, anno_n, anno_part)
    diag = dict(mean_dist=dist, frame=((0, 1, 2), (1, 1, 1)), searched=False)
    if dist <= dist_ok or not frame_search:
        return labels, diag
    try:
        from pc_repaint import apply_frame, frame_cands
    except ImportError:                                       # tools/ vs repo-root sys.path
        from tools.pc_repaint import apply_frame, frame_cands
    best = (dist, labels, ((0, 1, 2), (1, 1, 1)))
    for perm, sign in frame_cands():
        lab, d = _nn_transfer(gt_n, normalize_unit_sphere(apply_frame(anno_n, perm, sign)),
                              anno_part)
        if d < best[0]:
            best = (d, lab, (perm, sign))
    diag = dict(mean_dist=best[0], frame=best[2], searched=True)
    return best[1], diag


# ---------------------------------------------------------------------------
# Labeled-GT writer (her labeled_s3 npz contract) + resilient driver
# ---------------------------------------------------------------------------

def save_labeled_npz(out_dir, model_id, xyz, rgb, part, meta=None):
    """Atomic write of one labeled model — .tmp + os.replace, never a torn file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_id}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, xyz=np.asarray(xyz, np.float32),
                        rgb=np.asarray(rgb, np.float32),
                        part=np.asarray(part, np.int16),
                        meta=np.frombuffer(json.dumps(meta or {}).encode(), dtype=np.uint8))
    os.replace(tmp, path)
    return path


def load_labeled_npz(path):
    z = np.load(path)
    return z["xyz"], z["rgb"], z["part"].astype(np.int64)


def labeled_done_ids(out_dir):
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(out_dir, "*.npz"))}


def build_labeled_set(model_ids, load_bundle_fn, anno_clouds, out_dir, synset,
                      done=None, dist_reject=0.12, log=print):
    """Per-model resilient driver: fuse labels for every id not already done.

    model_ids: ids to build (usually bundles ∩ annotations). load_bundle_fn(mid) ->
    gt (M,6) — injected (HF download lives in the notebook). anno_clouds: from
    `load_h5_labeled_clouds`. Models whose best mean NN distance exceeds `dist_reject`
    are refused (bad frame/geometry match beats silently wrong labels). Returns
    (n_built, n_skipped, errors) like `pc_resilient.resilient_datagen`; failures are
    logged and retried on the next run.
    """
    done = set(done or set()) | labeled_done_ids(out_dir)
    n_built = n_skipped = 0
    errors = []
    for mid in model_ids:
        if mid in done:
            n_skipped += 1
            continue
        try:
            if mid not in anno_clouds:
                raise KeyError("no ShapeNet-Part annotation")
            gt = np.asarray(load_bundle_fn(mid), np.float32)
            anno_xyz, anno_part = anno_clouds[mid]
            labels, diag = fuse_labels(gt[:, :3], anno_xyz, anno_part)
            if diag["mean_dist"] > dist_reject:
                raise ValueError(f"label transfer too poor (mean NN {diag['mean_dist']:.3f})")
            save_labeled_npz(out_dir, mid, gt[:, :3], gt[:, 3:6], labels,
                             meta=dict(synset=synset, model_id=mid,
                                       label_source="shapenetpart_h5",
                                       mean_nn_dist=round(diag["mean_dist"], 5),
                                       frame=[list(diag["frame"][0]), list(diag["frame"][1])],
                                       frame_searched=diag["searched"]))
            n_built += 1
            log(f"  [{n_built + n_skipped}/{len(model_ids)}] {mid} "
                f"(nn {diag['mean_dist']:.3f}{', frame searched' if diag['searched'] else ''})")
        except Exception as e:
            errors.append((mid, f"{type(e).__name__}: {e}"))
            log(f"  !! {mid}: {type(e).__name__}: {e} (will retry next run)")
    return n_built, n_skipped, errors
