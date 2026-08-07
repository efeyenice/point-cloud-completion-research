"""Unit tests for pc_partlabels — CPU-only, no network, tiny synthetic clouds.

Run: python3 tools/test_pc_partlabels.py
"""

import json
import os
import tempfile

import numpy as np

try:
    from pc_partlabels import (SEG_CLASSES, build_labeled_set, fuse_labels,
                               labeled_done_ids, load_h5_labeled_clouds,
                               load_labeled_npz, normalize_unit_sphere, save_labeled_npz)
except ImportError:
    from tools.pc_partlabels import (SEG_CLASSES, build_labeled_set, fuse_labels,
                                     labeled_done_ids, load_h5_labeled_clouds,
                                     load_labeled_npz, normalize_unit_sphere,
                                     save_labeled_npz)


def _two_cluster_clouds(n_anno=600, n_gt=200, seed=0):
    """Annotation cloud + our GT cloud, three clusters placed so that NO signed axis
    permutation maps the shape onto itself — the correct frame is geometrically unique
    (mirror-symmetric toys would let the frame search pick a label-swapping frame, which
    real ShapeNet parts don't suffer from: wings are labeled 'wing' on both sides)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.5, 0.7]])
    labels = [0, 1, 0]
    fracs = [0.4, 0.4, 0.2]

    def cloud(n):
        xyz, part = [], []
        for c, lab, fr in zip(centers, labels, fracs):
            k = int(n * fr)
            xyz.append(rng.normal(c, 0.15, (k, 3)))
            part += [lab] * k
        return np.concatenate(xyz).astype(np.float32), np.array(part)

    anno_xyz, anno_part = cloud(n_anno)
    gt_xyz, gt_true = cloud(n_gt)
    return anno_xyz, anno_part, gt_xyz, gt_true


def test_fuse_labels_identity_frame():
    anno_xyz, anno_part, gt_xyz, gt_true = _two_cluster_clouds()
    labels, diag = fuse_labels(gt_xyz, anno_xyz, anno_part)
    assert (labels == gt_true).mean() > 0.97
    assert not diag["searched"] and diag["mean_dist"] < 0.05


def test_fuse_labels_recovers_flipped_frame():
    anno_xyz, anno_part, gt_xyz, gt_true = _two_cluster_clouds()
    # simulate a v1/v2 mesh-frame mismatch: annotation arrives axis-swapped + sign-flipped
    twisted = anno_xyz[:, [2, 0, 1]] * np.array([-1, 1, 1], np.float32)
    labels, diag = fuse_labels(gt_xyz, twisted, anno_part)
    assert diag["searched"]                                     # identity was rejected
    assert (labels == gt_true).mean() > 0.97                    # frame search recovered it
    # without the search, the transfer would be garbage on this geometry
    bad, _ = fuse_labels(gt_xyz, twisted, anno_part, frame_search=False)
    assert (bad == gt_true).mean() < 0.9


def test_labeled_npz_contract_and_atomicity():
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    xyz = rng.standard_normal((64, 3))
    rgb = rng.uniform(0, 1, (64, 3))
    part = rng.integers(0, 4, 64)
    p = save_labeled_npz(tmp, "abc123", xyz, rgb, part, meta=dict(synset="02691156"))
    assert not any(f.endswith(".tmp.npz") for f in os.listdir(tmp))   # atomic: no leftovers
    x, r, pt = load_labeled_npz(p)
    assert x.dtype == np.float32 and r.dtype == np.float32            # her labeled_s3 contract
    z = np.load(p)
    assert z["part"].dtype == np.int16
    assert x.shape == (64, 3) and r.shape == (64, 3) and pt.shape == (64,)
    assert (pt == part).all()
    meta = json.loads(bytes(z["meta"]).decode())
    assert meta["synset"] == "02691156"
    assert labeled_done_ids(tmp) == {"abc123"}


def test_build_labeled_set_resilient_and_idempotent():
    tmp = tempfile.mkdtemp()
    anno_xyz, anno_part, gt_xyz, _ = _two_cluster_clouds()
    gt6 = np.concatenate([gt_xyz, np.random.default_rng(2).uniform(0, 1, (len(gt_xyz), 3))],
                         1).astype(np.float32)
    calls = {"n": 0}

    def load_bundle(mid):
        calls["n"] += 1
        if mid == "flaky" and calls["n"] < 4:                   # fails on the first pass only
            raise IOError("simulated download crash")
        return gt6

    anno = {"m1": (anno_xyz, anno_part), "m2": (anno_xyz, anno_part),
            "flaky": (anno_xyz, anno_part)}
    ids = ["m1", "m2", "flaky", "no_anno"]
    built, skipped, errs = build_labeled_set(ids, load_bundle, anno, tmp, "02691156",
                                             log=lambda *a: None)
    assert built == 2 and skipped == 0
    assert {e[0] for e in errs} == {"flaky", "no_anno"}         # failures logged, not fatal
    # second run: done models skipped, the flaky one recovers, no_anno still refused
    built2, skipped2, errs2 = build_labeled_set(ids, load_bundle, anno, tmp, "02691156",
                                                log=lambda *a: None)
    assert built2 == 1 and skipped2 == 2
    assert {e[0] for e in errs2} == {"no_anno"}
    assert labeled_done_ids(tmp) == {"m1", "m2", "flaky"}
    # QA gate: a hopeless geometry match is refused rather than written with wrong labels
    junk = {"m1": (np.random.default_rng(3).uniform(-1, 1, (50, 3)).astype(np.float32),
                   np.zeros(50, np.int64))}
    built3, _, errs3 = build_labeled_set(["mx"], lambda m: gt6, junk | {"mx": junk["m1"]},
                                         tmp, "02691156", log=lambda *a: None)
    assert built3 == 0 and errs3 and "too poor" in errs3[0][1]


def test_load_h5_labeled_clouds_offsets():
    import h5py
    tmp = tempfile.mkdtemp()
    n, N = 3, 16
    rng = np.random.default_rng(4)
    data = rng.standard_normal((n, N, 3)).astype(np.float32)
    # Chair rows carry GLOBAL labels 12..15 — reader must remap to 0..3
    seg = np.full((n, N), 12) + rng.integers(0, 4, (n, N))
    with h5py.File(os.path.join(tmp, "train0.h5"), "w") as f:
        f["data"], f["seg"] = data, seg
        f["label"] = np.full((n, 1), 4)
    files = ["03001627/points/aaa.pts", "02691156/points/bbb.pts", "03001627/points/ccc.pts"]
    json.dump(files, open(os.path.join(tmp, "train0_id2file.json"), "w"))
    chairs = load_h5_labeled_clouds(tmp, "03001627")
    assert set(chairs) == {"aaa", "ccc"}
    xyz, part = chairs["aaa"]
    assert xyz.shape == (N, 3) and part.min() >= 0 and part.max() <= 3
    assert (part == seg[0] - SEG_CLASSES["Chair"][0]).all()
    assert set(load_h5_labeled_clouds(tmp, "02691156")) == {"bbb"}
    assert normalize_unit_sphere(xyz).max() <= 1.0 + 1e-6


if __name__ == "__main__":
    test_fuse_labels_identity_frame()
    test_fuse_labels_recovers_flipped_frame()
    test_labeled_npz_contract_and_atomicity()
    test_build_labeled_set_resilient_and_idempotent()
    test_load_h5_labeled_clouds_offsets()
    print("ALL PC_PARTLABELS TESTS PASSED")
