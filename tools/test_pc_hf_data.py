"""Local test of the HF->MODELS loader (no HF download needed: we write npz directly)."""
import os, tempfile
import numpy as np
from pc_resilient import save_model_npz
from pc_hf_data import load_models


def _write(npz_dir, mid, cat, color_std, n_views=8):
    gt = np.random.RandomState(0).rand(8192, 6).astype(np.float32)
    gt[:, 3:6] *= color_std                        # dial the color spread
    partials = [np.random.rand(2048, 6).astype(np.float32) for _ in range(n_views)]
    save_model_npz(npz_dir, mid, gt, partials,
                   {"synset": "0000", "category": cat, "model_id": mid,
                    "n_views": n_views, "gt_points": 8192, "color_std": float(color_std)})


def test_load_filters_and_shapes():
    d = tempfile.mkdtemp()
    _write(d, "a1", "chair", 0.3)
    _write(d, "a2", "car", 0.3)
    _write(d, "flat", "chair", 0.0)                # colorless -> must be dropped
    models, skipped = load_models(d, color_std_min=0.01)
    assert skipped == 1 and len(models) == 2, (skipped, len(models))
    m = next(x for x in models if x["model_id"] == "a1")
    assert m["gt"].shape == (8192, 6) and len(m["partials"]) == 8
    assert m["partials"][0].shape == (2048, 6) and m["category"] == "chair"
    # category filter
    only_car, _ = load_models(d, categories={"car"}, color_std_min=0.01)
    assert [x["model_id"] for x in only_car] == ["a2"]


if __name__ == "__main__":
    test_load_filters_and_shapes()
    print("PC_HF_DATA TESTS PASSED")
