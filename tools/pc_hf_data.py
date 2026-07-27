"""Load the HF per-model npz dataset into the training MODELS list (ADR-0008).

Replaces the old Drive-PLY data path: `pcnTraining` used to build MODELS by reading
PLYs from `Ortak/data_s4`. Now it downloads the HF dataset of `{model_id}.npz` bundles
(produced by pc_resilient) once, and builds the same in-memory MODELS list the split +
PairDataset already expect — but with partials as arrays instead of file paths.

MODELS entry: {synset, category, model_id, gt (M,6) float32,
               partials [ (Ni,6) float32, ... ], color_std}
"""
import os, glob

try:
    from pc_resilient import load_model_npz          # when tools/ is on sys.path
except ImportError:
    from tools.pc_resilient import load_model_npz     # when the repo root is on sys.path


def download_dataset(repo_id, local_dir, token=None, path_in_repo="data"):
    """snapshot_download the dataset's npz files into local_dir. Returns the npz dir.
    Idempotent + resumable — re-running only fetches what changed."""
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir,
                      token=token, allow_patterns=[f"{path_in_repo}/*.npz"])
    return os.path.join(local_dir, path_in_repo)


def load_models(npz_dir, categories=None, color_std_min=0.01):
    """Build MODELS from a directory of {model_id}.npz. Colorless models
    (color_std <= color_std_min) are excluded from ALL arms (same rule as ADR-0006),
    so an A/B/C comparison is never confounded by which models carry color.
    Returns (models, n_skipped_colorless)."""
    import numpy as np
    models, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        gt, partials, meta = load_model_npz(path)
        gt = np.asarray(gt, np.float32)
        cstd = float(meta.get("color_std", gt[:, 3:6].std()))
        if cstd <= color_std_min:
            skipped += 1
            continue
        cat = meta.get("category")
        if categories and cat not in categories:
            continue
        models.append(dict(
            synset=meta.get("synset"), category=cat,
            model_id=meta.get("model_id", os.path.splitext(os.path.basename(path))[0]),
            gt=gt, partials=[np.asarray(p, np.float32) for p in partials], color_std=cstd))
    return models, skipped


def load_models_from_hf(repo_id, local_dir, token=None, categories=None, color_std_min=0.01):
    """Download the HF dataset then build MODELS. One call replaces the old T1+T2 data path."""
    npz_dir = download_dataset(repo_id, local_dir, token=token)
    return load_models(npz_dir, categories=categories, color_std_min=color_std_min)
