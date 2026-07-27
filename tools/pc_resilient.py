"""Resilient, HF-native data-generation driver (ADR-0008).

Turns the stage-by-stage s4 pipeline into a PER-MODEL atomic loop, so a Colab
disconnect costs one model instead of the whole run:

    for each model:  skip if already on HF  ->  build (dense -> GT + partials)
                     ->  write {model_id}.npz locally  ->  CommitScheduler streams it to HF

Recovery after a disconnect is just re-running the driver cell — it is idempotent.
"Done" means the bytes are on HF, never "it was in a Python list once."

The heavy per-model functions (CloudCompare/EPFL sampling, FPS, make_partials) are
passed IN as `fns`, so this module imports without Colab and is unit-testable. See
`test_pc_resilient.py` for a simulated-crash test of the skip/commit/cleanup logic.

Colab integration (thin cell — wires the notebook's existing functions to `fns`):

    import sys; sys.path.insert(0, "/content/point-cloud-completion-research")
    from tools.pc_resilient import (resilient_datagen, start_committer,
                                     hf_done_ids, anti_idle)
    from google.colab import userdata
    TOKEN, REPO = userdata.get("HF_TOKEN"), "efeyenice/pc-completion-data"
    LOCAL, SCRATCH = "/content/npz", "/content/scratch"

    def _cameras(dx):            vps, Rl, intr, *_ = cameras_for_model(dx); return vps, Rl, intr
    def _gt(dense):              return color_aware_sample(dense, N_GT_TARGET,
                                     color_frac=COLOR_FRAC, k=KNN_COLOR, seed=GT_SEED)
    def _mkp(obj, dx, drgb, vps, Rl, intr):
        parts, _ = make_partials(obj, dx, drgb, vps, Rl, intr,
                                 RENDER_RES, RENDER_RES, ["meshray_nn"]); return parts
    fns = dict(epfl_sample=lambda o, out, n: sample_colored_cloud_epfl(o, out, n=n),
               load_ply=load_ply_xyzrgb, gt_sample=_gt, cameras=_cameras, make_partials=_mkp)
    jobs = [(s, NAME_BY_SYNSET[s], mid, os.path.join(root, "models", "model_normalized.obj"))
            for s in SYNSETS for (mid, root) in selected[s]]

    anti_idle()
    sched = start_committer(LOCAL, REPO, TOKEN, every_min=5)
    done  = hf_done_ids(REPO, TOKEN)
    resilient_datagen(jobs, fns, dict(local_dir=LOCAL, scratch_dir=SCRATCH,
                                      method="meshray_nn"), done=done)
    sched.trigger()   # final flush to HF
"""
from __future__ import annotations
import json, os, traceback


def hf_done_ids(repo_id, token, path_in_repo="data"):
    """model_ids whose npz is already committed to the HF dataset repo (the skip set)."""
    from huggingface_hub import HfApi
    try:
        files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return set()
    pref = f"{path_in_repo}/"
    return {os.path.splitext(os.path.basename(f))[0]
            for f in files if f.startswith(pref) and f.endswith(".npz")}


def start_committer(local_dir, repo_id, token, every_min=5, path_in_repo="data"):
    """Background scheduler: uploads new/changed files in local_dir to HF every few minutes.
    The loss window on a crash is <= every_min. Call .trigger() for a final flush."""
    from huggingface_hub import CommitScheduler
    os.makedirs(local_dir, exist_ok=True)
    return CommitScheduler(repo_id=repo_id, repo_type="dataset", folder_path=local_dir,
                           path_in_repo=path_in_repo, every=every_min, token=token)


def save_model_npz(local_dir, model_id, gt, partials, meta):
    """Atomic write of one model's bundle: gt (M,6) + partial_XX (Ni,6) + meta(json).
    Writes a .tmp then os.replace() -> a reader/committer never sees a half-written file."""
    import numpy as np
    os.makedirs(local_dir, exist_ok=True)
    arrays = {"gt": np.asarray(gt, dtype=np.float32),
              "meta": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)}
    for vi, p in enumerate(partials):
        arrays[f"partial_{vi:02d}"] = np.asarray(p, dtype=np.float32)
    final = os.path.join(local_dir, f"{model_id}.npz")
    tmp = final + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, final)   # atomic on the same filesystem
    return final


def load_model_npz(path):
    """Read a bundle back -> (gt (M,6), [partial arrays], meta dict). For the loader + tests."""
    import numpy as np
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    parts = [z[k] for k in sorted(z.files) if k.startswith("partial_")]
    return z["gt"], parts, meta


def resilient_datagen(jobs, fns, cfg, done=None, log=print):
    """Per-model atomic driver.

      jobs : list of (synset, category, model_id, obj_path)
      fns  : dict of callables reused from the notebook —
               epfl_sample(obj, out_ply, n)         write a dense EPFL cloud to out_ply
               load_ply(path) -> (N,6) xyzrgb
               gt_sample(dense) -> (M,6)             FPS/color-aware GT
               cameras(dx) -> (vps, Rl, intr)
               make_partials(obj, dx, drgb, vps, Rl, intr) -> {method: [(xyz,rgb), ...]}
      cfg  : {local_dir, scratch_dir, method='meshray_nn'}
      done : set of model_ids already on HF (from hf_done_ids) -> skipped

    Returns (n_built, n_skipped, errors). A failed model is logged and skipped; the next
    run retries it (it never got an npz, so it isn't in the skip set)."""
    import numpy as np
    done = set(done or ())
    local_dir, scratch = cfg["local_dir"], cfg["scratch_dir"]
    method = cfg.get("method", "meshray_nn")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)
    built = skipped = 0
    errors = []
    for i, (synset, category, mid, obj) in enumerate(jobs):
        if mid in done or os.path.exists(os.path.join(local_dir, f"{mid}.npz")):
            skipped += 1
            continue
        epfl = os.path.join(scratch, f"{mid}.ply")
        try:
            fns["epfl_sample"](obj, epfl, i)
            dense = fns["load_ply"](epfl)
            dx, drgb = dense[:, :3], dense[:, 3:6]
            gt = fns["gt_sample"](dense)
            vps, Rl, intr = fns["cameras"](dx)
            parts = fns["make_partials"](obj, dx, drgb, vps, Rl, intr)[method]
            partials = [np.concatenate([xyz, rgb], axis=1) for xyz, rgb in parts]
            meta = {"synset": synset, "category": category, "model_id": mid,
                    "method": method, "n_views": len(partials),
                    "gt_points": int(np.asarray(gt).shape[0]),
                    "color_std": float(np.asarray(gt)[:, 3:6].std())}
            save_model_npz(local_dir, mid, gt, partials, meta)
            done.add(mid)
            built += 1
            log(f"  [{built}] {category}/{mid}: gt {meta['gt_points']} + {len(partials)} partials -> npz")
        except Exception:
            errors.append((mid, traceback.format_exc()))
            log(f"  ERROR {category}/{mid} (skipped; retried next run)")
        finally:
            if os.path.exists(epfl):
                os.remove(epfl)   # scratch never accumulates -> no local disk blow-up
    log(f"done: {built} built, {skipped} skipped, {len(errors)} errors")
    return built, skipped, errors


# Reduces (does not eliminate) Colab idle-disconnects; resumability is the real fix.
ANTI_IDLE_JS = ("function _ka(){document.querySelector('colab-connect-button')"
                "?.shadowRoot?.querySelector('#connect')?.click();}setInterval(_ka,60000);")


def anti_idle():
    """Call once in a Colab cell to click the reconnect button every 60s."""
    try:
        from IPython.display import Javascript, display
        display(Javascript(ANTI_IDLE_JS))
    except Exception:
        pass
