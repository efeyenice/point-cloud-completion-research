"""Local test of the resilience layer (no Colab/CloudCompare/HF needed).

Proves the properties that make a Colab disconnect survivable: atomic npz round-trip,
per-model build, scratch cleanup, idempotent skip on re-run, and crash-recovery
(a model that failed mid-batch is rebuilt on the next run, finished ones are not)."""
import os, tempfile
import numpy as np
from pc_resilient import save_model_npz, load_model_npz, resilient_datagen


def make_fns(flaky_on=None):
    """Fake per-model functions. If flaky_on is set, gt_sample raises on that Nth call."""
    state = {"n": 0}

    def epfl_sample(obj, out, n):
        open(out, "w").close()                       # a scratch file, so cleanup is exercised

    def load_ply(path):
        return np.random.RandomState(0).rand(5000, 6).astype(np.float32)

    def gt_sample(dense):
        state["n"] += 1
        if flaky_on is not None and state["n"] == flaky_on:
            raise RuntimeError("simulated crash building this model")
        return dense[:2048]

    def cameras(dx):
        return ("vps", "Rl", "intr")

    def make_partials(obj, dx, drgb, vps, Rl, intr):
        return {"meshray_nn": [(np.zeros((100, 3), np.float32),
                                np.ones((100, 3), np.float32)) for _ in range(3)]}

    return dict(epfl_sample=epfl_sample, load_ply=load_ply, gt_sample=gt_sample,
                cameras=cameras, make_partials=make_partials)


def test_npz_roundtrip_is_atomic():
    d = tempfile.mkdtemp()
    gt = np.random.rand(8192, 6).astype(np.float32)
    parts = [np.random.rand(n, 6).astype(np.float32) for n in (100, 250)]
    save_model_npz(d, "m1", gt, parts, {"model_id": "m1", "category": "chair"})
    g2, p2, m2 = load_model_npz(os.path.join(d, "m1.npz"))
    assert np.allclose(g2, gt) and len(p2) == 2 and np.allclose(p2[1], parts[1])
    assert m2["model_id"] == "m1"
    assert not any(f.endswith(".tmp") for f in os.listdir(d))   # no half-written file left


def test_build_then_idempotent_skip_and_scratch_cleanup():
    local, scratch = tempfile.mkdtemp(), tempfile.mkdtemp()
    jobs = [("syn", "chair", f"mid{i}", "/fake.obj") for i in range(3)]
    cfg = dict(local_dir=local, scratch_dir=scratch, method="meshray_nn")
    b, s, e = resilient_datagen(jobs, make_fns(), cfg, done=set(), log=lambda *a: None)
    assert (b, s, e) == (3, 0, [])
    assert sorted(os.listdir(local)) == ["mid0.npz", "mid1.npz", "mid2.npz"]
    assert os.listdir(scratch) == []                              # scratch purged
    # re-run: everything already present -> all skipped, nothing rebuilt
    b, s, e = resilient_datagen(jobs, make_fns(), cfg, done=set(), log=lambda *a: None)
    assert (b, s) == (0, 3)


def test_crash_recovery():
    local, scratch = tempfile.mkdtemp(), tempfile.mkdtemp()
    jobs = [("syn", "chair", f"mid{i}", "/fake.obj") for i in range(3)]
    cfg = dict(local_dir=local, scratch_dir=scratch, method="meshray_nn")
    # 2nd model crashes mid-build
    b, s, e = resilient_datagen(jobs, make_fns(flaky_on=2), cfg, done=set(), log=lambda *a: None)
    assert b == 2 and len(e) == 1
    assert not os.path.exists(os.path.join(local, "mid1.npz"))    # failed one absent
    assert os.listdir(scratch) == []                              # scratch still clean
    # re-run healthy: the failed model is built, the finished two are skipped
    b, s, e = resilient_datagen(jobs, make_fns(), cfg, done=set(), log=lambda *a: None)
    assert (b, s, e) == (1, 2, [])
    assert len(os.listdir(local)) == 3


if __name__ == "__main__":
    test_npz_roundtrip_is_atomic()
    test_build_then_idempotent_skip_and_scratch_cleanup()
    test_crash_recovery()
    print("ALL RESILIENCE TESTS PASSED")
