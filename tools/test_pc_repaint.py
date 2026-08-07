"""Unit tests for pc_repaint — CPU-only, no network, no Colab, tiny sizes.

Run: python3 tools/test_pc_repaint.py   (pytest tools/ also works)
"""

import os
import tempfile

import numpy as np
import torch

try:
    from pc_repaint import (Diffusion, PartColorDenoiser, append_eval_row, chamfer_l1,
                            cosine_betas, crop_xyz, deltaE, frame_scan, get_schedule_jump,
                            load_done_keys, make_repaint_input, nn_color, pairwise_diversity,
                            part_mean_colors, part_pool, repaint_colors, row_key, read_rows,
                            save_ckpt, separate_colored, srgb_to_lab, summarize_rows,
                            train_color_ddpm, try_load_ckpt)
except ImportError:
    from tools.pc_repaint import (Diffusion, PartColorDenoiser, append_eval_row, chamfer_l1,
                                  cosine_betas, crop_xyz, deltaE, frame_scan, get_schedule_jump,
                                  load_done_keys, make_repaint_input, nn_color, pairwise_diversity,
                                  part_mean_colors, part_pool, repaint_colors, row_key, read_rows,
                                  save_ckpt, separate_colored, srgb_to_lab, summarize_rows,
                                  train_color_ddpm, try_load_ckpt)

torch.manual_seed(0)
np.random.seed(0)


def _tiny_model(P=3, part_cond=True):
    torch.manual_seed(1)
    return PartColorDenoiser(P, width=16, k=4, n_blocks=1, part_cond=part_cond)


def test_srgb_lab_and_deltaE_reference():
    white, black = np.ones((1, 3)), np.zeros((1, 3))
    assert abs(deltaE(white, black)[0] - 100.0) < 1.0          # L* spans 0..100
    assert deltaE(white, white)[0] == 0.0
    lab = srgb_to_lab(np.full((1, 3), 0.5))
    assert 52.0 < lab[0, 0] < 55.0                              # mid-gray L* ~ 53.4
    assert abs(lab[0, 1]) < 0.5 and abs(lab[0, 2]) < 0.5        # neutral: a*,b* ~ 0


def test_crop_partition_and_multires_consistency():
    rng = np.random.default_rng(3)
    dense = rng.standard_normal((400, 3)).astype(np.float32)
    gt6 = np.concatenate([dense[:100], rng.uniform(0, 1, (100, 3)).astype(np.float32)], 1)
    partial, miss = separate_colored(gt6, crop=0.5, seed=7)
    assert len(partial) + miss.sum() == len(gt6) and miss.sum() == 50
    # same seed + same underlying object at 2 resolutions -> same region kept
    sparse_kept = crop_xyz(dense[:100], crop=0.5, seed=7)
    dense_kept = crop_xyz(dense, crop=0.5, seed=7)
    from scipy.spatial import cKDTree
    d_kept, _ = cKDTree(dense_kept).query(sparse_kept, k=1)
    d_all, _ = cKDTree(dense).query(sparse_kept, k=1)
    assert (d_kept <= d_all + 1e-6).mean() > 0.85


def test_diffusion_invariants():
    dif = Diffusion(T=16, device="cpu")
    b = dif.betas.numpy()
    assert (b > 0).all() and (b < 1).all()
    assert (np.diff(dif.abar.numpy()) < 0).all()                # abar strictly decreasing
    assert float(dif.abar_prev[0]) == 1.0
    x0 = torch.randn(2, 5, 3)
    t = torch.tensor([3, 9])
    q = dif.q_sample(x0, t, noise=torch.zeros_like(x0))         # zero noise -> pure scaling
    expect = dif.sqrt_abar[t].view(-1, 1, 1) * x0
    assert torch.allclose(q, expect)
    eps = torch.zeros(1, 5, 3)
    xt = torch.randn(1, 5, 3)
    m1, x0a = dif.p_sample(eps, xt, 0)
    m2, _ = dif.p_sample(eps, xt, 0)
    assert torch.equal(m1, m2)                                  # t=0 adds no noise
    assert x0a.abs().max() <= 1.0                               # x0 clamped


def test_jump_schedule_structure():
    T, j, U = 20, 5, 3
    ts = get_schedule_jump(T, j, U)
    assert ts[0] == T - 1 and ts[-1] == -1
    diffs = np.diff(ts)
    assert set(np.unique(diffs)) <= {-1, 1}                     # only unit moves
    assert (diffs == -1).sum() - (diffs == 1).sum() == T        # net descent = T
    assert get_schedule_jump(T, j, 1) == list(range(T - 1, -2, -1))  # U=1 -> plain DDPM


def test_part_pool_matches_naive():
    B, N, C, P = 2, 12, 5, 3
    h = torch.randn(B, N, C)
    part = torch.randint(0, P, (B, N))
    got = part_pool(h, part, P)
    for b in range(B):
        for p in range(P):
            m = part[b] == p
            if m.any():
                assert torch.allclose(got[b][m], h[b][m].mean(0).expand(int(m.sum()), C), atol=1e-5)


def test_denoiser_equivariance_and_n_independence():
    model = _tiny_model().eval()
    for N in (24, 33):                                          # N-independent by design
        xyz = torch.randn(1, N, 3)
        part = torch.randint(0, 3, (1, N))
        c_t = torch.randn(1, N, 3)
        out = model(c_t, torch.tensor(2), model.build_ctx(xyz, part))
        assert out.shape == (1, N, 3)
    perm = torch.randperm(24)
    xyz = torch.randn(1, 24, 3)
    part = torch.randint(0, 3, (1, 24))
    c_t = torch.randn(1, 24, 3)
    a = model(c_t, torch.tensor(2), model.build_ctx(xyz, part))
    b = model(c_t[:, perm], torch.tensor(2), model.build_ctx(xyz[:, perm], part[:, perm]))
    assert torch.allclose(a[:, perm], b, atol=1e-4)             # permutation-equivariant


def test_repaint_known_preserved_anchorless_deterministic():
    model = _tiny_model()
    dif = Diffusion(T=8, device="cpu")
    rng = np.random.default_rng(5)
    N = 30
    xyz = rng.standard_normal((N, 3)).astype(np.float32)
    part = np.array([0] * 20 + [1] * 10)
    known = np.zeros(N, bool)
    known[:12] = True                                           # only part-0 points visible
    rgb = np.zeros((N, 3), np.float32)
    rgb[:12] = rng.uniform(0, 1, (12, 3))
    out, diag = repaint_colors(model, dif, xyz, part, known, rgb, jump_length=3,
                               jump_n_sample=2, seed=0, device="cpu", return_diag=True)
    assert diag["anchorless_parts"] == [1]                      # part 1 has zero visible points
    assert np.allclose(out[known], rgb[known])                  # visible colors verbatim
    assert out.min() >= 0.0 and out.max() <= 1.0
    out2 = repaint_colors(model, dif, xyz, part, known, rgb, jump_length=3,
                          jump_n_sample=2, seed=0, device="cpu")
    assert np.allclose(out, out2)                               # seed-deterministic
    out3 = repaint_colors(model, dif, xyz, part, known, rgb, jump_length=3,
                          jump_n_sample=2, seed=1, device="cpu")
    assert not np.allclose(out, out3)                           # stochastic across seeds


def test_make_repaint_input_drop_by_oracle_trap():
    rng = np.random.default_rng(9)
    partial = np.concatenate([rng.standard_normal((20, 3)), rng.uniform(0, 1, (20, 3))], 1)
    comp = rng.standard_normal((15, 3))
    true_labels = np.array([1] * 10 + [0] * 10 + [0] * 15)      # first 10 visible pts are part 1

    def oracle_fn(xyz):
        return true_labels[:len(xyz)]

    def blind_seg(xyz):                                         # segmenter NEVER predicts part 1
        return np.zeros(len(xyz), np.int64)

    inp = make_repaint_input(partial.astype(np.float32), comp.astype(np.float32),
                             blind_seg, oracle_fn, num_parts=2, drop_part=1)
    # the trap: dropping by the blind segmenter would drop nothing; oracle-drop must fire
    assert not inp["known"][:10].any()                          # part-1 visible pts masked
    assert inp["known"][10:20].all()                            # part-0 visible pts stay known
    assert (inp["rgb"][:10] == 0).all()
    assert inp["n_vis"] == 20 and len(inp["xyz"]) == 35


def test_ckpt_signature_accept_reject():
    tmp = tempfile.mkdtemp()
    model = _tiny_model()
    sig = dict(src="auto", cat="Airplane", n_models=150, ep=300, part_cond=True)
    save_ckpt(tmp, "ddpm_part", model, sig, print_fn=lambda *a: None)
    fresh = _tiny_model()
    with torch.no_grad():
        for p in fresh.parameters():
            p.add_(1.0)                                         # make weights differ
    assert try_load_ckpt(tmp, "ddpm_part", fresh, sig, print_fn=lambda *a: None)
    pa = next(model.parameters()).detach()
    pb = next(fresh.parameters()).detach()
    assert torch.allclose(pa, pb)                               # weights actually restored
    # different config -> REJECTED (the synthetic-into-real trap)
    bad = dict(sig, src="labeled_s3")
    assert not try_load_ckpt(tmp, "ddpm_part", fresh, bad, print_fn=lambda *a: None)
    assert not try_load_ckpt(tmp, "missing", fresh, sig, print_fn=lambda *a: None)
    assert not try_load_ckpt(tmp, "ddpm_part", fresh, sig, force_retrain=True,
                             print_fn=lambda *a: None)
    with open(os.path.join(tmp, "junk.pt"), "wb") as f:
        f.write(b"not a checkpoint")
    assert not try_load_ckpt(tmp, "junk", fresh, sig, print_fn=lambda *a: None)


def test_eval_rows_resume_and_summary():
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "eval_rows.csv")
    assert load_done_keys(csv_path) == set()
    rows = [dict(model_id="m1", table="E1", crop=0.5, variant="RePaint-part", seed=s, value=v)
            for s, v in [(0, 10.0), (1, 20.0)]]
    rows += [dict(model_id="m2", table="E1", crop=0.5, variant="RePaint-part", seed=0, value=30.0)]
    for r in rows:
        if row_key(r) not in load_done_keys(csv_path):          # the driver's skip pattern
            append_eval_row(csv_path, r)
    done = load_done_keys(csv_path)
    assert len(done) == 3 and row_key(rows[0]) in done
    for r in rows:                                              # re-run: everything skipped
        assert row_key(r) in done
    s = summarize_rows(read_rows(csv_path))[("E1", 0.5, "RePaint-part")]
    assert abs(s["mean"] - 22.5) < 1e-9                         # mean of per-model means (15, 30)
    assert abs(s["best_mean"] - 20.0) < 1e-9                    # mean of per-model mins (10, 30)
    assert s["n_models"] == 2
    # diversity: two constant colorings -> pairwise ΔE == their ΔE
    a = np.tile([[1.0, 1.0, 1.0]], (8, 1))
    b = np.tile([[0.0, 0.0, 0.0]], (8, 1))
    d = pairwise_diversity([a, b])
    assert abs(d - deltaE(a[:1], b[:1])[0]) < 1e-9
    assert np.isnan(pairwise_diversity([a]))


def test_frame_scan_identity():
    rng = np.random.default_rng(11)
    gt = rng.standard_normal((40, 3)).astype(np.float32)
    probes = [dict(partial=gt[:25], gt=gt)]

    def complete_fn(partial, perm, sign):
        if tuple(perm) == (0, 1, 2) and tuple(sign) == (1, 1, 1):
            return gt                                           # correct frame -> perfect
        return gt + 5.0                                         # wrong frame -> blob far away

    scores = frame_scan(complete_fn, probes)
    assert len(scores) == 48
    best, perm, sign = scores[0]
    assert tuple(perm) == (0, 1, 2) and tuple(sign) == (1, 1, 1) and best < 1e-6
    assert chamfer_l1(gt, gt) == 0.0


def test_baselines_part_mean_and_nn():
    vis_rgb = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], np.float32)
    vis_lab = np.array([0, 0, 1, 1])
    out = part_mean_colors(vis_rgb, vis_lab, np.array([0, 1, 2]), num_parts=3)
    assert np.allclose(out[0], [1, 0, 0]) and np.allclose(out[1], [0, 1, 0])
    assert np.allclose(out[2], vis_rgb.mean(0))                 # anchorless -> global mean
    partial = np.array([[0, 0, 0, 1, 0, 0], [10, 0, 0, 0, 0, 1]], np.float32)
    got = nn_color(partial, np.array([[0.1, 0, 0], [9.8, 0, 0]], np.float32))
    assert np.allclose(got, [[1, 0, 0], [0, 0, 1]])


def test_train_color_ddpm_smoke():
    torch.manual_seed(2)
    model = _tiny_model(P=2)
    dif = Diffusion(T=8, device="cpu")
    rng = np.random.default_rng(1)
    clouds = [dict(xyz=rng.standard_normal((16, 3)).astype(np.float32),
                   rgb=rng.uniform(0, 1, (16, 3)).astype(np.float32),
                   part=rng.integers(0, 2, 16)) for _ in range(3)]
    hist = train_color_ddpm(model, dif, clouds, epochs=2, bs=2, device="cpu",
                            log=0, print_fn=lambda *a: None)
    assert len(hist) == 2 and all(np.isfinite(h) for h in hist)
    logged = []
    train_color_ddpm(model, dif, clouds, epochs=1, bs=2, device="cpu", log=0,
                     log_fn=lambda ep, l: logged.append((ep, l)), print_fn=lambda *a: None)
    assert logged and logged[0][0] == 0                         # W&B hook fires per epoch


if __name__ == "__main__":
    test_srgb_lab_and_deltaE_reference()
    test_crop_partition_and_multires_consistency()
    test_diffusion_invariants()
    test_jump_schedule_structure()
    test_part_pool_matches_naive()
    test_denoiser_equivariance_and_n_independence()
    test_repaint_known_preserved_anchorless_deterministic()
    test_make_repaint_input_drop_by_oracle_trap()
    test_ckpt_signature_accept_reject()
    test_eval_rows_resume_and_summary()
    test_frame_scan_identity()
    test_baselines_part_mean_and_nn()
    test_train_color_ddpm_smoke()
    print("ALL PC_REPAINT TESTS PASSED")
