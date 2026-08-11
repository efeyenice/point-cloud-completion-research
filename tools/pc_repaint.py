"""Part-based RePaint color inpainting — the tested core for the s7 notebook.

Method by Pelin Kılıç (`RePaint_part.ipynb` / `RePaint_part_reel.ipynb`, shared Drive,
2026-07-31), after *RePaint: Inpainting using Denoising Diffusion Probabilistic Models*
(Lugmayr et al., CVPR 2022). This module is a faithful port of her D1-D4 + E cells with the
ops layer swapped to ours (ADR-0008/0010): config-signed checkpoints that sync to HF instead
of Drive, resumable per-row eval CSVs, injectable segmenters so everything unit-tests on CPU.

The idea in one paragraph: geometry completion is done by a FROZEN pretrained PoinTr (that
works, we don't touch it). The open question is how the filled-in points get COLORED. A DDPM
is trained UNCONDITIONALLY on complete colored clouds — the occlusion mask is never seen in
training — and inpainting happens entirely at inference: at each reverse step the known
(visible) colors are re-injected at the correct noise level, and the schedule occasionally
jumps back up in time (resampling / "time-travel") so the generated region harmonizes with
the known one semantically, not just texturally (the RePaint paper's main finding).

"Part-based" in three places (her design):
  1. the denoiser pools features WITHIN each part (`part_pool`) — a learned version of the
     part-mean color rule; cross-part color bleed becomes structurally hard;
  2. parts with ZERO visible points are marked *anchorless* and filled from the learned
     prior instead of stealing a neighbor's color;
  3. the resampling budget adapts to the worst part's visibility.

Typical Colab wiring (thin cells; see notebooks/repaintTraining.ipynb). NOTE: import the
modules FLAT with `<repo>/tools` on sys.path — `from tools.pc_repaint import ...` breaks
next to a PoinTr checkout, because PoinTr ships a regular `tools` package
(tools/__init__.py) and regular packages shadow our namespace-style tools/ regardless of
sys.path order (its __init__ then imports the never-built emd extension):

    sys.path.insert(0, os.path.join(REPO_DIR, "tools"))
    from pc_repaint import (Diffusion, PartColorDenoiser, train_color_ddpm,
                            repaint_colors, make_repaint_input, separate_colored,
                            deltaE, nn_color, part_mean_colors,
                            save_ckpt, try_load_ckpt, pull_dir_from_hf,
                            load_done_keys, append_eval_row, summarize_rows)
    dif = Diffusion(T=200, device="cuda")
    model = PartColorDenoiser(num_parts).to("cuda")
    if not try_load_ckpt(CKPT_DIR, "ddpm_part", model, sig):
        train_color_ddpm(model, dif, clouds, epochs=300)
        save_ckpt(CKPT_DIR, "ddpm_part", model, sig)

Everything here is CPU-capable and Colab-free; `python3 tools/test_pc_repaint.py` must pass
before any Colab run (house rule).
"""

import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Color metric + occlusion crops (numpy)
# ---------------------------------------------------------------------------

def srgb_to_lab(rgb):
    """sRGB in [0,1] (N,3) -> CIELAB (D65). Used only for the ΔE metric."""
    rgb = np.clip(rgb, 0, 1)
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def deltaE(a, b):
    """Per-point CIE76 color difference between two (N,3) sRGB arrays."""
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=1)


def _crop_order(xyz, seed):
    """Order points by distance to a random view direction (nearest get cropped).

    The direction depends ONLY on the seed, so different resolutions of the same object
    are cropped from the SAME region (needed to feed PoinTr a denser partial than the
    2048-pt color cloud). This is PoinTr's `seprate_point_cloud` viewpoint-crop logic.
    """
    c = xyz.mean(0)
    n = (xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-9)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    return np.argsort(np.linalg.norm(n - v[None], axis=1))


def separate_colored(gt, crop=0.5, seed=0):
    """(N,6) colored GT -> (partial (M,6), miss_mask (N,) True=missing)."""
    order = _crop_order(gt[:, :3], seed)
    N = len(gt)
    nc = int(round(N * crop))
    mask = np.zeros(N, bool)
    mask[order[:nc]] = True
    return gt[order[nc:]], mask


def crop_xyz(xyz, crop=0.5, seed=0):
    """Crop the SAME region as `separate_colored` (same seed), xyz-only, any resolution."""
    order = _crop_order(xyz, seed)
    nc = int(round(len(xyz) * crop))
    return np.ascontiguousarray(xyz[order[nc:]]).astype(np.float32)


# ---------------------------------------------------------------------------
# D1 · DDPM scheme + RePaint jump schedule
# ---------------------------------------------------------------------------

def cosine_betas(T, s=0.008):
    """Nichol & Dhariwal cosine schedule — better than linear on low-dim signals."""
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)


class Diffusion:
    """Plain DDPM (eps-prediction) over an (N,3) color field in [-1,1]."""

    def __init__(self, T=200, device="cpu"):
        self.T, self.device = T, device
        b = cosine_betas(T).to(device)
        a = 1.0 - b
        abar = torch.cumprod(a, 0)
        abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]])
        self.betas, self.alphas, self.abar, self.abar_prev = b, a, abar, abar_prev
        self.sqrt_abar, self.sqrt_1mabar = abar.sqrt(), (1 - abar).sqrt()
        self.post_var = b * (1 - abar_prev) / (1 - abar)            # q(x_{t-1}|x_t,x_0)
        self.post_c0 = b * abar_prev.sqrt() / (1 - abar)
        self.post_ct = (1 - abar_prev) * a.sqrt() / (1 - abar)

    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        sa = self.sqrt_abar[t].view(-1, *([1] * (x0.dim() - 1)))
        sb = self.sqrt_1mabar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sa * x0 + sb * noise

    def p_sample(self, eps, x_t, t, generator=None):
        x0 = ((x_t - self.sqrt_1mabar[t] * eps) / self.sqrt_abar[t]).clamp(-1, 1)
        mean = self.post_c0[t] * x0 + self.post_ct[t] * x_t
        if t == 0:
            return mean, x0
        z = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return mean + self.post_var[t].sqrt() * z, x0

    def forward_jump(self, x, t, generator=None):        # RePaint time-travel: x_t -> x_{t+1}
        z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return self.alphas[t].sqrt() * x + self.betas[t].sqrt() * z


def get_schedule_jump(T, jump_length=10, jump_n_sample=5):
    """RePaint's resampling ("time-travel") schedule — same as the official repo.

    In the returned list, a consecutive DECREASING pair = one reverse-diffusion step, an
    INCREASING pair = a forward jump. Jumps make the generated region harmonize with the
    known region SEMANTICALLY; without them it only matches texturally (the paper's core
    finding).
    """
    jumps = {j: jump_n_sample - 1 for j in range(0, T - jump_length, jump_length)}
    t, ts = T, []
    while t >= 1:
        t -= 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                ts.append(t)
    ts.append(-1)
    return ts


# ---------------------------------------------------------------------------
# D2 · part-conditional denoiser
# ---------------------------------------------------------------------------

def knn_graph(xyz, k, chunk=4096):
    """(B,N,3) -> (B,N,k) neighbor indices (self excluded), chunked over queries."""
    B, N, _ = xyz.shape
    out = torch.empty(B, N, k, dtype=torch.long, device=xyz.device)
    kk = min(k + 1, N)
    for s in range(0, N, chunk):
        d = torch.cdist(xyz[:, s:s + chunk], xyz)
        idx = d.topk(kk, dim=-1, largest=False).indices[:, :, 1:]
        if idx.shape[-1] < k:
            idx = idx[..., [i % idx.shape[-1] for i in range(k)]]
        out[:, s:s + chunk] = idx
    return out


def _gather_nb(h, idx):                        # h (B,N,C), idx (B,N,k) -> (B,N,k,C)
    B, N, C = h.shape
    k = idx.shape[-1]
    off = (torch.arange(B, device=h.device) * N).view(B, 1, 1)
    return h.reshape(B * N, C)[(idx + off).reshape(-1)].reshape(B, N, k, C)


def part_pool(h, part, P):
    """Within-part mean of h, broadcast back per point. THIS is the "part-based" core:
    a part's (injected) visible colors drive its own missing points."""
    B, N, C = h.shape
    flat = (part + torch.arange(B, device=h.device).view(B, 1) * P).reshape(-1)
    s = torch.zeros(B * P, C, device=h.device, dtype=h.dtype).index_add_(0, flat, h.reshape(-1, C))
    n = torch.zeros(B * P, 1, device=h.device, dtype=h.dtype).index_add_(
        0, flat, torch.ones(B * N, 1, device=h.device, dtype=h.dtype))
    return torch.gather((s / n.clamp(min=1.0)).reshape(B, P, C), 1,
                        part.unsqueeze(-1).expand(B, N, C))


def timestep_embedding(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t.float().view(-1, 1) * f.view(1, -1)
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    """EdgeConv (local geometry) + part pool + global pool, FiLM'ed by t."""

    def __init__(self, w, part_cond=True):
        super().__init__()
        self.part_cond = part_cond
        self.edge = nn.Sequential(nn.Linear(2 * w + 4, w), nn.GELU(), nn.Linear(w, w))
        ctx = w * (3 if part_cond else 2)
        self.fuse = nn.Sequential(nn.LayerNorm(ctx), nn.Linear(ctx, w), nn.GELU(), nn.Linear(w, w))
        self.film = nn.Linear(w, 2 * w)

    def forward(self, h, idx, rel, part, P, temb):
        hj = _gather_nb(h, idx)
        hi = h.unsqueeze(2).expand_as(hj)
        e = self.edge(torch.cat([hi, hj - hi, rel], -1)).max(2).values
        g = h.max(1, keepdim=True).values.expand_as(h)
        c = [e, g] + ([part_pool(h, part, P)] if self.part_cond else [])
        d = self.fuse(torch.cat(c, -1))
        sc, sh = self.film(temb).unsqueeze(1).chunk(2, -1)
        return h + d * (1 + sc) + sh


class PartColorDenoiser(nn.Module):
    """eps-prediction for the per-point color field; conditioned on xyz (+ part).

    Permutation-equivariant and N-independent: trained on 2048-pt GT clouds, run on the
    ~7k-pt PoinTr union cloud. `part_cond=False` -> the part-blind (vanilla RePaint)
    ablation.
    """

    def __init__(self, num_parts, width=128, k=16, n_blocks=3, part_cond=True):
        super().__init__()
        self.P, self.k, self.part_cond, self.width = num_parts, k, part_cond, width
        cin = 3 + 3 + (num_parts if part_cond else 0)          # xyz, c_t, part one-hot
        self.inp = nn.Linear(cin, width)
        self.temb = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([Block(width, part_cond) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                 nn.Linear(width, 3))

    def build_ctx(self, xyz, part):
        """Geometry is FIXED across diffusion steps -> build the kNN graph once (big speedup)."""
        idx = knn_graph(xyz, self.k)
        rel = _gather_nb(xyz, idx) - xyz.unsqueeze(2)
        scale = rel.norm(dim=-1).mean(dim=(1, 2), keepdim=True).clamp(min=1e-6).unsqueeze(-1)
        rel = torch.cat([rel / scale, rel.norm(dim=-1, keepdim=True) / scale], -1)  # density-independent
        return dict(xyz=xyz, idx=idx, rel=rel, part=part,
                    onehot=torch.nn.functional.one_hot(part, self.P).float())

    def forward(self, c_t, t, ctx):
        B = c_t.shape[0]
        f = [ctx["xyz"], c_t] + ([ctx["onehot"]] if self.part_cond else [])
        h = self.inp(torch.cat(f, -1))
        temb = self.temb(timestep_embedding(t.expand(B) if t.dim() else t.repeat(B), self.width))
        for blk in self.blocks:
            h = blk(h, ctx["idx"], ctx["rel"], ctx["part"], self.P, temb)
        return self.out(h)


# ---------------------------------------------------------------------------
# D3 · UNCONDITIONAL training (no mask)
# ---------------------------------------------------------------------------

def train_color_ddpm(model, dif, clouds, epochs=300, bs=8, lr=2e-4, device=None,
                     log=25, log_fn=None, print_fn=print):
    """Unconditional DDPM training on FULL colored clouds — the occlusion mask is never seen.

    That is RePaint's whole point: ONE model is trained, then every occlusion ratio is
    handled at inference time with no retraining.
    clouds: [{xyz (N,3), rgb (N,3) in [0,1], part (N,)}]
    log_fn: optional callback(epoch, eps_mse) — the notebook wires W&B here.
    """
    device = device or next(model.parameters()).device
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    n, hist = len(clouds), []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = [clouds[i] for i in perm[s:s + bs]]
            xyz = torch.stack([torch.as_tensor(d["xyz"]) for d in b]).float().to(device)
            rgb = torch.stack([torch.as_tensor(d["rgb"]) for d in b]).float().to(device)
            prt = torch.stack([torch.as_tensor(d["part"]) for d in b]).long().to(device)
            x0 = rgb * 2 - 1                                      # [0,1] -> [-1,1]
            t = torch.randint(0, dif.T, (len(b),), device=device)
            noise = torch.randn_like(x0)
            loss = ((model(dif.q_sample(x0, t, noise), t, model.build_ctx(xyz, prt)) - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        hist.append(tot / n)
        if log_fn:
            log_fn(ep, hist[-1])
        if log and (ep % log == 0 or ep == epochs - 1):
            print_fn(f"  ep{ep:4d}  eps-MSE {hist[-1]:.4f}")
    return hist


# ---------------------------------------------------------------------------
# D4 · RePaint inference (part-based)
# ---------------------------------------------------------------------------

@torch.no_grad()
def repaint_colors(model, dif, xyz, part, known, known_rgb, *, jump_length=10,
                   jump_n_sample=3, adaptive_jumps=True, seed=0, device=None,
                   return_diag=False):
    """Part-based RePaint: inpaints the colors of the filled-in points.

    Faithful to Lugmayr et al. (2022): at every reverse step the KNOWN colors are
    re-injected at the correct noise level, and the schedule jumps back up so the
    generated region harmonizes with them:

        x_{t-1} = m . q(x_known, t-1)  +  (1-m) . p_theta(x_t)

    Part-based accounting (see module docstring): per-part visibility, anchorless parts,
    adaptive resampling budget scaled by the worst part's visibility.

    xyz (N,3), part (N,), known (N,) bool, known_rgb (N,3) in [0,1] -> (N,3) in [0,1].
    """
    device = device or next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    xyz_t = torch.as_tensor(xyz).float().unsqueeze(0).to(device)
    prt = torch.as_tensor(np.asarray(part)).long().unsqueeze(0).to(device)
    m = torch.as_tensor(np.asarray(known)).bool().view(1, -1, 1).to(device)
    c0 = (torch.as_tensor(np.asarray(known_rgb)).float().unsqueeze(0).to(device) * 2 - 1) * m

    # --- part-based anchor accounting ---
    pid = prt[0]
    vis = np.array([(m[0, :, 0][pid == p].float().mean().item() if (pid == p).any() else np.nan)
                    for p in range(model.P)])
    present = ~np.isnan(vis)
    anchorless = [p for p in range(model.P) if present[p] and vis[p] == 0.0]
    if adaptive_jumps and present.any():
        worst = float(np.nanmin(np.where(present, vis, np.nan)))
        jump_n_sample = int(np.clip(round(jump_n_sample * (1.5 - worst)), 1, 2 * jump_n_sample))

    model.eval().to(device)
    ctx = model.build_ctx(xyz_t, prt)
    x = torch.randn(1, xyz_t.shape[1], 3, device=device, generator=g)

    ts = get_schedule_jump(dif.T, jump_length, jump_n_sample)
    for t_cur, t_next in zip(ts[:-1], ts[1:]):
        if t_next < t_cur:                                          # --- reverse step
            eps = model(x, torch.tensor(t_cur, device=device), ctx)
            x_unknown, _ = dif.p_sample(eps, x, t_cur, generator=g)
            if t_cur > 0:
                noise = torch.randn(x.shape, device=device, generator=g)
                x_known = dif.q_sample(c0, torch.tensor([t_cur - 1], device=device), noise)
            else:
                x_known = c0
            x = torch.where(m, x_known, x_unknown)                  # mask = union of per-part masks
        else:                                                       # --- forward jump
            x = dif.forward_jump(x, t_cur, generator=g)

    out = ((x[0] + 1) / 2).clamp(0, 1).cpu().numpy()
    kn = np.asarray(known, bool)
    out[kn] = np.asarray(known_rgb, np.float32)[kn]                 # visible colors kept verbatim
    if return_diag:
        return out, dict(part_visibility=vis, anchorless_parts=anchorless,
                         jump_n_sample=jump_n_sample, n_steps=len(ts))
    return out


def make_repaint_input(partial, comp, seg_fn, oracle_fn, num_parts, drop_part=None,
                       use_oracle=False):
    """Union cloud = visible partial (color KNOWN) + PoinTr-filled points (UNKNOWN).

    partial (M,6), comp (K,3). seg_fn / oracle_fn: (N,3) xyz -> (N,) part labels
    (PointNet segmenter / nearest-GT oracle — injected so this stays testable).
    use_oracle=True -> labels from the oracle (isolates segmentation error).
    drop_part -> that part's visible points are ALSO masked -> the *anchorless* scenario.
    Dropping is ALWAYS done by the ORACLE label: otherwise, when the segmenter never
    predicts that part, "nothing gets dropped" and the experiment silently degenerates
    (this trap was hit once — her note).
    """
    xyz = np.concatenate([partial[:, :3], comp], 0).astype(np.float32)
    c = xyz.mean(0)
    xyz = ((xyz - c) / (np.linalg.norm(xyz - c, axis=1).max() + 1e-9)).astype(np.float32)
    known = np.zeros(len(xyz), bool)
    known[:len(partial)] = True
    rgb = np.zeros((len(xyz), 3), np.float32)
    rgb[:len(partial)] = partial[:, 3:6]
    part = (oracle_fn(xyz) if use_oracle else seg_fn(xyz)).astype(np.int64)
    assert part.max() < num_parts, f"label {part.max()} >= num_parts {num_parts}"
    if drop_part is not None:
        known &= (oracle_fn(xyz) != drop_part)
        rgb[~known] = 0
    return dict(xyz=xyz, rgb=rgb, known=known, part=part, n_vis=len(partial))


# ---------------------------------------------------------------------------
# Baselines + frame scan
# ---------------------------------------------------------------------------

def nn_color(partial, comp):
    """BASELINE 1 — nearest visible point's color. partial (M,6), comp (K,3) -> (K,3)."""
    from scipy.spatial import cKDTree
    _, i = cKDTree(partial[:, :3]).query(comp[:, :3] if comp.shape[1] > 3 else comp, k=1)
    return partial[i, 3:6]


def part_mean_colors(vis_rgb, vis_labels, out_labels, num_parts):
    """BASELINE 2 — per-part mean of the VISIBLE colors; anchorless part -> global mean."""
    mean = np.tile(vis_rgb.mean(0), (num_parts, 1))
    for k in range(num_parts):
        msk = vis_labels == k
        if msk.any():
            mean[k] = vis_rgb[msk].mean(0)
    return mean[out_labels]


def chamfer_l1(a, b):
    """Symmetric mean nearest-neighbor distance (low = good)."""
    from scipy.spatial import cKDTree
    d1, _ = cKDTree(b).query(a, k=1)
    d2, _ = cKDTree(a).query(b, k=1)
    return float(d1.mean() + d2.mean())


def frame_cands():
    """All 48 signed axis permutations (ShapeNet-Part -> PoinTr ShapeNet-55 frame)."""
    import itertools
    return [(p, s) for p in itertools.permutations(range(3))
            for s in itertools.product((1, -1), repeat=3)]


def apply_frame(xyz, perm, sign):
    """x'[:,i] = x[:,perm[i]] * sign[i]."""
    return np.ascontiguousarray(xyz[:, list(perm)] * np.asarray(sign, np.float32))


def invert_frame(xyz, perm, sign):
    inv = np.argsort(perm)
    s = np.asarray(sign, np.float32)
    return np.ascontiguousarray(xyz[:, inv] * s[inv])


def frame_scan(complete_fn, probes):
    """Score all 48 frames by Chamfer against GT; return sorted [(score, perm, sign)].

    complete_fn(partial_xyz, perm, sign) -> completed xyz (already back in input frame).
    probes: [{partial (M,3), gt (N,3)}]. Wrong frame -> PoinTr sees an unrecognizable
    blob (this failure was hit before; measuring all 48 beats eyeballing 3).
    """
    scores = []
    for perm, sign in frame_cands():
        cs = [chamfer_l1(complete_fn(pr["partial"], perm, sign), pr["gt"]) for pr in probes]
        scores.append((float(np.mean(cs)), perm, sign))
    scores.sort(key=lambda x: x[0])
    return scores


# ---------------------------------------------------------------------------
# Config-signed checkpoints (her Drive idea, made HF-native)
# ---------------------------------------------------------------------------
# Every checkpoint is locked to a config signature. If data source, category, sizes,
# split, architecture, or epochs change, the checkpoint is REJECTED and the model
# retrains — otherwise a synthetic-trained DDPM silently loads into a real-texture run
# and the table is garbage without anyone noticing (her warning, kept verbatim).

def ckpt_path(ckpt_dir, name):
    return os.path.join(ckpt_dir, f"{name}.pt")


def save_ckpt(ckpt_dir, name, model, meta, print_fn=print):
    os.makedirs(ckpt_dir, exist_ok=True)
    p = ckpt_path(ckpt_dir, name)
    tmp = p + ".tmp"
    torch.save({"sd": model.state_dict(), "meta": dict(meta)}, tmp)
    os.replace(tmp, p)                       # atomic: never a half-written checkpoint
    print_fn(f"  saved {name} -> {p}")
    return p


def try_load_ckpt(ckpt_dir, name, model, meta, device="cpu", force_retrain=False,
                  print_fn=print):
    """Load iff the stored signature matches `meta` exactly. Returns True on success."""
    p = ckpt_path(ckpt_dir, name)
    if force_retrain or not os.path.exists(p):
        return False
    try:
        z = torch.load(p, map_location=device, weights_only=False)
    except Exception as e:
        print_fn(f"  {name}: unreadable ({type(e).__name__}) -> retraining")
        return False
    want, got = dict(meta), z.get("meta", {})
    diff = {k: (got.get(k, "<absent>"), v) for k, v in want.items() if got.get(k) != v}
    if diff:
        print_fn(f"  {name}: checkpoint belongs to a DIFFERENT config -> retraining")
        for k, (g, w) in diff.items():
            print_fn(f"      {k}: checkpoint={g!r}  now={w!r}")
        return False
    try:
        model.load_state_dict(z["sd"])
        model.to(device)
    except Exception as e:
        print_fn(f"  {name}: state_dict mismatch ({type(e).__name__}) -> retraining")
        return False
    return True


def pull_dir_from_hf(repo_id, path_in_repo, local_dir, token=None, print_fn=print):
    """Restore a runs/ckpt directory from HF before anything runs (cross-VM resume).

    Mirrors the pcnTraining T9 restore pattern; returns True if anything was copied.
    Safe to call offline — returns False on any error.
    """
    try:
        import shutil
        from huggingface_hub import snapshot_download
        snap = snapshot_download(repo_id, repo_type="dataset", token=token,
                                 allow_patterns=[f"{path_in_repo}/*", f"{path_in_repo}/**"])
        src = os.path.join(snap, path_in_repo)
        if not os.path.isdir(src):
            return False
        os.makedirs(local_dir, exist_ok=True)
        shutil.copytree(src, local_dir, dirs_exist_ok=True)
        print_fn(f"  restored {path_in_repo}/ from HF -> {local_dir}")
        return True
    except Exception as e:
        print_fn(f"  no prior HF state ({type(e).__name__}) -> fresh start")
        return False


# ---------------------------------------------------------------------------
# Resumable eval rows ("done" = a row in the CSV = bytes on HF via CommitScheduler)
# ---------------------------------------------------------------------------
# Row = one (model, table, crop, variant, seed) measurement. The driver skips keys that
# already exist, so a dropped runtime resumes mid-table for free (ADR-0008 rule).

ROW_FIELDS = ["model_id", "table", "crop", "variant", "seed", "value"]


def row_key(row):
    return (str(row["model_id"]), str(row["table"]), f"{float(row['crop']):.2f}",
            str(row["variant"]), int(row["seed"]))


def load_done_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def append_eval_row(csv_path, row):
    """Append one measurement; header is written on first use. Caller checks done-keys."""
    new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in ROW_FIELDS})
        f.flush()


def read_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        out = []
        for r in csv.DictReader(f):
            r["crop"] = float(r["crop"])
            r["seed"] = int(r["seed"])
            r["value"] = float(r["value"])
            out.append(r)
        return out


def summarize_rows(rows):
    """Two-row summary per (table, crop, variant):
    mean-of-seeds (honesty row) and best-of-k (min over seeds) — ADR-0009's protocol
    philosophy applied to ΔE. Returns {(table, crop, variant): {...}}.
    """
    per_model = {}
    for r in rows:
        per_model.setdefault((r["table"], r["crop"], r["variant"], r["model_id"]), []).append(r["value"])
    grouped = {}
    for (table, crop, variant, mid), vals in per_model.items():
        g = grouped.setdefault((table, crop, variant), {"means": [], "bests": []})
        g["means"].append(float(np.mean(vals)))
        g["bests"].append(float(np.min(vals)))
    return {k: dict(mean=float(np.mean(g["means"])), std=float(np.std(g["means"])),
                    best_mean=float(np.mean(g["bests"])), n_models=len(g["means"]))
            for k, g in grouped.items()}


def pairwise_diversity(samples, sel=None):
    """Mean pairwise ΔE between k sampled colorings (TMD analog for the color field).

    samples: [(N,3) rgb], sel: optional bool mask restricting to the missing region.
    """
    if len(samples) < 2:
        return float("nan")
    vals = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i], samples[j]
            if sel is not None:
                a, b = a[sel], b[sel]
            vals.append(deltaE(a, b).mean())
    return float(np.mean(vals))
