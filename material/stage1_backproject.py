"""Stage 1: virtual slicing of the 3DFusion cylinder A[r,phi,z] and back-projection of per-slice
features into the cells.  Every core cell takes ONE nearest transverse slice and ONE nearest
longitudinal slice, so the two families contribute 1:1 at every radius by construction.

features per pixel: DINOv2-base (effective stride ~3.5 via 4x4 shifted crops, PCA-64),
                    LAB + local mean/std at 3 scales (21), Gabor energy 3x4 + structure-tensor
                    coherence x2 (14).  Output F(c) = [dino64 | lab21 | tex14 | r | |z|].
"""
import os, sys, math, time, numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from planes import Vol
STATE, GRID, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
dv = 'cuda'; RES = 512; STEP_T = int(os.environ.get('STEP_T', 2)); STEP_L = int(os.environ.get('STEP_L', 4))
os.makedirs(OUT, exist_ok=True); t0 = time.time()

# ---------------- cylinder + core mask (same construction as x3dcyl.py) ----------------
vol = Vol(GRID, dv, res=256); E = vol.EXT; c = vol.c
A = torch.load(STATE, map_location=dv)['A'].float(); _, NR, NPHI, NZ = A.shape; K = NPHI // 2
r = (torch.arange(NR, device=dv) + 0.5) / NR * E; ph = torch.arange(NPHI, device=dv) / NPHI * 2 * math.pi; z = -E + (torch.arange(NZ, device=dv) + 0.5) / NZ * 2 * E
RR, PP, ZZ = torch.meshgrid(r, ph, z, indexing='ij')
p0 = (RR * torch.cos(PP) + c).reshape(1, -1); p1 = (ZZ + c).reshape(1, -1); p2 = (RR * torch.sin(PP) + c).reshape(1, -1)
CORE = (vol._samp(vol.CORE.expand(3, -1, -1, -1).contiguous(), p0, p1, p2).reshape(3, NR, NPHI, NZ)[0] > 0.5)
print(f'cylinder {NR}x{NPHI}x{NZ}, core cells {int(CORE.sum()):,}', flush=True)

# ---------------- slice renderers (unlit; background white like the photographs) ----------------
def to_img(t):  # (3,H,W) in [-1,1] -> uint8 RGB
    return ((t.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

def render_trans(j):
    """polar (r,phi) slice at z index j -> Cartesian RES^2 image + fg mask + pixel->(r_idx,phi_idx) maps"""
    lin = torch.linspace(-E, E, RES, device=dv); V, U = torch.meshgrid(lin, lin, indexing='ij')
    rad = torch.sqrt(U ** 2 + V ** 2); ang = torch.atan2(V, U) % (2 * math.pi)
    ri = rad / E * NR - 0.5; pi_ = ang / (2 * math.pi) * NPHI
    Ap = torch.cat([A[:, :, :, j], A[:, :1, :, j] * 0 + A[:, :1, :, j]], 1)  # no-op pad kept for clarity
    Ap = torch.cat([A[:, :, :, j], A[:, :, :1, j]], 2)                       # phi wrap
    gg = torch.stack([pi_ / NPHI * 2 - 1, ri / (NR - 1) * 2 - 1], -1)[None]  # x=phi (W), y=r (H)
    img = F.grid_sample(Ap[None], gg, mode='bilinear', padding_mode='border', align_corners=True)[0]
    Cp = torch.cat([CORE[:, :, j], CORE[:, :1, j]], 1).float()
    fg = F.grid_sample(Cp[None, None], gg, mode='nearest', padding_mode='zeros', align_corners=True)[0, 0] > 0.5
    fg &= rad < E
    img = torch.where(fg[None], img, torch.ones_like(img))
    return to_img(img), fg.cpu().numpy(), (ri, pi_)

def render_long(k):
    """longitudinal slice at phi index k (< K): columns k+K (flipped) | k  -> (3, NZ, 2NR), upsampled to RES"""
    a = A[:, :, k, :]; b = A[:, :, k + K, :].flip(1)                 # (3, NR, NZ)
    s = torch.cat([b, a], 1).permute(0, 2, 1)                         # (3, NZ, 2NR): rows z, cols s
    ca = CORE[:, k, :]; cb = CORE[:, k + K, :].flip(0); cs = torch.cat([cb, ca], 0).T.float()
    img = F.interpolate(s[None], size=(RES, RES), mode='bilinear', align_corners=False)[0]
    fg = F.interpolate(cs[None, None], size=(RES, RES), mode='nearest')[0, 0] > 0.5
    img = torch.where(fg[None], img, torch.ones_like(img))
    return to_img(img), fg.cpu().numpy()

# ---------------- per-pixel features ----------------
def lab_feats(im, fg):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32); out = [lab]
    m = fg.astype(np.float32)
    for k in (5, 15, 31):
        mu = cv2.blur(lab * m[..., None], (k, k)) / (cv2.blur(m, (k, k))[..., None] + 1e-6)
        sq = cv2.blur(lab ** 2 * m[..., None], (k, k)) / (cv2.blur(m, (k, k))[..., None] + 1e-6)
        out += [mu, np.sqrt(np.clip(sq - mu ** 2, 0, None))]
    return np.concatenate(out, -1)                                    # 21

GABOR = [cv2.getGaborKernel((int(6 * lam) | 1,) * 2, sigma=lam * 0.6, theta=th, lambd=lam, gamma=0.7, psi=0)
         for lam in (6, 12, 24) for th in np.linspace(0, np.pi, 4, endpoint=False)]
def tex_feats(im, fg):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255; g = g * fg + g[fg].mean() * (~fg)
    out = [np.abs(cv2.filter2D(g, -1, k)) for k in GABOR]
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    for s in (3, 9):
        jxx, jyy, jxy = [cv2.GaussianBlur(a, (0, 0), s) for a in (gx * gx, gy * gy, gx * gy)]
        lam1 = 0.5 * (jxx + jyy + np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)); lam2 = 0.5 * (jxx + jyy - np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2))
        out.append((lam1 - lam2) / (lam1 + lam2 + 1e-6))
    return np.stack(out, -1)                                          # 14

from transformers import AutoImageProcessor, AutoModel
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base'); dino = AutoModel.from_pretrained('facebook/dinov2-base').to(dv).eval().half()
MEAN = torch.tensor(proc.image_mean, device=dv).view(3, 1, 1); STD = torch.tensor(proc.image_std, device=dv).view(3, 1, 1)
SH = [0, 4, 7, 11]; P = 14; D = 518
@torch.no_grad()
def dino_feats(im):
    """-> (RES, RES, 768) float16: 4x4 shifted 518-px crops, patch tokens interleaved to ~3.5 px"""
    x = torch.from_numpy(im).to(dv).permute(2, 0, 1).float() / 255
    x = F.interpolate(x[None], size=(D, D), mode='bilinear', align_corners=False)[0]
    pad = F.pad(x, (0, P, 0, P), mode='replicate')                       # room for shifts
    n = D // P; acc = torch.zeros(4 * n, 4 * n, 768, device=dv, dtype=torch.float16)
    batch = torch.stack([pad[:, sy:sy + D, sx:sx + D] for sy in SH for sx in SH])
    batch = ((batch - MEAN) / STD).half()
    tok = dino(pixel_values=batch).last_hidden_state[:, 1:].reshape(16, n, n, 768)
    for i, sy in enumerate(SH):
        for j, sx in enumerate(SH):
            acc[i::4, j::4] = tok[i * 4 + j]                              # token grid of shift (sy,sx) lands at offset (i,j)
    f = acc.permute(2, 0, 1)[None].float()
    f = F.interpolate(f, size=(RES, RES), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    return f.half()

# ---------------- pass 0: PCA for DINO on a few slices ----------------
rng = np.random.default_rng(0); samp = []
for j in range(0, NZ, NZ // 6):
    im, fg, _ = render_trans(j)
    if fg.sum() < 500: continue
    f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(4000, len(ys)), replace=False)
    samp.append(f[ys[sel], xs[sel]].float().cpu())
for k in range(0, K, K // 6):
    im, fg = render_long(k); f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(4000, len(ys)), replace=False)
    samp.append(f[ys[sel], xs[sel]].float().cpu())
S = torch.cat(samp); mu = S.mean(0); U, Sv, Vt = torch.linalg.svd(S - mu, full_matrices=False); PCA = Vt[:64].T.contiguous().to(dv); mu = mu.to(dv)
print(f'PCA-64 fit on {len(S):,} tokens, explained {float((Sv[:64]**2).sum()/(Sv**2).sum()):.3f}  [{time.time()-t0:.0f}s]', flush=True)

# ---------------- main pass: back-project into cells ----------------
NF = 64 + 21 + 14
core_idx = torch.nonzero(CORE)                                    # (M, 3): r, phi, z
M = len(core_idx); Ft = torch.zeros(M, NF, dtype=torch.float16, device=dv); Fl = torch.zeros_like(Ft)
ri_c, pi_c, zi_c = core_idx[:, 0], core_idx[:, 1], core_idx[:, 2]
def sample_feats(fmap, py, px):  # fmap (H,W,C) tensor, px/py float pixel coords -> (n, C)
    gg = torch.stack([px / (RES - 1) * 2 - 1, py / (RES - 1) * 2 - 1], -1)[None, None]
    return F.grid_sample(fmap.permute(2, 0, 1)[None].float(), gg, mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0, :].T
def full_feats(im, fg):
    d = (dino_feats(im).float() - mu) @ PCA
    l = torch.from_numpy(lab_feats(im, fg)).to(dv); t = torch.from_numpy(tex_feats(im, fg)).to(dv)
    return torch.cat([d, l, t], -1)                                  # (RES,RES,NF)

# transverse: cell z -> nearest sampled slice
js = list(range(0, NZ, STEP_T)); jz = torch.tensor(js, device=dv)
near_t = jz[torch.argmin((zi_c[:, None] - jz[None]).abs(), 1)]
for n, j in enumerate(js):
    sel = torch.nonzero(near_t == j).squeeze(1)
    if len(sel) == 0: continue
    im, fg, (ri_map, pi_map) = render_trans(j)
    if fg.sum() < 50: continue
    fm = full_feats(im, fg)
    # cell (r,phi) -> pixel: u = r cos phi, v = r sin phi in [-E,E] -> pixel = (u/E+1)/2*(RES-1)
    rr = (ri_c[sel].float() + 0.5) / NR * E; pp = pi_c[sel].float() / NPHI * 2 * math.pi
    px = (rr * torch.cos(pp) / E + 1) / 2 * (RES - 1); py = (rr * torch.sin(pp) / E + 1) / 2 * (RES - 1)
    Ft[sel] = sample_feats(fm, py, px).half()
    if n % 16 == 0: print(f'  transverse {n}/{len(js)}  [{time.time()-t0:.0f}s]', flush=True)

# longitudinal: cell phi (mod K) -> nearest sampled slice
ks = list(range(0, K, STEP_L)); kk = torch.tensor(ks, device=dv)
pm = pi_c % K; dk = (pm[:, None] - kk[None]).abs(); dk = torch.minimum(dk, K - dk); near_l = kk[torch.argmin(dk, 1)]
for n, k in enumerate(ks):
    sel = torch.nonzero(near_l == k).squeeze(1)
    if len(sel) == 0: continue
    im, fg = render_long(k)
    if fg.sum() < 50: continue
    fm = full_feats(im, fg)
    # cell -> (row=z, col=s): phi<K side -> col = NR + r ; phi>=K side -> col = NR-1-r ; then x2 for RES
    side = (pi_c[sel] >= K)
    col = torch.where(side, (NR - 1 - ri_c[sel]).float(), (NR + ri_c[sel]).float()) + 0.5
    px = col / (2 * NR) * RES - 0.5; py = (zi_c[sel].float() + 0.5) / NZ * RES - 0.5
    Fl[sel] = sample_feats(fm, py, px).half()
    if n % 16 == 0: print(f'  longitudinal {n}/{len(ks)}  [{time.time()-t0:.0f}s]', flush=True)

Fc = (Ft.float() + Fl.float()) / 2
geo = torch.stack([(ri_c.float() + 0.5) / NR, ((zi_c.float() + 0.5) / NZ * 2 - 1).abs()], 1)
np.savez_compressed(os.path.join(OUT, 'cells.npz'), F=Fc.half().cpu().numpy(), Ft=Ft.cpu().numpy(), Fl=Fl.cpu().numpy(),
                    geo=geo.half().cpu().numpy(), idx=core_idx.to(torch.int16).cpu().numpy(), shape=np.array([NR, NPHI, NZ]),
                    names=np.array(['dino'] * 64 + ['lab'] * 21 + ['tex'] * 14))
torch.save({'PCA': PCA.cpu(), 'mu': mu.cpu()}, os.path.join(OUT, 'dino_pca.pt'))
print(f'wrote {OUT}/cells.npz  {M:,} cells x {NF}  [{time.time()-t0:.0f}s]', flush=True)
