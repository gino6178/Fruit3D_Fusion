"""Learn the material inventory on REAL cut photographs, then label the 3DFusion asset with it.
1. features on the spl photographs (both families), per-image whitened, + radius
2. GMM; K chosen by transfer: fit on spl, fit on hld, Hungarian agreement on hld
3. the spl GMM labels every virtual slice of the asset; posteriors back-projected into cells
   (one nearest slice per family, averaged) -> labels + posteriors for Stage 2"""
import os, sys, glob, math, json, time, numpy as np, torch, torch.nn.functional as F, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from feats import *; from planes import Vol
from gmm_gpu import GaussianMixtureGPU as GaussianMixture; from scipy.optimize import linear_sum_assignment
PHOTO, STATE, GRID, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True); t0 = time.time(); rng = np.random.default_rng(0)
KS = [2, 3, 4, 5, 6]; STEP_T = 2; STEP_L = 4

def load_photos(split, fam):
    out = []
    for f in sorted(glob.glob(f'{PHOTO}/{split}_{fam}/*.png')):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
        out.append((os.path.basename(f)[:-4], im, fg_mask(im), fam))
    return out
spl = load_photos('spl', 'trans') + load_photos('spl', 'long'); hld = load_photos('hld', 'trans') + load_photos('hld', 'long')
print(f'photographs: {len(spl)} spl, {len(hld)} held-out', flush=True)

# PCA for DINO fitted on the spl photographs
S = []
for _, im, fg, _ in spl:
    f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(6000, len(ys)), replace=False); S.append(f[ys[sel], xs[sel]].float().cpu())
S = torch.cat(S); mu = S.mean(0); _, Sv, Vt = torch.linalg.svd(S - mu, full_matrices=False); PCA = Vt[:64].T.contiguous().to(dv); mu = mu.to(dv)
print(f'PCA-64 on {len(S):,} photo tokens, explained {float((Sv[:64]**2).sum()/(Sv**2).sum()):.3f}', flush=True)

def photo_feats(ph):
    X, meta = [], []
    for name, im, fg, fam in ph:
        f = all_feats(im, fg, PCA, mu); r = radius_map(fg, fam)
        ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(40000, len(ys)), replace=False)
        X.append(np.concatenate([f[ys[sel], xs[sel]], 3.0 * r[ys[sel], xs[sel], None]], 1)); meta.append((name, fam, len(sel)))
    return np.concatenate(X), meta
Xs, ms = photo_feats(spl); Xh, mh = photo_feats(hld)
print(f'features: spl {Xs.shape}, hld {Xh.shape}  [{time.time()-t0:.0f}s]', flush=True)

# K by transfer
res = {}; best = None
for K in KS:
    gs = GaussianMixture(K, covariance_type='full', random_state=0, reg_covar=1e-3, n_init=2).fit(Xs)
    gh = GaussianMixture(K, covariance_type='full', random_state=1, reg_covar=1e-3, n_init=2).fit(Xh)
    la, lb = gs.predict(Xh), gh.predict(Xh); C = np.zeros((K, K))
    for i, j in zip(la, lb): C[i, j] += 1
    r_, c_ = linear_sum_assignment(-C); agree = C[r_, c_].sum() / len(la)
    res[K] = dict(agree=float(agree), ll_hld=float(gs.score(Xh)), ll_spl=float(gs.score(Xs)))
    print(f'K={K}  spl->hld Hungarian agreement {agree:.3f}   loglik spl {gs.score(Xs):.2f} hld {gs.score(Xh):.2f}', flush=True)
    if best is None or agree > best[1] + 0.02 or (agree > best[1] - 0.02 and K > best[0]): best = (K, agree, gs)   # rule, no object-specific constant: best transfer; within 0.02 of it, prefer the larger K
K, agree, gmm = best; print(f'chosen K={K} (agreement {agree:.3f})', flush=True)
# order classes by mean radius
lab_s = gmm.predict(Xs); order = np.argsort([Xs[lab_s == k, -1].mean() if (lab_s == k).any() else 9 for k in range(K)]); remap = np.zeros(K, int); remap[order] = np.arange(K)

# ---- label the photographs (spl + hld) for the figure
def label_photo(im, fg, fam):
    f = all_feats(im, fg, PCA, mu); r = radius_map(fg, fam); X = np.concatenate([f, 3.0 * r[..., None]], -1)
    L = np.full(fg.shape, -1, int); L[fg] = remap[gmm.predict(X[fg])]; return L
photo_labels = [(n, im, label_photo(im, fg, fam), 'spl') for n, im, fg, fam in spl] + [(n, im, label_photo(im, fg, fam), 'hld') for n, im, fg, fam in hld]

# ---- the asset: virtual slices, same pipeline, posteriors back-projected
vol = Vol(GRID, dv, res=256); E = vol.EXT; c = vol.c
A = torch.load(STATE, map_location=dv)['A'].float(); _, NR, NPHI, NZ = A.shape; K2 = NPHI // 2
r = (torch.arange(NR, device=dv) + 0.5) / NR * E; ph = torch.arange(NPHI, device=dv) / NPHI * 2 * math.pi; z = -E + (torch.arange(NZ, device=dv) + 0.5) / NZ * 2 * E
RR, PP, ZZ = torch.meshgrid(r, ph, z, indexing='ij')
CORE = (vol._samp(vol.CORE.expand(3, -1, -1, -1).contiguous(), (RR * torch.cos(PP) + c).reshape(1, -1), (ZZ + c).reshape(1, -1), (RR * torch.sin(PP) + c).reshape(1, -1)).reshape(3, NR, NPHI, NZ)[0] > 0.5)
def to_img(t): return ((t.clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
def render_trans(j):
    lin = torch.linspace(-E, E, RES, device=dv); V, U = torch.meshgrid(lin, lin, indexing='ij'); rad = torch.sqrt(U ** 2 + V ** 2); ang = torch.atan2(V, U) % (2 * math.pi)
    gg = torch.stack([ang / (2 * math.pi) * NPHI / NPHI * 2 - 1, (rad / E * NR - 0.5) / (NR - 1) * 2 - 1], -1)[None]
    Ap = torch.cat([A[:, :, :, j], A[:, :, :1, j]], 2); img = F.grid_sample(Ap[None], gg, mode='bilinear', padding_mode='border', align_corners=True)[0]
    Cp = torch.cat([CORE[:, :, j], CORE[:, :1, j]], 1).float(); fg = (F.grid_sample(Cp[None, None], gg, mode='nearest', padding_mode='zeros', align_corners=True)[0, 0] > 0.5) & (rad < E)
    return to_img(torch.where(fg[None], img, torch.ones_like(img))), fg.cpu().numpy()
def render_long(k):
    a = A[:, :, k, :]; b = A[:, :, k + K2, :].flip(1); s = torch.cat([b, a], 1).permute(0, 2, 1)
    ca = CORE[:, k, :]; cb = CORE[:, k + K2, :].flip(0); cs = torch.cat([cb, ca], 0).T.float()
    img = F.interpolate(s[None], size=(RES, RES), mode='bilinear', align_corners=False)[0]; fg = F.interpolate(cs[None, None], size=(RES, RES), mode='nearest')[0, 0] > 0.5
    return to_img(torch.where(fg[None], img, torch.ones_like(img))), fg.cpu().numpy()
STATS = {}
def posterior_map(im, fg, fam):
    if fg.sum() < 50: return None
    f = all_feats(im, fg, PCA, mu, STATS.get(fam)); rmap = radius_map(fg, fam); X = np.concatenate([f, 3.0 * rmap[..., None]], -1)
    Pm = np.zeros((RES, RES, K), np.float32); Pm[fg] = gmm.predict_proba(X[fg])[:, order]          # columns reordered by radius
    return torch.from_numpy(Pm).to(dv)
# per-family whitening statistics from the CENTRAL virtual slice, matching the photographs (all central cuts)
from feats import raw_feats, fg_stats
_im, _fg = render_trans(NZ // 2); STATS['trans'] = fg_stats(raw_feats(_im, _fg, PCA, mu), _fg)
_im, _fg = render_long(0);        STATS['long']  = fg_stats(raw_feats(_im, _fg, PCA, mu), _fg)
core_idx = torch.nonzero(CORE); M = len(core_idx); ri_c, pi_c, zi_c = core_idx[:, 0], core_idx[:, 1], core_idx[:, 2]
Pt = torch.zeros(M, K, device=dv); Pl = torch.zeros(M, K, device=dv)
def samp(Pm, py, px):
    gg = torch.stack([px / (RES - 1) * 2 - 1, py / (RES - 1) * 2 - 1], -1)[None, None]
    return F.grid_sample(Pm.permute(2, 0, 1)[None], gg, mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0, :].T
js = list(range(0, NZ, STEP_T)); jz = torch.tensor(js, device=dv); near_t = jz[torch.argmin((zi_c[:, None] - jz[None]).abs(), 1)]
for n, j in enumerate(js):
    sel = torch.nonzero(near_t == j).squeeze(1)
    if len(sel) == 0: continue
    im, fg = render_trans(j); Pm = posterior_map(im, fg, 'trans')
    if Pm is None: continue
    rr = (ri_c[sel].float() + 0.5) / NR * E; pp = pi_c[sel].float() / NPHI * 2 * math.pi
    Pt[sel] = samp(Pm, (rr * torch.sin(pp) / E + 1) / 2 * (RES - 1), (rr * torch.cos(pp) / E + 1) / 2 * (RES - 1))
    if n % 32 == 0: print(f'  transverse {n}/{len(js)} [{time.time()-t0:.0f}s]', flush=True)
ks = list(range(0, K2, STEP_L)); kk = torch.tensor(ks, device=dv); pm = pi_c % K2; dk = (pm[:, None] - kk[None]).abs(); dk = torch.minimum(dk, K2 - dk); near_l = kk[torch.argmin(dk, 1)]
for n, k in enumerate(ks):
    sel = torch.nonzero(near_l == k).squeeze(1)
    if len(sel) == 0: continue
    im, fg = render_long(k); Pm = posterior_map(im, fg, 'long')
    if Pm is None: continue
    side = pi_c[sel] >= K2; col = torch.where(side, (NR - 1 - ri_c[sel]).float(), (NR + ri_c[sel]).float()) + 0.5
    Pl[sel] = samp(Pm, (zi_c[sel].float() + 0.5) / NZ * RES - 0.5, col / (2 * NR) * RES - 0.5)
    if n % 32 == 0: print(f'  longitudinal {n}/{len(ks)} [{time.time()-t0:.0f}s]', flush=True)
Pc = (Pt + Pl) / 2; labels = Pc.argmax(1).cpu().numpy(); dis = float((Pt.argmax(1) != Pl.argmax(1)).float().mean())
frac = np.bincount(labels, minlength=K) / M
print(f'asset labelled: {M:,} cells, class fractions {np.round(frac,3)}, transverse/longitudinal label disagreement {dis:.3f}  [{time.time()-t0:.0f}s]', flush=True)
np.savez_compressed(os.path.join(OUT, 'cells_photo.npz'), P=Pc.half().cpu().numpy(), Pt=Pt.half().cpu().numpy(), Pl=Pl.half().cpu().numpy(),
                    labels=labels.astype(np.int8), idx=core_idx.to(torch.int16).cpu().numpy(), shape=np.array([NR, NPHI, NZ]), K=K)
json.dump(dict(K=K, transfer=res, fractions=frac.tolist(), disagreement=dis), open(os.path.join(OUT, 'photo2asset.json'), 'w'), indent=1)

# ---- figure: photographs (spl + held-out) with learned labels, then the asset's three slices
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140], [90, 200, 120], [150, 100, 220], [120, 120, 120]], np.uint8)
L3 = np.full((NR, NPHI, NZ), -1, np.int16); L3[core_idx[:, 0].cpu(), core_idx[:, 1].cpu(), core_idx[:, 2].cpu()] = labels
An = A.cpu().numpy()
lin = np.linspace(-1, 1, RES); V, U = np.meshgrid(lin, lin, indexing='ij')
def look(rr, pp, zz, what):
    ri = np.clip((rr * NR - 0.5).round().astype(int), 0, NR - 1); pi_ = (np.round(pp / (2 * np.pi) * NPHI).astype(int)) % NPHI; zi = np.clip(((zz + 1) / 2 * NZ - 0.5).round().astype(int), 0, NZ - 1)
    if what == 'rgb': im = (An[:, ri, pi_, zi].transpose(1, 2, 0) + 1) / 2; im[rr > 1] = 1; return np.clip(im, 0, 1)
    o = L3[ri, pi_, zi].copy(); o[rr > 1] = -1; return cmap[np.clip(o + 1, 0, 7)]
x = U; y = V * math.cos(math.pi / 4); zc = V * math.sin(math.pi / 4)
planes = [(np.sqrt(U ** 2 + V ** 2), np.arctan2(V, U) % (2 * np.pi), np.zeros_like(U)), (np.abs(U), np.where(U >= 0, 0.0, np.pi) * np.ones_like(U), V), (np.sqrt(x ** 2 + zc ** 2), np.arctan2(zc, x) % (2 * np.pi), y)]
n_ph = len(photo_labels); fig, ax = plt.subplots(4, max(n_ph, 3), figsize=(2.0 * max(n_ph, 3), 8.4))
for j, (n, im, L, sp) in enumerate(photo_labels):
    ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {n.replace("or_","")}', fontsize=7); ax[1, j].imshow(cmap[np.clip(L + 1, 0, 7)])
for j in range(n_ph, ax.shape[1]): ax[0, j].axis('off'); ax[1, j].axis('off')
for j, (pl, ttl) in enumerate(zip(planes, ['asset transverse z=0', 'asset longitudinal', 'asset oblique 45 deg'])):
    ax[2, j].imshow(look(*pl, 'rgb')); ax[2, j].set_title(ttl, fontsize=8); ax[3, j].imshow(look(*pl, 'lab'))
for j in range(3, ax.shape[1]): ax[2, j].axis('off'); ax[3, j].axis('off')
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel('photograph'); ax[1, 0].set_ylabel(f'learned K={K}'); ax[2, 0].set_ylabel('3DFusion asset'); ax[3, 0].set_ylabel('labels by the\nphoto classifier')
plt.suptitle(f'material inventory learned on real cut photographs (spl), K={K} chosen by transfer to held-out oranges ({agree:.2f}); applied to the asset', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'photo2asset.png'), dpi=120); print('saved', os.path.join(OUT, 'photo2asset.png'), flush=True)
