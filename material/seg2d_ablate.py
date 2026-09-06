"""Two Stage-1 diagnostics the seg2d figures point at:
 (A) radius weight: does the albedo separate from the peel when r stops dominating?
 (B) sparse inclusions: seeds are ~1.7% of pixels, so a GMM never spends a component on them.
     Detect them separately, inside each bulk class, and require the same detection on held-out
     photographs (the darkest-centre standoff test from notes/discrete-inclusions.md).
usage: seg2d_ablate.py OBJDIR OUT.png"""
import os, sys, glob, json, numpy as np, torch, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *; from gmm_gpu import GaussianMixtureGPU
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
OBJDIR, OUT = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(0)
def load(split, fam):
    o = []
    for f in sorted(glob.glob(f'{OBJDIR}/{split}_{fam}/*.png')):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA); o.append((os.path.basename(f)[:-4], im, fg_mask(im), fam, split))
    return o
spl = load('spl', 'trans') + load('spl', 'long'); hld = load('hld', 'trans') + load('hld', 'long')
S = []
for _, im, fg, _, _ in spl:
    f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(6000, len(ys)), replace=False); S.append(f[ys[sel], xs[sel]].float().cpu())
S = torch.cat(S); mu = S.mean(0); _, _, Vt = torch.linalg.svd(S - mu, full_matrices=False); PCA = Vt[:64].T.contiguous().to(dv); mu = mu.to(dv)
CACHE = {}
def feat_img(im, fg, fam):
    k = id(im)
    if k not in CACHE: CACHE[k] = (all_feats(im, fg, PCA, mu), radius_map(fg, fam))
    return CACHE[k]
def sample(ph, w_r, n=40000):
    X = []
    for _, im, fg, fam, _ in ph:
        f, r = feat_img(im, fg, fam); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(n, len(ys)), replace=False)
        X.append(np.concatenate([f[ys[sel], xs[sel]], w_r * r[ys[sel], xs[sel], None]], 1))
    return np.concatenate(X)
def transfer(w_r, K):
    Xs, Xh = sample(spl, w_r), sample(hld, w_r) if hld else None
    gs = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=2, random_state=0).fit(Xs)
    if Xh is None: return float('nan'), gs, Xs
    gh = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=2, random_state=1).fit(Xh)
    la, lb = gs.predict(Xh), gh.predict(Xh); C = np.zeros((K, K))
    for i, j in zip(la, lb): C[i, j] += 1
    r_, c_ = linear_sum_assignment(-C); return float(C[r_, c_].sum() / len(la)), gs, Xs
# ---- (A) radius weight sweep
print('(A) radius weight vs transfer'); tabA = {}
for w in (0.0, 1.0, 3.0):
    tabA[w] = {K: round(transfer(w, K)[0], 3) for K in (2, 3, 4)}
    print(f'  w_r={w}: ' + '  '.join(f'K={K}:{v}' for K, v in tabA[w].items()), flush=True)
# ---- (B) sparse inclusions inside the bulk classes
def bulk_labels(w_r, K, gs, Xs):
    labs = gs.predict(Xs); order = np.argsort([Xs[labs == k, -1].mean() if (labs == k).any() else 9 for k in range(K)]); remap = np.zeros(K, int); remap[order] = np.arange(K)
    def apply(im, fg, fam):
        f, r = feat_img(im, fg, fam); X = np.concatenate([f, w_r * r[..., None]], -1); L = np.full(fg.shape, -1, int); L[fg] = remap[gs.predict(X[fg])]; return L
    return apply
W_R, K0 = 3.0, 2
agree0, gs0, Xs0 = transfer(W_R, K0); apply0 = bulk_labels(W_R, K0, gs0, Xs0)
print(f'\n(B) inclusions inside the bulk classes (bulk K={K0}, w_r={W_R}, transfer {agree0:.3f})')
def inclusions(im, fg, fam, L, k):
    """within bulk class k: 2-means on LAB; the smaller cluster is an inclusion if it is <20% of the
    class and its centre stands off by >2x the within-cluster spread (discrete-inclusions.md test)."""
    m = (L == k) & fg
    if m.sum() < 500: return None, 0.0, 0.0
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[m]
    km = KMeans(2, n_init=4, random_state=0).fit(lab[rng.choice(len(lab), min(20000, len(lab)), replace=False)])
    c = km.predict(lab); sizes = np.bincount(c, minlength=2); small = int(sizes.argmin()); frac = sizes[small] / sizes.sum()
    spread = np.mean([lab[c == i].std(0).mean() for i in (0, 1)]); standoff = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]) / max(spread, 1e-6))
    out = np.zeros(fg.shape, bool); idx = np.nonzero(m); out[idx[0][c == small], idx[1][c == small]] = True
    return out, float(frac), standoff
stats = {'spl': [], 'hld': []}
for nm, im, fg, fam, sp in spl + hld:
    L = apply0(im, fg, fam)
    for k in range(K0):
        inc, frac, st = inclusions(im, fg, fam, L, k)
        if inc is not None: stats[sp].append(dict(name=nm, cls=k, frac=round(frac, 3), standoff=round(st, 2)))
for sp in ('spl', 'hld'):
    for k in range(K0):
        v = [d for d in stats[sp] if d['cls'] == k]
        if v: print(f'  {sp} bulk class {k}: inclusion fraction {np.median([d["frac"] for d in v]):.3f}, standoff {np.median([d["standoff"] for d in v]):.2f}  (n={len(v)})')
# ---- figure: bulk labels + accepted inclusions on a few photographs
show = spl[:3] + hld[:3]; cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140]], np.uint8)
fig, ax = plt.subplots(3, len(show), figsize=(2.05 * len(show), 6.4), squeeze=False)
for j, (nm, im, fg, fam, sp) in enumerate(show):
    L = apply0(im, fg, fam); vis = cmap[np.clip(L + 1, 0, K0)].copy(); acc = []
    for k in range(K0):
        inc, frac, st = inclusions(im, fg, fam, L, k)
        if inc is not None and frac < 0.20 and st > 2.0: vis[inc] = np.array([30, 30, 30], np.uint8); acc.append((k, round(frac, 3), round(st, 2)))
    ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {nm[-10:]}', fontsize=7)
    ax[1, j].imshow(cmap[np.clip(L + 1, 0, K0)]); ax[2, j].imshow(vis); ax[2, j].set_xlabel(str(acc), fontsize=6)
    for i in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
for i, t in enumerate(['photograph', f'bulk K={K0}', 'bulk + inclusions (black)']): ax[i, 0].set_ylabel(t, fontsize=9)
plt.suptitle(f'{os.path.basename(OBJDIR.rstrip("/"))}: (A) radius-weight sweep {tabA}   (B) inclusions inside bulk classes', fontsize=8)
plt.tight_layout(); plt.savefig(OUT, dpi=115); print('saved', OUT)
json.dump(dict(radius_sweep={str(k): v for k, v in tabA.items()}, inclusion_stats=stats), open(OUT.replace('.png', '.json'), 'w'), indent=1)
