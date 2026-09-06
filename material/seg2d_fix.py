"""Two fixes the seg2d figures asked for.
(1) Sparse inclusions (watermelon seeds): NOT a GMM component and NOT 2-means (which bisects any
    class). A dark-outlier detector inside the bulk class + connected components with a size gate,
    checked for consistency on held-out photographs.
(2) Orange anatomy: sweep radius weight x K and paint them side by side, so peel / albedo / flesh /
    columella can be judged directly instead of through a transfer number.
usage: seg2d_fix.py OBJDIR OUT_PREFIX [mode]   mode = inclusions | anatomy"""
import os, sys, glob, json, numpy as np, torch, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *; from gmm_gpu import GaussianMixtureGPU
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
OBJDIR, OUTP = sys.argv[1], sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else 'inclusions'; rng = np.random.default_rng(0)
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
C = {}
def fi(im, fg, fam):
    k = id(im)
    if k not in C: C[k] = (all_feats(im, fg, PCA, mu), radius_map(fg, fam))
    return C[k]
def fit(w_r, K, ph):
    X = []
    for _, im, fg, fam, _ in ph:
        f, r = fi(im, fg, fam); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(40000, len(ys)), replace=False)
        X.append(np.concatenate([f[ys[sel], xs[sel]], w_r * r[ys[sel], xs[sel], None]], 1))
    X = np.concatenate(X); g = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=2, random_state=0).fit(X)
    lab = g.predict(X); order = np.argsort([X[lab == k, -1].mean() if (lab == k).any() else 9 for k in range(K)]); remap = np.zeros(K, int); remap[order] = np.arange(K)
    return g, remap, X
def apply(g, remap, w_r, im, fg, fam):
    f, r = fi(im, fg, fam); X = np.concatenate([f, w_r * r[..., None]], -1); L = np.full(fg.shape, -1, int); L[fg] = remap[g.predict(X[fg])]; return L
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140], [90, 200, 120]], np.uint8)

if MODE == 'anatomy':
    combos = [(0.0, 3), (1.0, 3), (3.0, 3), (1.0, 4), (3.0, 4), (1.0, 5)]
    show = spl[:2] + [p for p in spl if p[3] == 'long'][:1] + hld[:2]
    fig, ax = plt.subplots(len(combos) + 1, len(show), figsize=(2.05 * len(show), 2.0 * (len(combos) + 1)), squeeze=False)
    for j, (nm, im, fg, fam, sp) in enumerate(show):
        ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {nm[-10:]}', fontsize=7)
    for i, (w, K) in enumerate(combos):
        g, remap, Xs = fit(w, K, spl)
        agree = float('nan')
        if hld:
            gh, _, Xh = fit(w, K, hld); la, lb = g.predict(Xh), gh.predict(Xh); M = np.zeros((K, K))
            for a, b in zip(la, lb): M[a, b] += 1
            r_, c_ = linear_sum_assignment(-M); agree = M[r_, c_].sum() / len(la)
        for j, (nm, im, fg, fam, sp) in enumerate(show):
            L = apply(g, remap, w, im, fg, fam); ax[i + 1, j].imshow(cmap[np.clip(L + 1, 0, K)])
        ax[i + 1, 0].set_ylabel(f'w_r={w} K={K}\ntransfer {agree:.2f}', fontsize=8)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel('photograph', fontsize=9)
    plt.suptitle(f'{os.path.basename(OBJDIR.rstrip("/"))}: does any (radius weight, K) give peel / albedo / flesh / columella?', fontsize=9)
    plt.tight_layout(); plt.savefig(OUTP + '_anatomy.png', dpi=115); print('saved', OUTP + '_anatomy.png')
else:
    W, K0 = 3.0, 2; g, remap, _ = fit(W, K0, spl)
    def detect(im, fg, L, k, z=2.5, amin=8, amax=4000):
        """dark outliers inside bulk class k, then connected components with a size gate"""
        m = (L == k) & fg
        if m.sum() < 500: return np.zeros_like(fg), {}
        Lst = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]
        v = Lst[m]; thr = v.mean() - z * v.std(); cand = m & (Lst < thr)
        lab_, n = ndimage.label(cand); sizes = np.bincount(lab_.ravel()); keep = np.zeros(n + 1, bool)
        keep[1:] = (sizes[1:] >= amin) & (sizes[1:] <= amax); out = keep[lab_]
        kept = int(keep[1:].sum()); areas = sizes[1:][keep[1:]]
        return out, dict(frac_of_class=float(out.sum() / m.sum()), frac_of_object=float(out.sum() / fg.sum()), n_pieces=kept,
                         median_area=float(np.median(areas)) if kept else 0.0, thr_L=float(thr), mean_L=float(v.mean()))
    rows = {'spl': [], 'hld': []}
    for nm, im, fg, fam, sp in spl + hld:
        L = apply(g, remap, W, im, fg, fam)
        for k in range(K0):
            _, st = detect(im, fg, L, k)
            if st: st.update(name=nm, cls=k, split=sp, fam=fam); rows[sp].append(st)
    print(f'dark-outlier inclusions inside each bulk class (z=2.5, area 8..4000 px)')
    for sp in ('spl', 'hld'):
        for k in range(K0):
            v = [d for d in rows[sp] if d['cls'] == k]
            if not v: continue
            print(f'  {sp} class {k}: frac of object {np.median([d["frac_of_object"] for d in v]):.4f}  pieces {np.median([d["n_pieces"] for d in v]):.0f}  median piece {np.median([d["median_area"] for d in v]):.0f} px  (n={len(v)})')
    show = spl[:3] + hld[:3]
    fig, ax = plt.subplots(3, len(show), figsize=(2.05 * len(show), 6.4), squeeze=False)
    for j, (nm, im, fg, fam, sp) in enumerate(show):
        L = apply(g, remap, W, im, fg, fam); vis = cmap[np.clip(L + 1, 0, K0)].copy(); info = []
        for k in range(K0):
            inc, st = detect(im, fg, L, k)
            if st and st['n_pieces'] > 0 and st['frac_of_class'] < 0.15: vis[inc] = np.array([20, 20, 20], np.uint8); info.append((k, st['n_pieces'], round(st['frac_of_object'], 4)))
        ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {nm[-10:]}', fontsize=7); ax[1, j].imshow(cmap[np.clip(L + 1, 0, K0)]); ax[2, j].imshow(vis); ax[2, j].set_xlabel(str(info), fontsize=6)
        for i in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for i, t in enumerate(['photograph', f'bulk K={K0}', 'bulk + dark inclusions']): ax[i, 0].set_ylabel(t, fontsize=9)
    plt.suptitle(f'{os.path.basename(OBJDIR.rstrip("/"))}: sparse inclusions detected inside the bulk classes, not as a GMM component', fontsize=9)
    plt.tight_layout(); plt.savefig(OUTP + '_incl.png', dpi=115); print('saved', OUTP + '_incl.png')
    json.dump(rows, open(OUTP + '_incl.json', 'w'), indent=1)
