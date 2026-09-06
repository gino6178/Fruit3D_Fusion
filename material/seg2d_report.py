"""Look at Stage 1 on its own: the 2D material segmentation on the photographs.
For each object: fit the classifier on spl exactly as stage1_photo2asset.py does, then show
photo | label | boundary overlay for every spl and held-out photograph, with per-class pixel
fractions.  This is the figure to judge Stage 1 by, without any 3D in the way."""
import os, sys, glob, json, math, numpy as np, torch, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *; from gmm_gpu import GaussianMixtureGPU
from scipy.optimize import linear_sum_assignment
OBJDIR, OUT = sys.argv[1], sys.argv[2]; KS = [2, 3, 4, 5, 6]; rng = np.random.default_rng(0)
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
def load(split, fam):
    out = []
    for f in sorted(glob.glob(f'{OBJDIR}/{split}_{fam}/*.png')):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
        out.append((os.path.basename(f)[:-4], im, fg_mask(im), fam, split))
    return out
spl = load('spl', 'trans') + load('spl', 'long'); hld = load('hld', 'trans') + load('hld', 'long')
if not spl: print(json.dumps({'object': os.path.basename(OBJDIR), 'error': 'no spl photographs'})); sys.exit()
S = []
for _, im, fg, _, _ in spl:
    f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(6000, len(ys)), replace=False); S.append(f[ys[sel], xs[sel]].float().cpu())
S = torch.cat(S); mu = S.mean(0); _, _, Vt = torch.linalg.svd(S - mu, full_matrices=False); PCA = Vt[:64].T.contiguous().to(dv); mu = mu.to(dv)
def feats_of(ph, n=40000):
    X = []
    for _, im, fg, fam, _ in ph:
        f = all_feats(im, fg, PCA, mu); r = radius_map(fg, fam); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(n, len(ys)), replace=False)
        X.append(np.concatenate([f[ys[sel], xs[sel]], 3.0 * r[ys[sel], xs[sel], None]], 1))
    return np.concatenate(X)
Xs = feats_of(spl); Xh = feats_of(hld) if hld else None
best = None; sweep = {}
for K in KS:
    gs = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=2, random_state=0).fit(Xs)
    if Xh is not None:
        gh = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=2, random_state=1).fit(Xh)
        la, lb = gs.predict(Xh), gh.predict(Xh); C = np.zeros((K, K))
        for i, j in zip(la, lb): C[i, j] += 1
        r_, c_ = linear_sum_assignment(-C); agree = float(C[r_, c_].sum() / len(la))
    else: agree = float('nan')
    sweep[K] = round(agree, 3)
    if best is None or agree > best[1] + 0.02 or (agree > best[1] - 0.02 and K > best[0]): best = (K, agree, gs)
K, agree, gmm = best
labs = gmm.predict(Xs); order = np.argsort([Xs[labs == k, -1].mean() if (labs == k).any() else 9 for k in range(K)]); remap = np.zeros(K, int); remap[order] = np.arange(K)
def label_of(im, fg, fam):
    f = all_feats(im, fg, PCA, mu); r = radius_map(fg, fam); X = np.concatenate([f, 3.0 * r[..., None]], -1)
    L = np.full(fg.shape, -1, int); L[fg] = remap[gmm.predict(X[fg])]; return L
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140], [90, 200, 120], [150, 100, 220]], np.uint8)
MAXH = int(os.environ.get('MAXH', 6)); allph = spl + hld[:MAXH]; fracs = np.zeros(K)
fig, ax = plt.subplots(3, len(allph), figsize=(2.05 * len(allph), 6.5), squeeze=False)
for j, (nm, im, fg, fam, sp) in enumerate(allph):
    L = label_of(im, fg, fam); f = np.bincount(L[fg], minlength=K) / max(fg.sum(), 1)
    if sp == 'spl': fracs += f / max(len(spl), 1)
    ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {nm[-12:]}\n{fam}', fontsize=7)
    ax[1, j].imshow(cmap[np.clip(L + 1, 0, K)])
    ov = im.copy()
    for k in range(K):
        e = cv2.Canny(((L == k) * 255).astype(np.uint8), 50, 150); ov[e > 0] = cmap[k + 1]
    ax[2, j].imshow(ov); ax[2, j].set_xlabel(' '.join(f'{x:.2f}' for x in f), fontsize=6)
    for i in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
for i, t in enumerate(['photograph', f'K={K} labels', 'boundaries on photo']): ax[i, 0].set_ylabel(t, fontsize=9)
obj = os.path.basename(OBJDIR.rstrip('/'))
plt.suptitle(f'{obj}: Stage 1 material segmentation. K={K} by spl->held-out transfer (sweep {sweep}); spl class fractions {np.round(fracs,3)}', fontsize=9)
plt.tight_layout(); plt.savefig(OUT, dpi=115); print(json.dumps({'object': obj, 'K': K, 'transfer': sweep, 'spl_fractions': [round(float(x), 3) for x in fracs], 'n_spl': len(spl), 'n_hld': len(hld)}))
