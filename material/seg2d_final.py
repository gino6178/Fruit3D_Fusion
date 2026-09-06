"""Stage 1, final form. Three steps, one rule set for every object:
 1. bulk GMM, K = the largest K whose spl->held-out transfer is within TOL of the best (TOL=0.10),
    so a K that is anatomically finer is not rejected for a 0.06 drop;
 2. merge classes that are the same substance: agglomerative merge while the distance between two
    class centroids in feature space is below MTAU x the mean within-class spread (an orange's
    albedo and columella are one substance seen in two places; a peel and a flesh are not);
 3. sparse inclusions: dark outliers inside each merged class (L* below mean - Z sd), connected
    components with a size gate. Seeds are ~0.5% of pixels, so they can never be a GMM component.
usage: seg2d_final.py OBJDIR OUT_PREFIX"""
import os, sys, glob, json, numpy as np, torch, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *; from gmm_gpu import GaussianMixtureGPU
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
OBJDIR, OUTP = sys.argv[1], sys.argv[2]
KS = [2, 3, 4, 5, 6]; TOL = float(os.environ.get('TOL', 0.10)); MTAU = float(os.environ.get('MTAU', 1.6))
W_R = float(os.environ.get('W_R', 3.0)); Z = float(os.environ.get('Z', 2.5)); AMIN, AMAX = 8, 4000
rng = np.random.default_rng(0)
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
def sample(ph, n=40000, seed=0):
    """features and the matching LAB colours from the SAME pixels (they were drawn separately before,
    which is why the merge rule saw dE 1 between classes whose true dE is 13)"""
    r_ = np.random.default_rng(seed); X, L = [], []
    for _, im, fg, fam, _ in ph:
        f, rr = fi(im, fg, fam); ys, xs = np.nonzero(fg); sel = r_.choice(len(ys), min(n, len(ys)), replace=False)
        X.append(np.concatenate([f[ys[sel], xs[sel]], W_R * rr[ys[sel], xs[sel], None]], 1))
        L.append(cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[ys[sel], xs[sel]])
    return np.concatenate(X), np.concatenate(L)
Xs, LABs = sample(spl, seed=0); Xh = sample(hld, seed=1)[0] if hld else None
NSEED = int(os.environ.get('NSEED', 3)); sweep = {}
for K in KS:
    g = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=0).fit(Xs)
    if Xh is None: sweep[K] = (float('nan'), g); continue
    ag = []
    for s_ in range(NSEED):                       # K=4 on the orange swung 0.61..0.89 on one draw; average it
        Xa = sample(spl, seed=s_)[0]; Xb = sample(hld, seed=100 + s_)[0]
        ga = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=s_).fit(Xa)
        gb = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=50 + s_).fit(Xb)
        la, lb = ga.predict(Xb), gb.predict(Xb); M = np.zeros((K, K))
        for a, b in zip(la, lb): M[a, b] += 1
        r_, c_ = linear_sum_assignment(-M); ag.append(M[r_, c_].sum() / len(la))
    sweep[K] = (float(np.mean(ag)), g)
best_a = max(v[0] for v in sweep.values() if not np.isnan(v[0])) if Xh is not None else float('nan')
K = max([k for k, v in sweep.items() if np.isnan(v[0]) or v[0] >= best_a - TOL]) if Xh is not None else 3
agree, gmm = sweep[K]
print(f'transfer sweep {{k: round(v[0],3) for k,v in sweep.items()}}'.replace('{k: round(v[0],3) for k,v in sweep.items()}', str({k: round(v[0], 3) for k, v in sweep.items()})) + f' -> K={K} (best {best_a:.3f}, tol {TOL})', flush=True)
# --- merge same-substance classes by PERCEPTUAL colour distance (LAB dE), not feature-space distance:
# in the DINO+texture space an orange's flesh and its columella sit 0.34 spreads apart, which merges the
# wrong pair. Two classes are the same substance if their mean colour is within DE of each other.
DE = float(os.environ.get('DE', 20.0))
lab = gmm.predict(Xs)
cen = np.stack([LABs[lab == k].mean(0) if (lab == k).any() else np.full(3, 1e6) for k in range(K)])
groups = [[k] for k in range(K)]
while len(groups) > 2:
    best = None
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            ci = np.mean([cen[k] for k in groups[i]], 0); cj = np.mean([cen[k] for k in groups[j]], 0)
            d = float(np.linalg.norm(ci - cj))
            if best is None or d < best[0]: best = (d, i, j)
    if best[0] > DE: break
    d, i, j = best; print(f'  merge classes {groups[i]} + {groups[j]} (LAB dE {d:.0f} < {DE})', flush=True)
    groups = [g for t, g in enumerate(groups) if t not in (i, j)] + [groups[i] + groups[j]]
gmap = np.zeros(K, int)
for gi, grp in enumerate(groups):
    for k in grp: gmap[k] = gi
KM = len(groups)
# order merged classes by mean radius
r_of = np.array([Xs[np.isin(lab, grp), -1].mean() for grp in groups]); order = np.argsort(r_of); remap = np.zeros(KM, int); remap[order] = np.arange(KM)
final_map = remap[gmap]
print(f'  {K} bulk classes -> {KM} substances after merging', flush=True)
def labels_of(im, fg, fam):
    f, r = fi(im, fg, fam); X = np.concatenate([f, W_R * r[..., None]], -1); L = np.full(fg.shape, -1, int); L[fg] = final_map[gmm.predict(X[fg])]; return L
def inclusions(im, fg, L):
    """dark outliers inside each substance -> one extra class index KM"""
    Lst = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]; out = np.zeros(fg.shape, bool); info = []
    for k in range(KM):
        m = (L == k) & fg
        if m.sum() < 500: continue
        v = Lst[m]; cand = m & (Lst < v.mean() - Z * v.std())
        lb, n = ndimage.label(cand); sz = np.bincount(lb.ravel()); keep = np.zeros(n + 1, bool); keep[1:] = (sz[1:] >= AMIN) & (sz[1:] <= AMAX)
        sel = keep[lb]
        if sel.sum() and sel.sum() / m.sum() < 0.15: out |= sel; info.append(dict(host=k, n=int(keep[1:].sum()), frac_obj=round(float(sel.sum() / fg.sum()), 4), median_px=float(np.median(sz[1:][keep[1:]]))))
    return out, info
rows = []
for nm, im, fg, fam, sp in spl + hld:
    L = labels_of(im, fg, fam); inc, info = inclusions(im, fg, L); Lf = L.copy(); Lf[inc] = KM
    f = np.bincount(Lf[fg] + 1, minlength=KM + 2)[1:] / max(fg.sum(), 1)
    rows.append(dict(name=nm, split=sp, fam=fam, fractions=[round(float(x), 4) for x in f], inclusions=info))
for sp in ('spl', 'hld'):
    v = [r for r in rows if r['split'] == sp]
    if v: print(f'  {sp}: mean fractions {np.round(np.mean([r["fractions"] for r in v], 0), 3)} (last = inclusions), n={len(v)}', flush=True)
cmap = np.array([[255, 255, 255], [250, 220, 120], [230, 90, 40], [60, 140, 230], [90, 200, 120], [150, 100, 220], [25, 25, 25]], np.uint8)
show = spl[:3] + hld[:3]
fig, ax = plt.subplots(3, len(show), figsize=(2.05 * len(show), 6.5), squeeze=False)
for j, (nm, im, fg, fam, sp) in enumerate(show):
    L = labels_of(im, fg, fam); inc, info = inclusions(im, fg, L); Lf = L.copy(); Lf[inc] = KM
    vis = cmap[np.clip(Lf + 1, 0, 6)].copy(); vis[Lf == KM] = cmap[6]
    ov = im.copy()
    for k in range(KM + 1):
        e = cv2.Canny(((Lf == k) * 255).astype(np.uint8), 50, 150); ov[e > 0] = cmap[min(k + 1, 6)]
    ax[0, j].imshow(im); ax[0, j].set_title(f'{sp} {nm[-10:]}', fontsize=7); ax[1, j].imshow(vis); ax[2, j].imshow(ov)
    ax[2, j].set_xlabel(str([(d['host'], d['n']) for d in info]), fontsize=6)
    for i in range(3): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
for i, t in enumerate(['photograph', f'{KM} substances + inclusions', 'boundaries on photo']): ax[i, 0].set_ylabel(t, fontsize=9)
obj = os.path.basename(OBJDIR.rstrip('/'))
plt.suptitle(f'{obj}: bulk K={K} (transfer {agree:.2f}) -> {KM} substances after merging, plus sparse dark inclusions', fontsize=9)
plt.tight_layout(); plt.savefig(OUTP + '_final.png', dpi=115)
json.dump(dict(object=obj, K_bulk=K, transfer=float(agree), K_merged=KM, groups=[[int(x) for x in g] for g in groups], rows=rows), open(OUTP + '_final.json', 'w'), indent=1)
print('saved', OUTP + '_final.png', flush=True)
