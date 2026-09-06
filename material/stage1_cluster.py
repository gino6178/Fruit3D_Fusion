"""Cluster the back-projected cell features; report a stability-based K; paint labels on a
transverse, a longitudinal, and an unsupervised 45-degree oblique slice of the cylinder."""
import os, sys, math, numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
CELLS, STATE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
d = np.load(CELLS); Fall = d['F'].astype(np.float32); geo = d['geo'].astype(np.float32); idx = d['idx'].astype(np.int64); names = d['names']
NR, NPHI, NZ = [int(x) for x in d['shape']]; K2 = NPHI // 2
A = torch.load(STATE, map_location='cpu')['A'].float().numpy()               # (3,NR,NPHI,NZ) in [-1,1]
rng = np.random.default_rng(0)
SETS = {'lab': ['lab'], 'lab+tex': ['lab', 'tex'], 'all': ['dino', 'lab', 'tex']}
def feats(setname, with_geo):
    cols = np.isin(names, SETS[setname]); X = Fall[:, cols]
    if with_geo: X = np.concatenate([X, geo * 3.0], 1)                       # r, |z| up-weighted: the anatomical prior
    mu, sd = X.mean(0), X.std(0) + 1e-6; return (X - mu) / sd

def stability(X, K, half):
    """fit on half A, fit on half B; agreement of the two labelings on half B after Hungarian matching"""
    a, b = X[half], X[~half]
    sa = rng.choice(len(a), min(200000, len(a)), replace=False); sb = rng.choice(len(b), min(200000, len(b)), replace=False)
    ga = GaussianMixture(K, covariance_type='full', random_state=0, reg_covar=1e-3).fit(a[sa])
    gb = GaussianMixture(K, covariance_type='full', random_state=1, reg_covar=1e-3).fit(b[sb])
    la, lb = ga.predict(b[sb]), gb.predict(b[sb])
    C = np.zeros((K, K));
    for i, j in zip(la, lb): C[i, j] += 1
    r_, c_ = linear_sum_assignment(-C); return C[r_, c_].sum() / len(la), ga

def paint(labels_full, K):
    """labels over ALL core cells -> three slice images (trans mid-z, long phi=0, oblique 45deg)"""
    L = np.full((NR, NPHI, NZ), -1, np.int16); L[idx[:, 0], idx[:, 1], idx[:, 2]] = labels_full
    R = 512; lin = np.linspace(-1, 1, R); V, U = np.meshgrid(lin, lin, indexing='ij')
    def lookup(rr, pp, zz):  # normalised r in [0,1], phi rad, z in [-1,1]
        ri = np.clip((rr * NR - 0.5).round().astype(int), 0, NR - 1); pi_ = (np.round(pp / (2 * np.pi) * NPHI).astype(int)) % NPHI
        zi = np.clip(((zz + 1) / 2 * NZ - 0.5).round().astype(int), 0, NZ - 1)
        out = L[ri, pi_, zi]; out[rr > 1] = -1; return out
    trans = lookup(np.sqrt(U ** 2 + V ** 2), np.arctan2(V, U) % (2 * np.pi), np.zeros_like(U))                 # z = 0 (mid)
    longi = lookup(np.abs(U), np.where(U >= 0, 0.0, np.pi) * np.ones_like(U), V)                              # plane phi=0
    # oblique: plane through the axis tilted 45deg: world = u*e_x + v*(cos45 e_axis + sin45 e_z)
    x = U; y = V * math.cos(math.pi / 4); zc = V * math.sin(math.pi / 4)
    obl = lookup(np.sqrt(x ** 2 + zc ** 2), np.arctan2(zc, x) % (2 * np.pi), y)
    return trans, longi, obl

def rgb_slices():
    R = 512; lin = np.linspace(-1, 1, R); V, U = np.meshgrid(lin, lin, indexing='ij')
    def lookup(rr, pp, zz):
        ri = np.clip((rr * NR - 0.5).round().astype(int), 0, NR - 1); pi_ = (np.round(pp / (2 * np.pi) * NPHI).astype(int)) % NPHI
        zi = np.clip(((zz + 1) / 2 * NZ - 0.5).round().astype(int), 0, NZ - 1)
        im = (A[:, ri, pi_, zi].transpose(1, 2, 0) + 1) / 2; im[rr > 1] = 1; return np.clip(im, 0, 1)
    x = U; y = V * math.cos(math.pi / 4); zc = V * math.sin(math.pi / 4)
    return (lookup(np.sqrt(U ** 2 + V ** 2), np.arctan2(V, U) % (2 * np.pi), np.zeros_like(U)),
            lookup(np.abs(U), np.where(U >= 0, 0.0, np.pi) * np.ones_like(U), V),
            lookup(np.sqrt(x ** 2 + zc ** 2), np.arctan2(zc, x) % (2 * np.pi), y))

cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140], [90, 200, 120], [150, 100, 220], [120, 120, 120]], np.uint8)
half = (idx[:, 2] // 2) % 2 == 0                                                # alternating transverse slices
rows = [('RGB', None)]; results = {}
configs = [('lab', 3, False), ('lab', 4, False), ('lab+geo', 4, True), ('lab+tex+geo', 4, True), ('all+geo', 4, True), ('all+geo', 5, True)]
for setname, K, wg in configs:
    base = setname.replace('+geo', ''); X = feats(base, wg)
    agree, g = stability(X, K, half)
    sub = rng.choice(len(X), min(300000, len(X)), replace=False); bic = g.bic(X[sub])
    lab = g.predict(X)
    # order clusters by mean radius so colours are comparable across rows
    order = np.argsort([geo[lab == k, 0].mean() if (lab == k).any() else 9 for k in range(K)]); remap = np.zeros(K, int); remap[order] = np.arange(K); lab = remap[lab]
    frac = np.bincount(lab, minlength=K) / len(lab)
    results[f'{setname} K={K}'] = dict(agree=float(agree), bic=float(bic), frac=frac.round(3).tolist())
    print(f'{setname:14s} K={K}  half-split agreement {agree:.3f}  BIC {bic:12.0f}  fractions {np.round(frac,3)}', flush=True)
    rows.append((f'{setname} K={K}\nagree {agree:.2f}', lab))
    np.save(os.path.join(OUT, f'labels_{setname}_K{K}.npy'), lab.astype(np.int8))

fig, ax = plt.subplots(len(rows), 3, figsize=(9.6, 3.1 * len(rows)))
rgb = rgb_slices()
for i, (title, lab) in enumerate(rows):
    ims = rgb if lab is None else [cmap[np.clip(t + 1, 0, 7)] for t in paint(lab, lab.max() + 1)]
    for j, (im, cn) in enumerate(zip(ims, ['transverse z=0', 'longitudinal phi=0', 'oblique 45 deg (unsupervised)'])):
        ax[i, j].imshow(im); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title(cn, fontsize=9)
    ax[i, 0].set_ylabel(title, fontsize=8)
plt.suptitle('orange 3DFusion asset: cell clustering from back-projected slice features', fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'clusters.png'), dpi=120); print('saved', os.path.join(OUT, 'clusters.png'))
import json; json.dump(results, open(os.path.join(OUT, 'cluster_results.json'), 'w'), indent=1)
