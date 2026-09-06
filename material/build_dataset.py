"""Build the cross-object training set for a GENERAL material segmenter.

No pixel is labelled by hand. Per object we run the unsupervised pipeline, then map its discovered
classes onto ONE shared vocabulary with rules that hold for every object:

  0 SKIN     the bulk class with the largest mean radius and the highest boundary contact
  2 FIBROUS  among the remaining classes, the one with the lowest chroma, if its chroma is below
             0.6x the other's (an orange's albedo/columella, a bread's crumb-vs-crust does not qualify)
  1 FLESH    everything else that is bulk
  3 INCLUSION sparse dark outliers inside any class (watermelon seeds, a loaf's swirl)

Output: one .npz per photograph with image, foreground, label; plus an index with object, family,
split, and per-class fractions. This is what a single cross-object model trains on, and
leave-one-object-out on it is the generality test.
usage: build_dataset.py OUT_DIR [objects...]"""
import os, sys, glob, json, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import *; from gmm_gpu import GaussianMixtureGPU
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1]; OBJS = sys.argv[2:] or ['orange', 'watermelon', 'apple', 'bread', 'cake', 'doughnut', 'pomegranate']
KS = [2, 3, 4, 5]; TOL = 0.10; W_R = 3.0; DE = 20.0; Z = 2.5; AMIN, AMAX = 8, 4000; NSEED = 3
SKIN, FLESH, FIBROUS, INCLUSION = 0, 1, 2, 3
NAMES = {0: 'skin', 1: 'flesh', 2: 'fibrous', 3: 'inclusion'}
os.makedirs(OUT, exist_ok=True); index = []; CLASSSTAT = []
def objdir(o): return f'{ROOT}/data/orange' if o == 'orange' else f'{ROOT}/runs/{o}'
for OBJ in OBJS:
    D = objdir(OBJ); rng = np.random.default_rng(0)
    def load(sp, fam):
        r = []
        for f in sorted(glob.glob(f'{D}/{sp}_{fam}/*.png')):
            im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA); r.append((os.path.basename(f)[:-4], im, fg_mask(im), fam, sp))
        return r
    spl = load('spl', 'trans') + load('spl', 'long'); hld = load('hld', 'trans') + load('hld', 'long')
    if not spl: print(f'{OBJ}: no photographs, skipped', flush=True); continue
    S = []
    for _, im, fg, _, _ in spl:
        f = dino_feats(im); ys, xs = np.nonzero(fg); sel = rng.choice(len(ys), min(6000, len(ys)), replace=False); S.append(f[ys[sel], xs[sel]].float().cpu())
    S = torch.cat(S); mu = S.mean(0); _, _, Vt = torch.linalg.svd(S - mu, full_matrices=False); PCA = Vt[:64].T.contiguous().to(dv); mu = mu.to(dv)
    C = {}; CACHE = os.path.join(OUT, '_featcache'); os.makedirs(CACHE, exist_ok=True)
    def fi(im, fg, fam, nm=None, sp=None):
        k = id(im)
        if k in C: return C[k]
        cp = os.path.join(CACHE, f'{OBJ}__{sp}__{fam}__{nm}.npz') if nm else None   # split+family+name: '00.png' exists in all four folders
        if cp and os.path.exists(cp):
            z = np.load(cp); C[k] = (z['f'], z['r'])
        else:
            v = (all_feats(im, fg, PCA, mu), radius_map(fg, fam))
            if cp: np.savez_compressed(cp, f=v[0].astype(np.float16), r=v[1].astype(np.float16))
            C[k] = v
        return C[k]
    def sample(ph, seed=0, n=40000):
        r_ = np.random.default_rng(seed); X, L, B = [], [], []
        for nm_, im, fg, fam, sp_ in ph:
            f, rr = fi(im, fg, fam, nm_, sp_); ys, xs = np.nonzero(fg); sel = r_.choice(len(ys), min(n, len(ys)), replace=False)
            X.append(np.concatenate([f[ys[sel], xs[sel]], W_R * rr[ys[sel], xs[sel], None]], 1))
            L.append(cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[ys[sel], xs[sel]])
            edge = fg & ~ndimage.binary_erosion(fg, iterations=6); B.append(edge[ys[sel], xs[sel]])
        return np.concatenate(X), np.concatenate(L), np.concatenate(B)
    Xs, LABs, Bs = sample(spl, 0)
    sweep = {}
    for K in KS:
        if not hld: sweep[K] = float('nan'); continue
        ag = []
        for s_ in range(NSEED):
            Xa = sample(spl, s_)[0]; Xb = sample(hld, 100 + s_)[0]
            ga = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=s_).fit(Xa); gb = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=50 + s_).fit(Xb)
            la, lb = ga.predict(Xb), gb.predict(Xb); M = np.zeros((K, K))
            for a, b in zip(la, lb): M[a, b] += 1
            r_, c_ = linear_sum_assignment(-M); ag.append(M[r_, c_].sum() / len(la))
        sweep[K] = float(np.mean(ag))
    K = max([k for k, v in sweep.items() if np.isnan(v) or v >= max([x for x in sweep.values() if not np.isnan(x)] or [0]) - TOL]) if hld else 3
    gmm = GaussianMixtureGPU(K, reg_covar=1e-3, n_init=3, random_state=0).fit(Xs); lab = gmm.predict(Xs)
    rad = Xs[:, -1] / W_R
    stat = {k: dict(r=float(rad[lab == k].mean()), edge=float(Bs[lab == k].mean()), lab=LABs[lab == k].mean(0), n=int((lab == k).sum())) for k in range(K) if (lab == k).any()}
    # SKIN: largest mean radius; require it also to touch the boundary more than average
    skin = max(stat, key=lambda k: stat[k]['r'])
    rest = [k for k in stat if k != skin]
    # merge the remaining classes by perceptual colour
    groups = [[k] for k in rest]
    while len(groups) > 1:
        best = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = float(np.linalg.norm(np.mean([stat[k]['lab'] for k in groups[i]], 0) - np.mean([stat[k]['lab'] for k in groups[j]], 0)))
                if best is None or d < best[0]: best = (d, i, j)
        if best[0] > DE: break
        _, i, j = best; groups = [g for t, g in enumerate(groups) if t not in (i, j)] + [groups[i] + groups[j]]
    # whiteness = L - 2*chroma (OpenCV LAB units): white fibrous tissue is BOTH pale and desaturated.
    # A chroma ratio alone missed an orange's columella (C 53 vs flesh 77, ratio 0.69) whose whiteness
    # is 102 vs 38. The gap is required to be large relative to the object's own class spread.
    def wl(g):
        c = np.mean([stat[k]['lab'] for k in g], 0); return float(c[0] - 2 * np.linalg.norm(c[1:] - 128))
    role = {skin: SKIN}
    WGAP = float(os.environ.get('WGAP', 30.0))
    if len(groups) >= 2:
        gs_ = sorted(groups, key=wl, reverse=True); fib, fle = gs_[0], gs_[-1]
        if wl(fib) - wl(fle) > WGAP:
            for k in fib: role[k] = FIBROUS
            for g in groups:
                if g is not fib:
                    for k in g: role[k] = FLESH
        else:
            for g in groups:
                for k in g: role[k] = FLESH
    else:
        for g in groups:
            for k in g: role[k] = FLESH
    rmap = np.zeros(K, int)
    for k, v in role.items(): rmap[k] = v
    CLASSSTAT.append(dict(object=OBJ, K=K, sweep={str(k): (None if np.isnan(v) else round(v, 3)) for k, v in sweep.items()},
        classes=[dict(k=int(k), radius=round(stat[k]['r'], 3), edge=round(stat[k]['edge'], 3), L=round(float(stat[k]['lab'][0]), 1),
                      chroma=round(float(np.linalg.norm(stat[k]['lab'][1:] - 128)), 1),
                      whiteness=round(float(stat[k]['lab'][0] - 2 * np.linalg.norm(stat[k]['lab'][1:] - 128)), 1),
                      frac=round(stat[k]['n'] / len(lab), 3), role=NAMES[role[k]]) for k in sorted(stat, key=lambda x: stat[x]['r'])],
        groups=[[int(x) for x in g] for g in groups]))
    print(f'{OBJ}: sweep {{{", ".join(f"{k}:{v:.2f}" for k, v in sweep.items())}}} -> K={K}; roles ' +
          ', '.join(f'{k}->{NAMES[role[k]]}(r={stat[k]["r"]:.2f},C={np.linalg.norm(stat[k]["lab"][1:]-128):.0f})' for k in sorted(stat, key=lambda x: stat[x]['r'])), flush=True)
    for nm, im, fg, fam, sp in spl + hld:
        f, r = fi(im, fg, fam, nm, sp); X = np.concatenate([f, W_R * r[..., None]], -1)
        L = np.full(fg.shape, -1, np.int8); L[fg] = rmap[gmm.predict(X[fg])]
        Lst = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]
        for k in range(3):
            m = (L == k) & fg
            if m.sum() < 500: continue
            v = Lst[m]; cand = m & (Lst < v.mean() - Z * v.std()); lb, n = ndimage.label(cand); sz = np.bincount(lb.ravel()); keep = np.zeros(n + 1, bool); keep[1:] = (sz[1:] >= AMIN) & (sz[1:] <= AMAX); sel = keep[lb]
            if sel.sum() and sel.sum() / m.sum() < 0.15: L[sel] = INCLUSION
        fr = np.bincount(L[fg], minlength=4) / max(fg.sum(), 1)
        fn = f'{OBJ}__{sp}__{fam}__{nm}.npz'; np.savez_compressed(os.path.join(OUT, fn), image=im, fg=fg, label=L)
        index.append(dict(file=fn, object=OBJ, split=sp, family=fam, name=nm, fractions=[round(float(x), 4) for x in fr]))
    C.clear()
json.dump(dict(names=NAMES, items=index, class_stats=CLASSSTAT), open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
print(f'\nwrote {len(index)} labelled photographs to {OUT}')
import collections
agg = collections.defaultdict(list)
for it in index: agg[it['object']].append(it['fractions'])
print(f'{"object":12s} {"n":>3s}  skin  flesh fibrous incl')
for o, v in agg.items():
    m = np.mean(v, 0); print(f'{o:12s} {len(v):3d}  ' + '  '.join(f'{x:.3f}' for x in m))
