"""Stage 4, simulation side: predicted cut-force profile shape for four material fields.
A blade is a straight cutting edge lying in the cut plane and advancing along d; at advance s the edge
front is the set of voxels of the plane at coordinate s, and
    F(s) = sum_{voxels on the front} tau(k(x)) * dl
tau is RELATIVE (peel 1.0, fibrous 0.5, flesh 0.15): only the SHAPE of the curve is claimed.
Fields: NMFS, GMM+MRF, homogeneous (tau const -> F is the chord length), and the GaussianFluent rule
re-implemented automatically (K-means on the voxel colour, classes ordered by radius)."""
import os, sys, math, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')); from planes import Vol
NMFS, MRF, STATE, GRID, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True); dv = 'cuda'
# class roles by mean radius (same rule as stage2): peel = outermost, flesh = largest remaining, fibre = the rest
_d = np.load(NMFS); _idx = _d['idx'].astype(int); _lab = _d['labels'].astype(int); _NR = int(_d['shape'][0]); _K = int(_lab.max()) + 1
_mr = np.array([((_idx[_lab == k, 0] + 0.5) / _NR).mean() if (_lab == k).any() else -1 for k in range(_K)]); PEEL = int(_mr.argmax())
_rest = [k for k in range(_K) if k != PEEL]; FLESH = max(_rest, key=lambda k: (_lab == k).sum()); FIBRE = [k for k in _rest if k != FLESH]
TAU = {PEEL: 1.0, FLESH: 0.15}; TAU.update({k: 0.5 for k in FIBRE}); print('class roles:', dict(peel=PEEL, flesh=FLESH, fibre=FIBRE), 'tau', TAU, flush=True)
vol = Vol(GRID, dv, res=256); E = vol.EXT; N = vol.N; X = torch.load(STATE, map_location=dv)['X'].float()
occ = vol.OCC[0] > 0.5; shell = vol.SHELL[0] > 0.5
rad = torch.sqrt(vol.u0 ** 2 + vol.u1 ** 2); ang = torch.atan2(vol.u1, vol.u0) % (2 * math.pi)

def cyl_to_cart(npz):
    d = np.load(npz); NR, NPHI, NZ = [int(x) for x in d['shape']]; idx = torch.from_numpy(d['idx'].astype(np.int64)).to(dv); lab = torch.from_numpy(d['labels'].astype(np.int64)).to(dv)
    L = torch.full((NR, NPHI, NZ), -1, dtype=torch.int64, device=dv); L[idx[:, 0], idx[:, 1], idx[:, 2]] = lab
    ri = (rad / E * NR - 0.5).round().long().clamp(0, NR - 1); pi_ = (ang / (2 * math.pi) * NPHI).round().long() % NPHI; zi = ((vol.v + E) / (2 * E) * NZ - 0.5).round().long().clamp(0, NZ - 1)
    Lc = L[ri, pi_, zi]; Lc[~occ] = -1; Lc[shell] = PEEL
    # occupied voxels the cylinder did not cover (outside its core) : nearest labelled neighbour by simple dilation
    for _ in range(4):
        miss = occ & (Lc < 0)
        if not miss.any(): break
        for sh in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            nb = torch.roll(Lc, sh, (0, 1, 2)); fill = miss & (nb >= 0); Lc[fill] = nb[fill]; miss = occ & (Lc < 0)
    Lc[occ & (Lc < 0)] = FLESH
    return Lc
fields = {'NMFS': cyl_to_cart(NMFS), 'GMM+MRF': cyl_to_cart(MRF)}
fields['homogeneous'] = torch.where(occ, torch.full_like(fields['NMFS'], FLESH), torch.full_like(fields['NMFS'], -1))
# GaussianFluent rule, automated: K-means (K=3) on LAB colour of occupied voxels, classes ordered by mean radius
import cv2
rgb = ((X.permute(1, 2, 3, 0)[occ] + 1) / 2).clamp(0, 1).cpu().numpy(); lab = cv2.cvtColor((rgb[None] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
from sklearn.cluster import KMeans
km = KMeans(max(_K, 2), n_init=4, random_state=0).fit(lab[np.random.default_rng(0).choice(len(lab), 200000, replace=False)]); cl = torch.from_numpy(km.predict(lab)).to(dv)
rv = rad[occ]; order = np.argsort([float(rv[cl == k].mean()) for k in range(max(_K, 2))]); remap = torch.zeros(max(_K, 2), dtype=torch.long, device=dv); remap[torch.from_numpy(order).to(dv)] = torch.arange(max(_K, 2), device=dv)
Lcol = torch.full_like(fields['NMFS'], -1); Lcol[occ] = remap[cl]; fields['colour rule (GaussianFluent)'] = Lcol
print('colour-rule class fractions (by radius order: inner, mid, outer):', (torch.bincount(remap[cl], minlength=max(_K, 2)).float() / occ.sum()).cpu().numpy().round(3), flush=True)
# (b) the same colour rule PLUS our shell rule -- the symmetric baseline (the reviewer's W2)
Lcs = Lcol.clone(); Lcs[shell] = PEEL; fields['colour rule + shell'] = Lcs
# (c) radial-only: k as a function of r alone, thresholds from the photo classifier's radial class histogram (no features at all)
d = np.load(NMFS); ridx = d['idx'][:, 0].astype(int); labs = d['labels'].astype(int); NRc = int(d['shape'][0])
hist = np.zeros((NRc, _K))
for r_, l_ in zip(ridx, labs): hist[r_, l_] += 1
kr = torch.from_numpy(hist.argmax(1)).to(dv)                       # most frequent class at each radius
ri_all = (rad / E * NRc - 0.5).round().long().clamp(0, NRc - 1)
Lr = torch.full_like(fields['NMFS'], -1); Lr[occ] = kr[ri_all[occ]]; Lr[shell] = PEEL; fields['radial-only k(r) + shell'] = Lr
print('radial-only class fractions:', (torch.bincount(Lr[occ], minlength=_K).float() / occ.sum()).cpu().numpy().round(3), flush=True)

def tau_of(L):
    T = torch.zeros_like(L, dtype=torch.float32)
    for k, v in TAU.items(): T[L == k] = v
    return T
def profile(L, cut):
    """cut = ('trans', z0) : plane v = z0 (perpendicular to the axis), blade edge along u1, advancing along u0
       cut = ('long', 0)   : plane u1 = 0 (through the axis), blade edge along v, advancing along u0"""
    T = tau_of(L); i0 = int(round(vol.c)); AX = vol.AXD; rest = [d for d in range(3) if d != AX]
    if cut[0] == 'trans': plane = T.select(AX, i0 + cut[1]); adv, edge = 0, 1              # plane indices are (rest[0], rest[1]) = (u0, u1)
    else: plane = T.select(rest[1], i0); adv = 0; edge = 1                                  # (u0, v)
    F_s = plane.sum(edge)                                                                  # sum over the edge at each advance index
    return F_s.cpu().numpy()
cuts = {'transverse cut (plane z=0, blade advancing along x)': ('trans', 0), 'longitudinal cut (plane through the axis)': ('long', 0)}
res = {}
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [1.7, 1]})
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120]], np.uint8)
for i, (title, cut) in enumerate(cuts.items()):
    res[title] = {}
    for name, L in fields.items():
        f = profile(L, cut); s = np.nonzero(f)[0]; f = f[s.min():s.max() + 1]; f = f / f.max(); res[title][name] = f.tolist()
        ax[i, 0].plot(np.linspace(0, 1, len(f)), f, label=name, lw=2.2 if name == 'NMFS' else 1.3, alpha=0.95 if name == 'NMFS' else 0.8, ls='--' if 'radial' in name else '-')
    ax[i, 0].set_title(title, fontsize=10); ax[i, 0].set_xlabel('blade advance s (normalised)'); ax[i, 0].set_ylabel('F(s) / max, relative tau'); ax[i, 0].legend(fontsize=8); ax[i, 0].grid(alpha=0.3)
    AX = vol.AXD; rest = [d for d in range(3) if d != AX]; i0 = int(round(vol.c))
    Lp = fields['NMFS'].select(AX, i0) if cut[0] == 'trans' else fields['NMFS'].select(rest[1], i0)
    ax[i, 1].imshow(cmap[np.clip(Lp.cpu().numpy().T + 1, 0, 3)], origin='lower'); ax[i, 1].set_title('NMFS field on the cut plane (blade sweeps left to right)', fontsize=8); ax[i, 1].set_xticks([]); ax[i, 1].set_yticks([])
plt.suptitle('predicted cut-force profile shape: heterogeneous fields vs homogeneous vs the colour rule  (tau: peel 1.0, fibrous 0.5, flesh 0.15; shape only)', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'cutforce_profiles.png'), dpi=120); json.dump(res, open(os.path.join(OUT, 'cutforce_profiles.json'), 'w'))
torch.save({k: v.cpu().to(torch.int8) for k, v in fields.items()}, os.path.join(OUT, 'fields_cartesian.pt'))
# shape statistics: peel spikes at entry/exit and the centre level, per field
for title in res:
    print(title)
    for name, f in res[title].items():
        f = np.array(f); n = len(f); edge = max(f[:n // 10].max(), f[-n // 10:].max()); mid = f[n // 2 - n // 20: n // 2 + n // 20].mean(); flesh = np.median(np.r_[f[n // 5: 2 * n // 5], f[3 * n // 5: 4 * n // 5]])
        print(f'  {name:28s} entry/exit spike {edge:.2f}   flesh plateau {flesh:.2f}   centre {mid:.2f}   centre/plateau {mid / max(flesh, 1e-6):.2f}')
print('saved', os.path.join(OUT, 'cutforce_profiles.png'))
