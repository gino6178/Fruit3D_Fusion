"""Stage 2: from per-cell class posteriors to the material field (k, t).
 - shell cells (cell_level 1, the O-Voxel skin) are peel by rule, not by classifier
 - anisotropic Potts MRF on the cylinder by mean-field: label changes are cheap along r (layers),
   expensive along phi and z (a substance is constant tangentially)
 - t: first principal component of the cell features inside the flesh class, normalised
 - exports a Cartesian 128^3 label volume for the solver, and before/after metrics"""
import os, sys, math, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')); from planes import Vol
CELLS, FEATS, STATE, GRID, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True); dv = 'cuda'
W_R, W_T, ITERS = float(os.environ.get('W_R', 0.3)), float(os.environ.get('W_T', 1.5)), int(os.environ.get('ITERS', 12))
d = np.load(CELLS); P = torch.from_numpy(d['P'].astype(np.float32)).to(dv); Pt = d['Pt'].astype(np.float32); Pl = d['Pl'].astype(np.float32)
idx = torch.from_numpy(d['idx'].astype(np.int64)).to(dv); NR, NPHI, NZ = [int(x) for x in d['shape']]; K = int(d['K']); M = len(idx)
lab0 = P.argmax(1)
# class roles from the data, not from K: the outermost class (largest mean radius) is the peel, the most
# populous remaining class is the flesh, anything else is fibrous/inclusion tissue (K=2: flesh + peel only)
_r = (idx[:, 0].float() + 0.5) / NR; _mean_r = torch.stack([_r[lab0 == k].mean() if (lab0 == k).any() else torch.tensor(-1., device=dv) for k in range(K)])
PEEL = int(_mean_r.argmax()); _rest = [k for k in range(K) if k != PEEL]; _cnt = torch.bincount(lab0, minlength=K)
FLESH = int(max(_rest, key=lambda k: int(_cnt[k]))); FIBRE = [k for k in _rest if k != FLESH][0] if len(_rest) > 1 else FLESH
print(f'class roles by mean radius: peel={PEEL} flesh={FLESH} fibre={FIBRE}', flush=True)

# ---- dense posterior grid (K, NR, NPHI, NZ); non-core cells carry no evidence (uniform)
Q0 = torch.full((K, NR, NPHI, NZ), 1.0 / K, device=dv); Q0[:, idx[:, 0], idx[:, 1], idx[:, 2]] = P.T.clamp(1e-4, 1)
core = torch.zeros(NR, NPHI, NZ, dtype=torch.bool, device=dv); core[idx[:, 0], idx[:, 1], idx[:, 2]] = True
logU = torch.log(Q0)

# ---- mean-field Potts with anisotropic neighbour weights (phi wraps)
def neigh_sum(Q):
    s = torch.zeros_like(Q)
    s[:, 1:] += W_R * Q[:, :-1]; s[:, :-1] += W_R * Q[:, 1:]                                   # along r
    s += W_T * (torch.roll(Q, 1, 2) + torch.roll(Q, -1, 2))                                     # along phi (wrap)
    s[:, :, :, 1:] += W_T * Q[:, :, :, :-1]; s[:, :, :, :-1] += W_T * Q[:, :, :, 1:]           # along z
    return s
Q = Q0.clone()
for it in range(ITERS):
    Q = torch.softmax(logU + neigh_sum(Q), 0); Q = torch.where(core[None], Q, Q0)
lab1 = Q.argmax(0)[idx[:, 0], idx[:, 1], idx[:, 2]]

# ---- shell rule and the full cylinder label grid
vol = Vol(GRID, dv, res=256); E = vol.EXT; c = vol.c
r = (torch.arange(NR, device=dv) + 0.5) / NR * E; ph = torch.arange(NPHI, device=dv) / NPHI * 2 * math.pi; z = -E + (torch.arange(NZ, device=dv) + 0.5) / NZ * 2 * E
RR, PP, ZZ = torch.meshgrid(r, ph, z, indexing='ij'); p0, p1, p2 = (RR * torch.cos(PP) + c).reshape(1, -1), (ZZ + c).reshape(1, -1), (RR * torch.sin(PP) + c).reshape(1, -1)
SHELL = vol._samp(vol.SHELL.expand(3, -1, -1, -1).contiguous(), p0, p1, p2).reshape(3, NR, NPHI, NZ)[0] > 0.5
L = torch.full((NR, NPHI, NZ), -1, dtype=torch.int8, device=dv); L[idx[:, 0], idx[:, 1], idx[:, 2]] = lab1.to(torch.int8); L[SHELL & ~core] = PEEL

# ---- t inside the flesh class: PC1 of the cell features, [0,1]
Fc = torch.from_numpy(np.load(FEATS)['F'].astype(np.float32)).to(dv); Fc = (Fc - Fc.mean(0)) / (Fc.std(0) + 1e-6)
fl = lab1 == FLESH; X = Fc[fl]; sub = X[torch.randperm(len(X), device=dv)[:200000]]
_, _, Vt = torch.linalg.svd(sub - sub.mean(0), full_matrices=False); pc = (X - sub.mean(0)) @ Vt[0]
lo, hi = torch.quantile(pc, 0.02), torch.quantile(pc, 0.98); t = torch.zeros(M, device=dv); t[fl] = ((pc - lo) / (hi - lo)).clamp(0, 1)

# ---- metrics
lt, ll = torch.from_numpy(Pt.argmax(1)).to(dv), torch.from_numpy(Pl.argmax(1)).to(dv)
def dis(l): return float(((l != lt) | (l != ll)).float().mean()), float((l != lt).float().mean()), float((l != ll).float().mean())
def frac(l): return (torch.bincount(l, minlength=K).float() / M).cpu().numpy().round(3).tolist()
# label change rate along an unsupervised oblique plane vs along a transverse plane (seam test)
def change_rate(plane):
    rr, pp, zz = plane; ri = ((rr * NR - 0.5).round().long().clamp(0, NR - 1)); pi_ = ((pp / (2 * math.pi) * NPHI).round().long()) % NPHI; zi = (((zz + 1) / 2 * NZ - 0.5).round().long().clamp(0, NZ - 1))
    l = L[ri, pi_, zi].long(); valid = (rr <= 1) & (l >= 0)
    ch = ((l[:, 1:] != l[:, :-1]) & valid[:, 1:] & valid[:, :-1]).float().sum() + ((l[1:] != l[:-1]) & valid[1:] & valid[:-1]).float().sum()
    return float(ch / valid.float().sum().clamp_min(1))
R = 512; lin = torch.linspace(-1, 1, R, device=dv); V, U = torch.meshgrid(lin, lin, indexing='ij'); x = U; y = V * math.cos(math.pi / 4); zc = V * math.sin(math.pi / 4)
planes = dict(trans=(torch.sqrt(U ** 2 + V ** 2), torch.atan2(V, U) % (2 * math.pi), torch.zeros_like(U)), obl=(torch.sqrt(x ** 2 + zc ** 2), torch.atan2(zc, x) % (2 * math.pi), y))
m0, m1 = dis(lab0), dis(lab1)
fibre_in_flesh_before = float(((lab0 == FIBRE) & ((idx[:, 0].float() + 0.5) / NR > 0.25)).float().mean()); fibre_in_flesh_after = float(((lab1 == FIBRE) & ((idx[:, 0].float() + 0.5) / NR > 0.25)).float().mean())
metrics = dict(K=K, W_R=W_R, W_T=W_T, iters=ITERS, fractions_before=frac(lab0), fractions_after=frac(lab1),
               disagree_any_before=m0[0], disagree_any_after=m1[0], vs_trans_after=m1[1], vs_long_after=m1[2],
               changed_by_mrf=float((lab0 != lab1).float().mean()), shell_cells=int((SHELL & ~core).sum()),
               fibre_outside_core_before=fibre_in_flesh_before, fibre_outside_core_after=fibre_in_flesh_after,
               seam_change_rate_trans=change_rate(planes['trans']), seam_change_rate_oblique=change_rate(planes['obl']))
print(json.dumps(metrics, indent=1)); json.dump(metrics, open(os.path.join(OUT, 'stage2_metrics.json'), 'w'), indent=1)

# ---- Cartesian export for the solver: every occupied voxel -> nearest cylinder cell label
N = vol.N; rad = torch.sqrt(vol.u0 ** 2 + vol.u1 ** 2); ang = torch.atan2(vol.u1, vol.u0) % (2 * math.pi)
ri = (rad / E * NR - 0.5).round().long().clamp(0, NR - 1); pi_ = (ang / (2 * math.pi) * NPHI).round().long() % NPHI; zi = ((vol.v + E) / (2 * E) * NZ - 0.5).round().long().clamp(0, NZ - 1)
Lc = L[ri, pi_, zi]; occ = vol.OCC[0] > 0.5; Lc[~occ] = -1; Lc[(vol.SHELL[0] > 0.5)] = PEEL
tgrid = torch.zeros(NR, NPHI, NZ, device=dv); tgrid[idx[:, 0], idx[:, 1], idx[:, 2]] = t; tc = tgrid[ri, pi_, zi]
torch.save({'labels': Lc.cpu().to(torch.int8), 't': tc.cpu().half(), 'classes': {FIBRE: 'fibrous core/membranes', FLESH: 'flesh', PEEL: 'peel'}, 'AXD': vol.AXD}, os.path.join(OUT, 'material_cartesian.pt'))
np.savez_compressed(os.path.join(OUT, 'field_cells.npz'), labels=lab1.cpu().numpy().astype(np.int8), t=t.cpu().half().numpy(), idx=idx.cpu().numpy().astype(np.int16), shape=np.array([NR, NPHI, NZ]))
print('cartesian volume: occupied', int(occ.sum()), 'labelled', int((Lc >= 0).sum()), 'per class', torch.bincount(Lc[Lc >= 0].long(), minlength=K).tolist())

# ---- figure: before / after / t on transverse, longitudinal, oblique
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140]], np.uint8)
L0 = torch.full((NR, NPHI, NZ), -1, dtype=torch.int8, device=dv); L0[idx[:, 0], idx[:, 1], idx[:, 2]] = lab0.to(torch.int8)
A = torch.load(STATE, map_location=dv)['A'].float()
def look(plane, what):
    rr, pp, zz = plane; ri = ((rr * NR - 0.5).round().long().clamp(0, NR - 1)); pi_ = ((pp / (2 * math.pi) * NPHI).round().long()) % NPHI; zi = (((zz + 1) / 2 * NZ - 0.5).round().long().clamp(0, NZ - 1))
    out_of = (rr > 1)
    if what == 'rgb': im = ((A[:, ri, pi_, zi].permute(1, 2, 0) + 1) / 2).clamp(0, 1); im[out_of] = 1; return im.cpu().numpy()
    if what == 't': v = tgrid[ri, pi_, zi]; msk = (L[ri, pi_, zi] == FLESH) & ~out_of; im = plt.cm.viridis(v.cpu().numpy())[..., :3]; im[~msk.cpu().numpy()] = 1; return im
    src = L0 if what == 'before' else L; l = src[ri, pi_, zi].long(); l[out_of] = -1; return cmap[np.clip(l.cpu().numpy() + 1, 0, 4)]
planes['long'] = (U.abs(), torch.where(U >= 0, 0.0, math.pi) * torch.ones_like(U), V)
order = ['trans', 'long', 'obl']; fig, ax = plt.subplots(4, 3, figsize=(9.6, 12.4))
for i, what in enumerate(['rgb', 'before', 'after', 't']):
    for j, pn in enumerate(order):
        ax[i, j].imshow(look(planes[pn], what)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title({'trans': 'transverse z=0', 'long': 'longitudinal phi=0', 'obl': 'oblique 45 deg'}[pn], fontsize=9)
    ax[i, 0].set_ylabel({'rgb': 'asset', 'before': 'posterior argmax\n(disagree %.2f)' % m0[0], 'after': 'MRF + shell rule\n(disagree %.2f)' % m1[0], 't': 't within flesh (PC1)'}[what], fontsize=8)
plt.suptitle(f'Stage 2 material field (k, t): K={K} learned on photographs; MRF w_r={W_R} w_phi=w_z={W_T}', fontsize=9); plt.tight_layout(); plt.savefig(os.path.join(OUT, 'stage2_field.png'), dpi=120); print('saved', os.path.join(OUT, 'stage2_field.png'))
