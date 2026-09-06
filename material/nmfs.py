"""NMFS -- neural material-field segmenter on the cylinder, the corrected version.
  M_theta : (PE(r,phi,z), F_t, F_l, local context) -> (P_t, P_l, P, P_incl)
losses   : L_pseudo  CE(P, photo-classifier label) on confident cells only   (weak supervision, real photographs)
           L_cross   symmetric KL(P_t || P_l)                                  (self-supervision, the two families)
           L_geom    anisotropic gradient penalty on P: L1 along r (cheap), L2 along phi and z (expensive)
           L_incl    focal loss for thin fibrous structures outside the core; those cells are exempt from L_geom
No 3D labels are used.  t stays the Stage-2 PC1 (no supervision exists for it)."""
import os, sys, math, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
CELLS, PHOTO, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True); dv = 'cuda'; torch.manual_seed(0)
STEPS, BS = int(os.environ.get('STEPS', 2500)), 32768
W_PSEUDO, W_CROSS, W_R, W_T, W_INCL = 1.0, 0.5, float(os.environ.get('W_R', 0.05)), float(os.environ.get('W_T', 0.5)), 0.5
CONF = 0.8; R_CORE = 0.25; FIBRE, FLESH, PEEL = 0, 1, 2

d = np.load(CELLS); idx = torch.from_numpy(d['idx'].astype(np.int64)).to(dv); NR, NPHI, NZ = [int(x) for x in d['shape']]; M = len(idx)
Ft = torch.from_numpy(d['Ft']).to(dv).float(); Fl = torch.from_numpy(d['Fl']).to(dv).float(); Fa = torch.from_numpy(d['F']).to(dv).float()
mu, sd = Fa.mean(0), Fa.std(0) + 1e-6; Ft = (Ft - mu) / sd; Fl = (Fl - mu) / sd; Fa = (Fa - mu) / sd; NF = Ft.shape[1]
p = np.load(PHOTO); K = int(p['K']); PSEUDO_FAM = os.environ.get('PSEUDO_FAM', 'both')     # 'trans' | 'long' | 'both'
_src = {'both': 'P', 'trans': 'Pt', 'long': 'Pl'}[PSEUDO_FAM]
P0 = torch.from_numpy(p[_src].astype(np.float32)).to(dv); lab0 = P0.argmax(1); conf = P0.max(1).values > CONF
print(f'pseudo-labels from: {PSEUDO_FAM} ({_src})', flush=True)
rn = (idx[:, 0].float() + 0.5) / NR; zn = (idx[:, 2].float() + 0.5) / NZ * 2 - 1; phn = idx[:, 1].float() / NPHI * 2 * math.pi
incl_t = ((lab0 == FIBRE) & (rn > R_CORE)).float()                       # thin fibrous tissue outside the core: membranes / albedo
print(f'{M:,} cells, K={K}, confident pseudo-labels {float(conf.float().mean()):.3f}, inclusion targets {float(incl_t.mean()):.3f}', flush=True)

# ---- positional encoding
def pe(r, ph, z):
    out = []
    for k in range(4):
        f = 2 ** k * math.pi; out += [torch.sin(f * r), torch.cos(f * r), torch.sin(f * z), torch.cos(f * z)]
    for k in range(1, 5): out += [torch.sin(k * ph), torch.cos(k * ph)]
    return torch.stack(out, 1)                                            # 24
PE = pe(rn, phn, zn)

# ---- local context: mean-pooled features on a coarse grid -> two thin periodic 3D convs -> trilinear sample
CR, CP, CZ = NR // 2, NPHI // 4, NZ // 2
cid = (idx[:, 0] // 2) * (CP * CZ) + (idx[:, 1] // 4) * CZ + (idx[:, 2] // 2)
pool = torch.zeros(CR * CP * CZ, NF, device=dv).index_add_(0, cid, Fa); cnt = torch.zeros(CR * CP * CZ, device=dv).index_add_(0, cid, torch.ones(M, device=dv))
pool = (pool / cnt.clamp_min(1)[:, None]).T.reshape(1, NF, CR, CP, CZ)
class Context(nn.Module):
    def __init__(s, c=16):
        super().__init__(); s.c1 = nn.Conv3d(NF, c, 3); s.c2 = nn.Conv3d(c, c, 3)
    def forward(s, g):
        pad = lambda x: F.pad(F.pad(x, (1, 1, 0, 0, 1, 1), mode='replicate'), (0, 0, 1, 1, 0, 0), mode='circular')   # z, r replicate; phi periodic
        return s.c2(pad(F.silu(s.c1(pad(g)))))
gcoord = torch.stack([(idx[:, 2].float() + 0.5) / NZ * 2 - 1, (idx[:, 1].float() + 0.5) / NPHI * 2 - 1, (idx[:, 0].float() + 0.5) / NR * 2 - 1], 1)  # (z, phi, r) as (x, y, z) for grid_sample
def sample_ctx(ctx, rows): return F.grid_sample(ctx, gcoord[rows][None, None, None], mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0, 0, :].T

class NMFS(nn.Module):
    def __init__(s, h=128, c=16):
        super().__init__(); s.ctx = Context(c); inp = 24 + NF + c
        mk = lambda: nn.Sequential(nn.Linear(inp, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        s.ft, s.fl = mk(), mk(); s.head = nn.Linear(h, K); s.fuse = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, K + 1))
    def forward(s, rows, ctx):
        c = sample_ctx(ctx, rows); ht = s.ft(torch.cat([PE[rows], Ft[rows], c], 1)); hl = s.fl(torch.cat([PE[rows], Fl[rows], c], 1))
        o = s.fuse(torch.cat([ht, hl], 1)); return s.head(ht), s.head(hl), o[:, :K], o[:, K]
net = NMFS().to(dv); opt = torch.optim.Adam(net.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

# ---- neighbour lookup for the geometry loss
rowmap = torch.full((NR, NPHI, NZ), -1, dtype=torch.int64, device=dv); rowmap[idx[:, 0], idx[:, 1], idx[:, 2]] = torch.arange(M, device=dv)
def neighbours(rows):
    r, ph, z = idx[rows].T
    nb = {'r': rowmap[(r + 1).clamp(max=NR - 1), ph, z], 'phi': rowmap[r, (ph + 1) % NPHI, z], 'z': rowmap[r, ph, (z + 1).clamp(max=NZ - 1)]}
    return nb
def focal(logit, target, gamma=2.0):
    p = torch.sigmoid(logit); pt = torch.where(target > 0.5, p, 1 - p); w = (1 - pt) ** gamma
    return (w * F.binary_cross_entropy_with_logits(logit, target, reduction='none')).mean()

t0 = time.time(); log = []
for step in range(STEPS):
    ctx = net.ctx(pool)
    rows = torch.randint(0, M, (BS,), device=dv); nb = neighbours(rows)
    allrows = torch.cat([rows] + [nb[k] for k in ('r', 'phi', 'z')]); valid = allrows >= 0; allrows = allrows.clamp(min=0)
    lt, ll, lp, li = net(allrows, ctx)
    P = torch.softmax(lp, 1); Pt = torch.log_softmax(lt, 1); Pl = torch.log_softmax(ll, 1)
    n = BS; P0b, Ptb, Plb, lib = P[:n], Pt[:n], Pl[:n], li[:n]
    L_pseudo = F.cross_entropy(lp[:n][conf[rows]], lab0[rows][conf[rows]]) if conf[rows].any() else lp.sum() * 0
    L_cross = 0.5 * (F.kl_div(Ptb, Plb, log_target=True, reduction='batchmean') + F.kl_div(Plb, Ptb, log_target=True, reduction='batchmean'))
    keep = (incl_t[rows] < 0.5).float()                                     # inclusion cells are exempt from smoothing
    dr = (P[n:2 * n] - P0b).abs().sum(1) * valid[n:2 * n] * keep; dp = ((P[2 * n:3 * n] - P0b) ** 2).sum(1) * valid[2 * n:3 * n] * keep; dz = ((P[3 * n:] - P0b) ** 2).sum(1) * valid[3 * n:] * keep
    L_geom = W_R * dr.mean() + W_T * (dp.mean() + dz.mean())
    L_incl = focal(lib, incl_t[rows])
    loss = W_PSEUDO * L_pseudo + W_CROSS * L_cross + L_geom + W_INCL * L_incl
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 250 == 0 or step == STEPS - 1:
        with torch.no_grad(): dis = float((Ptb.argmax(1) != Plb.argmax(1)).float().mean())
        print(f'step {step:5d}  loss {loss.item():.3f}  pseudo {L_pseudo.item():.3f} cross {L_cross.item():.3f} geom {L_geom.item():.3f} incl {L_incl.item():.3f}  batch cross-disagree {dis:.3f}  [{time.time()-t0:.0f}s]', flush=True)

# ---- inference over all cells
net.eval(); outs = {'lt': [], 'll': [], 'lp': [], 'li': []}
with torch.no_grad():
    ctx = net.ctx(pool)
    for i in range(0, M, 1000000):
        rows = torch.arange(i, min(i + 1000000, M), device=dv); lt, ll, lp, li = net(rows, ctx)
        outs['lt'].append(lt.argmax(1)); outs['ll'].append(ll.argmax(1)); outs['lp'].append(torch.softmax(lp, 1)); outs['li'].append(torch.sigmoid(li))
lt, ll, P, Pi = torch.cat(outs['lt']), torch.cat(outs['ll']), torch.cat(outs['lp']), torch.cat(outs['li']); lab = P.argmax(1)
lab_final = torch.where(Pi > 0.5, torch.full_like(lab, FIBRE), lab)          # inclusion head overrides: thin fibrous structures are kept

# ---- metrics against the classifier (raw), Stage-2 MRF, and the held-out-verified pseudo-labels on NON-confident cells
mrf = torch.from_numpy(np.load(os.path.join(os.path.dirname(PHOTO), 'field_cells.npz'))['labels'].astype(np.int64)).to(dv)
Pt0, Pl0 = torch.from_numpy(p['Pt'].astype(np.float32)).to(dv).argmax(1), torch.from_numpy(p['Pl'].astype(np.float32)).to(dv).argmax(1)
def frac(l): return (torch.bincount(l, minlength=K).float() / M).cpu().numpy().round(3).tolist()
fib_out = lambda l: float(((l == FIBRE) & (rn > R_CORE)).float().mean())
def seam(l):
    L = torch.full((NR, NPHI, NZ), -1, dtype=torch.int64, device=dv); L[idx[:, 0], idx[:, 1], idx[:, 2]] = l
    R = 512; lin = torch.linspace(-1, 1, R, device=dv); V, U = torch.meshgrid(lin, lin, indexing='ij'); out = {}
    for name, (rr, pp, zz) in {'trans': (torch.sqrt(U ** 2 + V ** 2), torch.atan2(V, U) % (2 * math.pi), torch.zeros_like(U)),
                               'oblique': (torch.sqrt(U ** 2 + (V * math.sin(math.pi / 4)) ** 2), torch.atan2(V * math.sin(math.pi / 4), U) % (2 * math.pi), V * math.cos(math.pi / 4))}.items():
        ri = (rr * NR - 0.5).round().long().clamp(0, NR - 1); pi_ = (pp / (2 * math.pi) * NPHI).round().long() % NPHI; zi = ((zz + 1) / 2 * NZ - 0.5).round().long().clamp(0, NZ - 1)
        g = L[ri, pi_, zi]; v = (rr <= 1) & (g >= 0)
        ch = ((g[:, 1:] != g[:, :-1]) & v[:, 1:] & v[:, :-1]).float().sum() + ((g[1:] != g[:-1]) & v[1:] & v[:-1]).float().sum(); out[name] = float(ch / v.float().sum().clamp_min(1))
    return out
nc = ~conf
_other = {'trans': 'Pl', 'long': 'Pt', 'both': None}[PSEUDO_FAM]
if _other is not None:
    Po = torch.from_numpy(p[_other].astype(np.float32)).to(dv); oc = Po.max(1).values > CONF; ol = Po.argmax(1)
    heldout_family = dict(family=_other, agree_nmfs=float((lab_final[oc] == ol[oc]).float().mean()), agree_mrf=float((mrf[oc] == ol[oc]).float().mean()), agree_pseudo_source=float((lab0[oc] == ol[oc]).float().mean()), n=int(oc.sum()))
    print('HELD-OUT FAMILY', json.dumps(heldout_family), flush=True)
else: heldout_family = None
metrics = dict(K=K, steps=STEPS, w_r=W_R, w_t=W_T, pseudo_fam=PSEUDO_FAM, heldout_family=heldout_family, confident_fraction=float(conf.float().mean()),
    cross_disagree=dict(raw_classifier=float((Pt0 != Pl0).float().mean()), nmfs=float((lt != ll).float().mean())),
    agree_with_classifier_on_nonconfident=dict(mrf=float((mrf[nc] == lab0[nc]).float().mean()), nmfs=float((lab_final[nc] == lab0[nc]).float().mean())),
    fibre_outside_core=dict(classifier=fib_out(lab0), mrf=fib_out(mrf), nmfs_main=fib_out(lab), nmfs_with_inclusions=fib_out(lab_final)),
    seam=dict(mrf=seam(mrf), nmfs=seam(lab_final)), fractions=dict(classifier=frac(lab0), mrf=frac(mrf), nmfs=frac(lab_final)),
    changed_vs_mrf=float((lab_final != mrf).float().mean()), train_seconds=round(time.time() - t0))
print(json.dumps(metrics, indent=1)); json.dump(metrics, open(os.path.join(OUT, 'nmfs_metrics.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'nmfs_cells.npz'), labels=lab_final.cpu().numpy().astype(np.int8), P=P.half().cpu().numpy(), P_incl=Pi.half().cpu().numpy(), idx=idx.cpu().numpy().astype(np.int16), shape=np.array([NR, NPHI, NZ]))
torch.save(net.state_dict(), os.path.join(OUT, 'nmfs.pt'))

# ---- figure: classifier argmax | MRF | NMFS | inclusion head, on transverse / longitudinal / oblique
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120]], np.uint8)
def grid_of(l): L = torch.full((NR, NPHI, NZ), -1, dtype=torch.int64, device=dv); L[idx[:, 0], idx[:, 1], idx[:, 2]] = l; return L
G = {'photo classifier': grid_of(lab0), 'Stage-2 MRF': grid_of(mrf), 'NMFS': grid_of(lab_final)}
Gi = torch.zeros((NR, NPHI, NZ), device=dv); Gi[idx[:, 0], idx[:, 1], idx[:, 2]] = Pi
R = 512; lin = torch.linspace(-1, 1, R, device=dv); V, U = torch.meshgrid(lin, lin, indexing='ij')
planes = {'transverse z=0': (torch.sqrt(U ** 2 + V ** 2), torch.atan2(V, U) % (2 * math.pi), torch.zeros_like(U)), 'longitudinal phi=0': (U.abs(), torch.where(U >= 0, 0.0, math.pi) * torch.ones_like(U), V),
          'oblique 45 deg': (torch.sqrt(U ** 2 + (V * math.sin(math.pi / 4)) ** 2), torch.atan2(V * math.sin(math.pi / 4), U) % (2 * math.pi), V * math.cos(math.pi / 4))}
def look(Lg, pl, prob=False):
    rr, pp, zz = pl; ri = (rr * NR - 0.5).round().long().clamp(0, NR - 1); pi_ = (pp / (2 * math.pi) * NPHI).round().long() % NPHI; zi = ((zz + 1) / 2 * NZ - 0.5).round().long().clamp(0, NZ - 1)
    if prob: v = Lg[ri, pi_, zi]; im = plt.cm.magma(v.cpu().numpy())[..., :3]; im[(rr > 1).cpu().numpy()] = 1; return im
    g = Lg[ri, pi_, zi].clone(); g[rr > 1] = -1; return cmap[np.clip(g.cpu().numpy() + 1, 0, 3)]
fig, ax = plt.subplots(4, 3, figsize=(9.6, 12.4))
for i, (name, Lg) in enumerate(list(G.items()) + [('inclusion head P', Gi)]):
    for j, (pn, pl) in enumerate(planes.items()):
        ax[i, j].imshow(look(Lg, pl, prob=(i == 3))); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0: ax[i, j].set_title(pn, fontsize=9)
    ax[i, 0].set_ylabel(name, fontsize=9)
plt.suptitle(f'NMFS vs classifier and MRF: cross-family disagreement {metrics["cross_disagree"]["raw_classifier"]:.3f} -> {metrics["cross_disagree"]["nmfs"]:.3f}; fibre outside core: MRF {metrics["fibre_outside_core"]["mrf"]:.3f}, NMFS {metrics["fibre_outside_core"]["nmfs_with_inclusions"]:.3f}', fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'nmfs.png'), dpi=120); print('saved', os.path.join(OUT, 'nmfs.png'))
