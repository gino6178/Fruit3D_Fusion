"""De-circularised evaluation of any per-cell label field on the orange: agreement with each cut family's
confident classifier argmax (Pt, Pl), seam rate on an oblique slice, fibre outside the core.
usage: eval_field.py CELLS_PHOTO.npz LABELS.npy|field_cells.npz|nmfs_cells.npz|radial|classifier [name]"""
import sys, math, json, numpy as np
CP, SRC = sys.argv[1], sys.argv[2]; name = sys.argv[3] if len(sys.argv) > 3 else SRC
p = np.load(CP); idx = p['idx'].astype(int); NR, NPHI, NZ = [int(x) for x in p['shape']]; K = int(p['K'])
Pt, Pl, P = p['Pt'].astype(np.float32), p['Pl'].astype(np.float32), p['P'].astype(np.float32)
if SRC == 'classifier': lab = P.argmax(1)
elif SRC == 'radial':
    lab0 = P.argmax(1); hist = np.zeros((NR, K))
    for r_, l_ in zip(idx[:, 0], lab0): hist[r_, l_] += 1
    lab = hist.argmax(1)[idx[:, 0]]
elif SRC.endswith('.npy'): lab = np.load(SRC).astype(int)
else: lab = np.load(SRC)['labels'].astype(int)
out = {'name': name}
for fam, Q in (('trans', Pt), ('long', Pl)):
    c = Q.max(1) > 0.8; out[f'agree_{fam}_confident'] = float((lab[c] == Q.argmax(1)[c]).mean())
out['agree_nonconfident_vs_classifier'] = float((lab[P.max(1) <= 0.8] == P.argmax(1)[P.max(1) <= 0.8]).mean())
rn = (idx[:, 0] + 0.5) / NR; out['fibre_outside_core'] = float(((lab == 0) & (rn > 0.25)).mean())
L = np.full((NR, NPHI, NZ), -1, np.int8); L[idx[:, 0], idx[:, 1], idx[:, 2]] = lab
R = 512; lin = np.linspace(-1, 1, R); V, U = np.meshgrid(lin, lin, indexing='ij')
def seam(rr, pp, zz):
    ri = np.clip((rr * NR - 0.5).round().astype(int), 0, NR - 1); pi_ = (np.round(pp / (2 * np.pi) * NPHI).astype(int)) % NPHI; zi = np.clip(((zz + 1) / 2 * NZ - 0.5).round().astype(int), 0, NZ - 1)
    g = L[ri, pi_, zi]; v = (rr <= 1) & (g >= 0)
    ch = ((g[:, 1:] != g[:, :-1]) & v[:, 1:] & v[:, :-1]).sum() + ((g[1:] != g[:-1]) & v[1:] & v[:-1]).sum(); return float(ch / max(v.sum(), 1))
out['seam_trans'] = seam(np.sqrt(U ** 2 + V ** 2), np.arctan2(V, U) % (2 * np.pi), np.zeros_like(U))
out['seam_oblique'] = seam(np.sqrt(U ** 2 + (V * math.sin(math.pi / 4)) ** 2), np.arctan2(V * math.sin(math.pi / 4), U) % (2 * np.pi), V * math.cos(math.pi / 4))
out['fractions'] = (np.bincount(lab, minlength=K) / len(lab)).round(3).tolist()
print(json.dumps(out))
