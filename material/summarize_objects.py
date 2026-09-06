"""E1 / E8 summary across objects: material inventory K (by transfer), class fractions, cross-family
disagreement, MRF metrics, cut-force profile shape per field, and a figure of every object's three
labelled slices.  Reads what run_all_objects.sh wrote."""
import os, sys, json, glob, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); objs = sys.argv[1:] or ['orange_repro', 'watermelon', 'apple', 'bread', 'cake', 'doughnut', 'pomegranate']
rows = []
for o in objs:
    r = dict(object=o.replace('_repro', ''))
    try: j = json.load(open(f'{ROOT}/runs/{o}/material_v2/photo2asset.json')); r.update(K=j['K'], transfer={k: round(v['agree'], 3) for k, v in j['transfer'].items()}, fractions=[round(x, 3) for x in j['fractions']], cross_disagree=round(j['disagreement'], 3))
    except Exception as e: r['photo2asset'] = 'missing'
    try: m = json.load(open(f'{ROOT}/runs/{o}/material_v2/stage2_metrics.json')); r.update(mrf_changed=round(m['changed_by_mrf'], 3), seam_obl=round(m['seam_change_rate_oblique'], 3), seam_trans=round(m['seam_change_rate_trans'], 3), shell_cells=m['shell_cells'])
    except Exception: r['stage2'] = 'missing'
    try:
        p = json.load(open(f'{ROOT}/runs/{o}/stage4/cutforce_profiles.json')); key = [k for k in p if k.startswith('transverse')][0]; shp = {}
        for name, f in p[key].items():
            f = np.array(f); n = len(f); shp[name] = dict(spike=round(float(max(f[:n // 10].max(), f[-n // 10:].max())), 2), plateau=round(float(np.median(np.r_[f[n // 5:2 * n // 5], f[3 * n // 5:4 * n // 5]])), 2))
        r['cutforce_trans'] = shp
    except Exception: r['stage4'] = 'missing'
    rows.append(r)
json.dump(rows, open(f'{ROOT}/runs/objects_summary.json', 'w'), indent=1)
print(f'{"object":12s} {"K":>2s} {"transfer(K)":>28s} {"fractions":>22s} {"x-fam":>6s} {"seam obl/tr":>12s} {"spike MRF/homog/colour+shell/radial":>38s}')
for r in rows:
    tr = ' '.join(f'{k}:{v}' for k, v in r.get('transfer', {}).items()); cf = r.get('cutforce_trans', {})
    sp = '/'.join(str(cf.get(k, {}).get('spike', '-')) for k in ('GMM+MRF', 'homogeneous', 'colour rule + shell', 'radial-only k(r) + shell'))
    print(f'{r["object"]:12s} {str(r.get("K","-")):>2s} {tr:>28s} {str(r.get("fractions","-")):>22s} {str(r.get("cross_disagree","-")):>6s} {str(r.get("seam_obl","-"))+"/"+str(r.get("seam_trans","-")):>12s} {sp:>38s}')
# figure: three labelled slices per object (from stage2_field.png crops is messy; re-render from field_cells.npz)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, math
cmap = np.array([[255, 255, 255], [60, 140, 230], [230, 90, 40], [250, 220, 120], [220, 60, 140], [90, 200, 120], [150, 100, 220]], np.uint8)
have = [o for o in objs if os.path.exists(f'{ROOT}/runs/{o}/material_v2/field_cells.npz')]
if have:
    fig, ax = plt.subplots(len(have), 4, figsize=(12.5, 3.1 * len(have)))
    if len(have) == 1: ax = ax[None]
    for i, o in enumerate(have):
        d = np.load(f'{ROOT}/runs/{o}/material_v2/field_cells.npz'); idx = d['idx'].astype(int); NR, NPHI, NZ = [int(x) for x in d['shape']]; lab = d['labels'].astype(int)
        L = np.full((NR, NPHI, NZ), -1, np.int16); L[idx[:, 0], idx[:, 1], idx[:, 2]] = lab
        A = torch.load(f'{ROOT}/runs/{o}/cyl/state.pt', map_location='cpu')['A'].float().numpy()
        R = 384; lin = np.linspace(-1, 1, R); V, U = np.meshgrid(lin, lin, indexing='ij')
        def look(rr, pp, zz, rgb=False):
            ri = np.clip((rr * NR - 0.5).round().astype(int), 0, NR - 1); pi_ = (np.round(pp / (2 * np.pi) * NPHI).astype(int)) % NPHI; zi = np.clip(((zz + 1) / 2 * NZ - 0.5).round().astype(int), 0, NZ - 1)
            if rgb: im = (A[:, ri, pi_, zi].transpose(1, 2, 0) + 1) / 2; im[rr > 1] = 1; return np.clip(im, 0, 1)
            g = L[ri, pi_, zi].copy(); g[rr > 1] = -1; return cmap[np.clip(g + 1, 0, 6)]
        planes = [(np.sqrt(U ** 2 + V ** 2), np.arctan2(V, U) % (2 * np.pi), np.zeros_like(U)), (np.abs(U), np.where(U >= 0, 0.0, np.pi) * np.ones_like(U), V), (np.sqrt(U ** 2 + (V * math.sin(math.pi / 4)) ** 2), np.arctan2(V * math.sin(math.pi / 4), U) % (2 * np.pi), V * math.cos(math.pi / 4))]
        ax[i, 0].imshow(look(*planes[0], rgb=True)); ax[i, 0].set_ylabel(o.replace('_repro', ''), fontsize=10)
        for j, pl in enumerate(planes): ax[i, j + 1].imshow(look(*pl))
        for a in ax[i]: a.set_xticks([]); a.set_yticks([])
        if i == 0:
            for j, t in enumerate(['asset (transverse)', 'field: transverse', 'field: longitudinal', 'field: oblique 45 deg']): ax[0, j].set_title(t, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.985)); plt.suptitle('material field (GMM + MRF) per object, one program, no per-object settings', fontsize=10, y=0.997); plt.savefig(f'{ROOT}/runs/objects_fields.png', dpi=110); print('saved runs/objects_fields.png')
