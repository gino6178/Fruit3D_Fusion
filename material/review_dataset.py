"""Audit figure for the cross-object dataset: every object, a few photographs, the shared-vocabulary
labels and the boundaries on the photo, plus per-object class fractions. This is the figure to check
the automatic role mapping by eye."""
import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1]; OUT = sys.argv[2]; NPER = int(sys.argv[3]) if len(sys.argv) > 3 else 4
idx = json.load(open(os.path.join(D, 'index.json'))); items = idx['items']
NAMES = {0: 'skin', 1: 'flesh', 2: 'fibrous', 3: 'inclusion'}
COL = np.array([[255, 255, 255], [250, 220, 120], [230, 90, 40], [60, 140, 230], [25, 25, 25]], np.uint8)  # -1,0,1,2,3
objs = []
for it in items:
    if it['object'] not in objs: objs.append(it['object'])
rows = []
for o in objs:
    v = [it for it in items if it['object'] == o]
    spl = [it for it in v if it['split'] == 'spl'][:max(1, NPER // 2)]
    hld = [it for it in v if it['split'] == 'hld'][:NPER - len(spl)]
    rows.append((o, spl + hld, np.mean([it['fractions'] for it in v], 0), len(v)))
ncol = max(len(r[1]) for r in rows) * 2
fig, ax = plt.subplots(len(rows), ncol, figsize=(1.75 * ncol, 2.0 * len(rows)), squeeze=False)
for i, (o, its, frac, n) in enumerate(rows):
    for j, it in enumerate(its):
        d = np.load(os.path.join(D, it['file'])); im = d['image']; L = d['label'].astype(int); fg = d['fg']
        ax[i, 2 * j].imshow(im); ax[i, 2 * j].set_title(f"{it['split']} {it['family']}", fontsize=6)
        vis = COL[np.clip(L + 1, 0, 4)]; ax[i, 2 * j + 1].imshow(vis)
        ax[i, 2 * j + 1].set_title(' '.join(f'{x:.2f}' for x in it['fractions']), fontsize=5.5)
    for j in range(2 * len(its), ncol): ax[i, j].axis('off')
    ax[i, 0].set_ylabel(f"{o}\nn={n}\n" + ' '.join(f'{NAMES[k][:4]} {frac[k]:.2f}' for k in range(4)), fontsize=6.5)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
plt.suptitle('cross-object material dataset, shared vocabulary: skin (pale yellow) / flesh (orange) / fibrous (blue) / inclusion (black). No pixel labelled by hand.', fontsize=9)
plt.tight_layout(); plt.savefig(OUT, dpi=120); print('saved', OUT)
print(f'{"object":12s} {"n":>4s}  ' + '  '.join(f'{NAMES[k]:>9s}' for k in range(4)))
for o, _, frac, n in rows: print(f'{o:12s} {n:4d}  ' + '  '.join(f'{frac[k]:9.3f}' for k in range(4)))
