"""Ground truth for 2D material segmentation.

Boundaries come from SAM (propose_batch.py). Naming is done here, per object, by explicit rules
written after measuring the photographs -- this is annotation, not a method: every map is rendered
and checked by eye, and the rules are allowed to be object-specific because ground truth is.

Where SAM found the boundary (orange peel/flesh/core, pomegranate arils) the rule only names its
regions. Where SAM missed it (the watermelon rind on the longitudinal cut, the apple's one-pixel
skin, every seed) the rule cuts the boundary itself from colour and depth.

usage: make_gt.py OUT_DIR obj:family:image ..."""
import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feats import RES, fg_mask
from scipy import ndimage

def ctx(src, regf):
    im = cv2.resize(cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_AREA)
    fg = fg_mask(im); lab = cv2.cvtColor(im.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    d = ndimage.distance_transform_edt(fg); d = d / max(d.max(), 1e-6)
    reg = np.load(regf) if os.path.exists(regf) else np.full(fg.shape, -1, np.int16)
    return dict(im=im, fg=fg, L=lab[..., 0], a=lab[..., 1], b=lab[..., 2], d=d, reg=reg)

def regstat(c):
    """mean depth / L* / a* / b* / area of every SAM region"""
    out = {}
    for i in [int(x) for x in np.unique(c['reg'][c['reg'] >= 0])]:
        m = c['reg'] == i
        out[i] = dict(d=float(c['d'][m].mean()), L=float(c['L'][m].mean()), a=float(c['a'][m].mean()),
                      b=float(c['b'][m].mean()), area=float(m.mean()), m=m)
    return out

def clean(lab, fg, k=5, minpx=120):
    """majority filter, then absorb specks smaller than minpx into the surrounding class"""
    out = lab.copy()
    for c in np.unique(lab[fg]):
        m = (lab == c) & fg
        m = ndimage.binary_opening(m, np.ones((3, 3)))
        cc, n = ndimage.label(m)
        for j in range(1, n + 1):
            s = cc == j
            if s.sum() < minpx: out[s] = 255
    miss = fg & (out == 255)
    if miss.any() and (out != 255).any():
        idx = ndimage.distance_transform_edt(out == 255, return_distances=False, return_indices=True)
        out = np.where(miss, out[tuple(idx)], out)
    out[~fg] = 255
    return out

def blobs(mask, fg, amin, amax):
    cc, n = ndimage.label(mask & fg); keep = np.zeros_like(mask)
    for j in range(1, n + 1):
        s = cc == j
        if amin <= s.sum() <= amax: keep |= s
    return keep

# ---------------------------------------------------------------- per-object rules
def orange(c):
    """SAM separates peel / flesh / central column cleanly. Name its regions:
       touching the surface -> peel; low chroma -> the white column and its membranes; else flesh."""
    g = np.full(c['fg'].shape, 255, np.uint8)
    for i, s in regstat(c).items():
        g[s['m']] = 0 if s['d'] < 0.25 else (2 if s['a'] < 12 else 1)
    return g, ['peel', 'flesh', 'white core'], [[244, 170, 60], [235, 110, 35], [250, 248, 240]]

def watermelon(c):
    """SAM merges the two rind layers and misses every seed, and on the longitudinal cut it returns
       one region for the whole fruit. Cut the rind from chroma (flesh a*~60, rind a*<15), split it
       green/white at a*=0, and take the seeds as pale compact blobs inside the flesh."""
    g = np.where(c['a'] < 15, np.where(c['a'] < 0, 0, 1), 2).astype(np.uint8)
    g[(c['d'] > 0.25) & (g < 2)] = 2                      # only the outer band can be rind
    g = clean(g, c['fg'], minpx=200)
    seed = blobs((g == 2) & (c['L'] > 58) & (c['a'] < 35), c['fg'], 25, 1500)
    g[seed] = 3
    g[~c['fg']] = 255
    return g, ['green rind', 'white rind', 'red flesh', 'seed'], [[90, 140, 60], [225, 230, 200], [215, 45, 55], [225, 205, 150]]

def apple(c):
    """The skin is one or two pixels wide and SAM never returns it; the core carpels are the only
       dark thing in the frame."""
    g = np.full(c['fg'].shape, 1, np.uint8)
    g[(c['L'] < 62)] = 2
    g[(c['d'] < 0.055) & (c['a'] > 12)] = 0
    g = clean(g, c['fg'], minpx=60)
    return g, ['skin', 'flesh', 'core'], [[200, 50, 60], [245, 230, 190], [90, 55, 35]]

def pomegranate(c):
    """SAM finds the arils one by one, which no colour rule does. Name a region an aril when it is
       dark red; classify what is left per pixel into rind (at the surface) and white membrane."""
    g = np.full(c['fg'].shape, 255, np.uint8)
    for i, s in regstat(c).items():
        if s['a'] > 28 and s['L'] < 55 and s['area'] < 0.05: g[s['m']] = 2
    rest = c['fg'] & (g == 255)
    g[rest & (c['d'] < 0.10)] = 0
    g[rest & (c['d'] >= 0.10)] = 1
    g[rest & (c['a'] > 30) & (c['L'] < 45)] = 2
    g = clean(g, c['fg'], minpx=120)
    return g, ['rind', 'white membrane', 'aril'], [[190, 40, 55], [245, 240, 210], [140, 15, 30]]

def bread(c):
    """crust is the dark browned band at the surface; everything inside is crumb, and the holes in
       the crumb are foreground already."""
    g = np.full(c['fg'].shape, 1, np.uint8)
    g[(c['d'] < 0.10) & (c['L'] < 78)] = 0
    g = clean(g, c['fg'], minpx=120)
    return g, ['crust', 'crumb'], [[150, 95, 45], [240, 225, 190]]

def cake(c):
    g = np.full(c['fg'].shape, 1, np.uint8)
    g[(c['L'] > 85) & (c['b'] < 22)] = 0
    g = clean(g, c['fg'], minpx=150)
    return g, ['cream', 'sponge'], [[250, 248, 240], [215, 175, 110]]

def doughnut(c):
    g = np.full(c['fg'].shape, 1, np.uint8)
    g[(c['d'] < 0.14) & (c['L'] < 72)] = 0
    g = clean(g, c['fg'], minpx=150)
    return g, ['glaze / crust', 'dough'], [[130, 80, 45], [238, 205, 150]]

RULES = dict(orange=orange, watermelon=watermelon, apple=apple, pomegranate=pomegranate,
             bread=bread, cake=cake, doughnut=doughnut)

OUTD = sys.argv[1]; os.makedirs(OUTD, exist_ok=True); index = {}
for spec in sys.argv[2:]:
    obj, fam, src = spec.split(':', 2)
    tag = f'{obj}_spl_{fam}_' + os.path.splitext(os.path.basename(src))[0].replace(f'{obj[:2]}_{fam}_', '')
    tag = f'{obj}_spl_{fam}_00'
    c = ctx(src, f'{OUTD}/{tag}_regions.npy')
    g, names, cols = RULES[obj](c)
    g[~c['fg']] = 255
    np.save(f'{OUTD}/{tag}_gt.npy', g)
    fr = {names[k]: round(float((g == k).mean()), 4) for k in range(len(names))}
    index[tag] = dict(src=src, classes=names, fractions=fr)
    print(f'{tag}: {fr}', flush=True)
    pal = np.array(cols, np.uint8)
    vis = np.full(g.shape + (3,), 255, np.uint8)
    for k in range(len(names)): vis[g == k] = pal[k]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    ax[0].imshow(c['im']); ax[1].imshow(vis); ax[2].imshow(c['im']); ax[2].imshow(vis, alpha=0.45)
    for a, t in zip(ax, ['photograph', 'ground truth', 'overlay']): a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    h = [plt.Rectangle((0, 0), 1, 1, fc=pal[k] / 255, ec='0.4') for k in range(len(names))]
    ax[1].legend(h, names, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.20), ncol=4, frameon=False)
    plt.tight_layout(); plt.savefig(f'{OUTD}/{tag}_gt.png', dpi=130); plt.close(fig)
json.dump(index, open(f'{OUTD}/index.json', 'w'), indent=1)
