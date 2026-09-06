"""SAM region proposals for many photographs at once (loads the model once).
usage: propose_batch.py OUT_DIR img1.png img2.png ...
Output per image: <OUT_DIR>/<tag>_regions.npy and _regions.png, tag = <obj>_<split>_<fam>_<name>."""
import os, sys, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feats import fg_mask, RES
from transformers import pipeline
from PIL import Image
from scipy import ndimage
OUTD, SRCS = sys.argv[1], sys.argv[2:]; os.makedirs(OUTD, exist_ok=True)
PPS = int(os.environ.get('PPS', 64))
gen = pipeline('mask-generation', model='facebook/sam-vit-huge', device=0, points_per_batch=64)

def tag_of(p):
    parts = p.rstrip('/').split('/'); nm = os.path.splitext(parts[-1])[0]
    return f'{parts[-3]}_{parts[-2]}_{nm}'

def run(SRC, OUT):
    im = cv2.resize(cv2.cvtColor(cv2.imread(SRC), cv2.COLOR_BGR2RGB), (RES, RES), interpolation=cv2.INTER_AREA)
    fg = fg_mask(im)
    out = gen(Image.fromarray(im), points_per_side=PPS, pred_iou_thresh=0.5, stability_score_thresh=0.6)
    masks = [m for m in (np.array(m) & fg for m in out['masks']) if m.sum() > 250]
    masks.sort(key=lambda m: -m.sum())
    part = np.full(fg.shape, -1, np.int16)
    for i, m in enumerate(masks): part[m] = i
    if (fg & (part < 0)).any():
        idx = ndimage.distance_transform_edt(part < 0, return_distances=False, return_indices=True); part = np.where(fg & (part < 0), part[tuple(idx)], part)
    part[~fg] = -1
    final = np.full(fg.shape, -1, np.int16); nid = 0
    for i in np.unique(part[part >= 0]):
        cc, k = ndimage.label(part == i)
        for c in range(1, k + 1):
            m = cc == c
            if m.sum() < 250: continue
            final[m] = nid; nid += 1
    if (fg & (final < 0)).any():
        idx = ndimage.distance_transform_edt(final < 0, return_distances=False, return_indices=True); final = np.where(fg & (final < 0), final[tuple(idx)], final)
    final[~fg] = -1
    np.save(OUT + '_regions.npy', final)
    rng = np.random.default_rng(0); cols = rng.integers(60, 240, (max(nid, 1), 3))
    vis = np.full((RES, RES, 3), 255, np.uint8)
    for i in range(nid): vis[final == i] = cols[i]
    ov = im.copy()
    for i in range(nid):
        m = final == i; c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, c, -1, (255, 0, 0), 1)
        ys, xs = np.nonzero(m); cy, cx = int(ys.mean()), int(xs.mean())
        for img in (vis, ov):
            cv2.putText(img, str(i), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3); cv2.putText(img, str(i), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
    ax[0].imshow(im); ax[1].imshow(vis); ax[2].imshow(ov)
    for a, t in zip(ax, ['photograph', f'{nid} regions', 'boundaries + ids']): a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(OUT + '_regions.png', dpi=130); plt.close(fig)
    areas = {int(i): round(float((final == i).mean()), 4) for i in range(nid)}
    print(f'{os.path.basename(OUT)}: {nid} regions; areas {areas}', flush=True)

for s in SRCS:
    run(s, os.path.join(OUTD, tag_of(s)))
