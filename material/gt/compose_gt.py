"""Turn SAM region proposals + an explicit region->material assignment into a ground-truth map.

The assignment lives in assign.json and is written by a human reading the numbered figure. Nothing
here guesses: a region id absent from the assignment is left unlabelled (255) and reported, so an
incomplete GT is visible rather than silently wrong.

  assign.json = { "<tag>": { "classes": [...names...], "map": {"0": 0, "1": 2, ...} }, ... }

usage: compose_gt.py GT_DIR assign.json [tag ...]"""
import os, sys, json, glob, numpy as np, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
GTD, ASSIGN = sys.argv[1], sys.argv[2]; TAGS = sys.argv[3:]
A = json.load(open(ASSIGN))
PAL = np.array([[236, 148, 40], [250, 245, 230], [232, 108, 60], [250, 250, 250], [110, 90, 70],
                [70, 140, 60], [200, 40, 50], [30, 30, 30], [180, 160, 120], [120, 60, 140]], np.uint8)
for tag in (TAGS or sorted(A)):
    if tag not in A: print(f'{tag}: no assignment, skipped'); continue
    spec = A[tag]; names = spec['classes']; mp = {int(k): int(v) for k, v in spec['map'].items()}
    pal = np.array(spec['colours'], np.uint8) if 'colours' in spec else PAL
    reg = np.load(f'{GTD}/{tag}_regions.npy')
    gt = np.full(reg.shape, 255, np.uint8)
    ids = [int(i) for i in np.unique(reg[reg >= 0])]
    for i in ids:
        if i in mp: gt[reg == i] = mp[i]
    np.save(f'{GTD}/{tag}_gt.npy', gt)
    missing = [i for i in ids if i not in mp]
    fr = {names[c]: round(float((gt == c).mean()), 4) for c in sorted(set(mp.values()))}
    print(f'{tag}: {fr}' + (f'  UNASSIGNED regions {missing}' if missing else ''), flush=True)
    src = spec.get('src')
    if src and os.path.exists(src):
        im = cv2.resize(cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB), gt.shape[::-1], interpolation=cv2.INTER_AREA)
    else: im = np.zeros(gt.shape + (3,), np.uint8)
    vis = np.full(gt.shape + (3,), 255, np.uint8)
    for c in range(len(names)): vis[gt == c] = pal[c % len(pal)]
    vis[gt == 255] = (255, 0, 255)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
    ax[0].imshow(im); ax[1].imshow(vis); ax[2].imshow(im); ax[2].imshow(vis, alpha=0.45)
    for a, t in zip(ax, ['photograph', 'ground truth', 'overlay']): a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    h = [plt.Rectangle((0, 0), 1, 1, fc=pal[c % len(pal)] / 255) for c in range(len(names))]
    ax[1].legend(h, names, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout(); plt.savefig(f'{GTD}/{tag}_gt.png', dpi=130); plt.close(fig)
