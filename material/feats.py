"""Per-pixel features shared by the photograph side and the virtual-slice side.
DINOv2-base at effective stride ~3.5 (4x4 shifted 518-px crops), LAB + 3-scale local stats (21),
Gabor 3x4 + structure-tensor coherence x2 (14).  Every block is whitened per image on the foreground,
which is what made the photograph clusters transfer across individuals."""
import numpy as np, torch, torch.nn.functional as F, cv2
dv = 'cuda'; RES = 512
_dino = None; _MEAN = _STD = None
def _load():
    global _dino, _MEAN, _STD
    if _dino is None:
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        _dino = AutoModel.from_pretrained('facebook/dinov2-base').to(dv).eval().half()
        _MEAN = torch.tensor(proc.image_mean, device=dv).view(3, 1, 1); _STD = torch.tensor(proc.image_std, device=dv).view(3, 1, 1)
SH = [0, 4, 7, 11]; P = 14; D = 518
@torch.no_grad()
def dino_feats(im):
    _load(); x = torch.from_numpy(np.ascontiguousarray(im)).to(dv).permute(2, 0, 1).float() / 255
    x = F.interpolate(x[None], size=(D, D), mode='bilinear', align_corners=False)[0]
    pad = F.pad(x, (0, P, 0, P), mode='replicate'); n = D // P
    batch = ((torch.stack([pad[:, sy:sy + D, sx:sx + D] for sy in SH for sx in SH]) - _MEAN) / _STD).half()
    tok = _dino(pixel_values=batch).last_hidden_state[:, 1:].reshape(16, n, n, 768)
    acc = torch.zeros(4 * n, 4 * n, 768, device=dv, dtype=torch.float16)
    for i in range(4):
        for j in range(4): acc[i::4, j::4] = tok[i * 4 + j]
    f = F.interpolate(acc.permute(2, 0, 1)[None].float(), size=(RES, RES), mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    return f.half()                                                       # (RES,RES,768)
def lab_feats(im, fg):
    lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32); out = [lab]; m = fg.astype(np.float32)
    for k in (5, 15, 31):
        w = cv2.blur(m, (k, k))[..., None] + 1e-6
        mu = cv2.blur(lab * m[..., None], (k, k)) / w; sq = cv2.blur(lab ** 2 * m[..., None], (k, k)) / w
        out += [mu, np.sqrt(np.clip(sq - mu ** 2, 0, None))]
    return np.concatenate(out, -1)
GABOR = [cv2.getGaborKernel((int(6 * lam) | 1,) * 2, sigma=lam * 0.6, theta=th, lambd=lam, gamma=0.7, psi=0)
         for lam in (6, 12, 24) for th in np.linspace(0, np.pi, 4, endpoint=False)]
def tex_feats(im, fg):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255; g = g * fg + (g[fg].mean() if fg.any() else 0.5) * (~fg)
    out = [np.abs(cv2.filter2D(g, -1, k)) for k in GABOR]
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    for s in (3, 9):
        jxx, jyy, jxy = [cv2.GaussianBlur(a, (0, 0), s) for a in (gx * gx, gy * gy, gx * gy)]
        d = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2); out.append(d / (jxx + jyy + 1e-6))
    return np.stack(out, -1)
def whiten(x, fg, stats=None):
    """standardise each channel; stats=(mu,sd) fixed, else on the foreground of THIS image"""
    if stats is None: v = x[fg]; mu = v.mean(0); sd = v.std(0) + 1e-6
    else: mu, sd = stats
    return (x - mu) / sd
def raw_feats(im, fg, pca, mu):
    d = (dino_feats(im).float() - mu) @ pca; d = d.cpu().numpy()
    return np.concatenate([d, lab_feats(im, fg), tex_feats(im, fg)], -1).astype(np.float32)
def fg_stats(x, fg): v = x[fg]; return (v.mean(0), v.std(0) + 1e-6)
def all_feats(im, fg, pca, mu, stats=None):
    """-> (RES,RES, 64+21+14) float32.  stats=None: whitened on this image's foreground (photographs,
    which are all central cuts).  stats=(mu,sd): fixed whitening, used for the asset's virtual slices so
    that a slice near the pole is not re-normalised on a foreground that is almost all peel."""
    return whiten(raw_feats(im, fg, pca, mu), fg, stats)
def fg_mask(im):
    g = im.astype(np.float32).mean(2); m = (g < 245) & (im.astype(np.float32).std(2) > 4)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1: m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return m.astype(bool)
def radius_map(fg, family):
    """normalised radius per pixel in the photograph's canonical frame: transverse -> distance to centroid / max;
    longitudinal (axis vertical) -> |x - cx| / half-width"""
    ys, xs = np.nonzero(fg); cy, cx = ys.mean(), xs.mean()
    Y, X = np.mgrid[0:fg.shape[0], 0:fg.shape[1]]
    if family == 'trans':
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2); return (r / np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max()).astype(np.float32)
    r = np.abs(X - cx); return (r / np.abs(xs - cx).max()).astype(np.float32)
