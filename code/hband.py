# Horizontal banding on longitudinal cuts: per-transverse-plane brightness differences seen edge-on.
# Row-mean luminance over the flesh columns, detrended, std -- the row-profile detector, this time
# pointed the right way.  Canonical frame via vol.long_slice.
import sys,glob,numpy as np,torch
from PIL import Image
from planes import Vol
from fidelity import canon,photos
vol=Vol(sys.argv[1],"cpu"); g=torch.load(sys.argv[1],map_location="cpu")
def to_np(t):
    a=((t.squeeze(0).clamp(-1,1)*0.5+0.5)).permute(1,2,0).numpy()
    return np.asarray(Image.fromarray((a*255).astype(np.uint8)).resize((128,128),Image.LANCZOS)).astype(np.float32)/255
def hband(a):
    L=a.mean(2); yy,xx=np.meshgrid(np.linspace(-1,1,128),np.linspace(-1,1,128),indexing="ij"); r=np.sqrt(xx**2+yy**2)/0.82
    m=(r>0.28)&(r<0.57)&(np.abs(xx)>0.2)          # flesh, off the columella
    prof=np.array([L[i][m[i]].mean() if m[i].any() else np.nan for i in range(128)]); ok=~np.isnan(prof)
    p=prof[ok]; k=np.ones(9)/9; tr=np.convolve(np.pad(p,4,mode="edge"),k,mode="valid")
    return float((p-tr).std())
cols=[("O-Voxel",g["V"].float())]+[(p.split("/")[-2],torch.load(p,map_location="cpu")["X"].float()) for p in sys.argv[2:]]
print(f"  {'h-band (row profile)':22s}"+" ".join(f"{n[:11]:>11s}" for n,_ in cols)+f"{'photo':>11s}")
v=[np.mean([hband(to_np(vol.long_slice(X,t))) for t in (0.7,1.4,2.1)]) for _,X in cols]
ph=np.mean([hband(canon(p,128)) for p in photos("long")])
print(f"  {'':22s}"+" ".join(f"{x:11.4f}" for x in v)+f"{ph:11.4f}")
