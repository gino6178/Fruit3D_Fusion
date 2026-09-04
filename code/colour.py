# Flesh colour of every candidate in the canonical frame -- tex15 is blind to a colour shift.
import sys,glob,numpy as np,torch,colorsys
from PIL import Image
from planes import Vol
import os
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("DATA",os.path.join(os.path.dirname(ROOT),"data","orange"))
def photos(fam):   # every photograph of one family, training and held out alike
    import glob as _g; return sorted(_g.glob(f"{DATA}/spl_{fam}/*.png"))+sorted(_g.glob(f"{DATA}/hld_{fam}/*.png"))

vol=Vol(sys.argv[1],"cpu"); g=torch.load(sys.argv[1],map_location="cpu")
def flesh(imgs):
    px=[]
    for t in imgs:
        H=t.shape[1]; yy,xx=torch.meshgrid(torch.linspace(-1,1,H),torch.linspace(-1,1,H),indexing="ij")
        r=torch.sqrt(xx**2+yy**2)/0.82; m=(r>0.28)&(r<0.57); px.append(t[:,m].T.numpy())
    px=np.concatenate(px); hsv=np.array([colorsys.rgb_to_hsv(*p) for p in px[::37]])
    return px.mean(0),hsv[:,1].mean(),hsv[:,2].mean()
def vi(X): return [(vol.long_slice(X,th).squeeze(0)+1)/2 for th in (0.7,1.4,2.1)]
rows=[("O-Voxel",vi(g["V"].float()))]+[(p.split("/")[-2],vi(torch.load(p,map_location="cpu")["X"].float())) for p in sys.argv[2:]]
rows.append(("photo",[torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256,256)))/255.).permute(2,0,1).float() for p in photos("long")]))
print(f"{'':14s} {'R':>6s} {'G':>6s} {'B':>6s} | {'sat':>5s} {'val':>5s}")
for n,ims in rows:
    m,s,v=flesh(ims); print(f"{n:14s} {m[0]:6.3f} {m[1]:6.3f} {m[2]:6.3f} | {s:5.3f} {v:5.3f}")
