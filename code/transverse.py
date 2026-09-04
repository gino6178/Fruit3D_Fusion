# The transverse view, which the longitudinal scorer never looked at: tex15 on transverse slices
# NOTE: spokes@NAZ does not separate the 90/rev artifact from anatomy at 128 px (Nyquist); the artifact was
# confirmed and its removal confirmed by eye on the composite. Chroma orders correctly but the photograph
# scores mid. Needs the native 256 px and the true NAZ to be a detector.
# against the horizontal photographs, and azimuthal periodicity -- spokes from per-plane
# disagreement of the longitudinal family show up here and nowhere else.
import sys,glob,numpy as np,torch
from PIL import Image
from planes import Vol
from fidelity import radial,photos
import os
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("DATA",os.path.join(os.path.dirname(ROOT),"data","orange"))
def photos(fam):   # every photograph of one family, training and held out alike
    import glob as _g; return sorted(_g.glob(f"{DATA}/spl_{fam}/*.png"))+sorted(_g.glob(f"{DATA}/hld_{fam}/*.png"))

vol=Vol(sys.argv[1],"cpu"); g=torch.load(sys.argv[1],map_location="cpu")
hs=vol.occupied_heights(); H=[float(hs[int(len(hs)*q)]) for q in (0.35,0.5,0.65)]
def to_np(t):
    t=t.squeeze(0); a=((t.clamp(-1,1)*0.5+0.5)).permute(1,2,0).numpy()
    return np.asarray(Image.fromarray((a*255).astype(np.uint8)).resize((128,128),Image.LANCZOS)).astype(np.float32)/255
NAZ=90
def azprof(ch,m,th,nb=360):
    b=((th[m]+np.pi)/(2*np.pi)*nb).astype(int)%nb; return np.bincount(b,ch[m],nb)/np.maximum(np.bincount(b,None,nb),1)
def spokes(a):  # artifact, not anatomy: power at the NAZ frequency of the azimuthal profile (segments sit at ~10/rev), and chroma
    n=128; yy,xx=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n),indexing="ij")
    r=np.sqrt(xx**2+yy**2)/0.82; th=np.arctan2(yy,xx); m=(r>0.28)&(r<0.57)
    L=azprof(a.mean(2),m,th); F=np.abs(np.fft.rfft(L-L.mean()))
    naz=F[NAZ-2:NAZ+3].max()/ (F[5:40].mean()+1e-9)          # NAZ-band peak relative to the anatomy band
    C=azprof(a[...,0]-a[...,1],m,th); C2=azprof(a[...,1]-a[...,2],m,th)
    return naz, float(np.sqrt(C.var()+C2.var()))
cols=[("O-Voxel",[to_np(vol.trans_slice(g["V"].float(),h)) for h in H])]
for p in sys.argv[2:]:
    X=torch.load(p,map_location="cpu")["X"].float(); cols.append((p.split("/")[-2],[to_np(vol.trans_slice(X,h)) for h in H]))
ph=[np.asarray(Image.open(p).convert("RGB").resize((128,128),Image.LANCZOS)).astype(np.float32)/255 for p in photos("trans")]
cols.append(("photo",ph))
print("  transverse view:")
print("  "+" ".join(f"{l[:14]:>14s}" for l,_ in cols))
for lo,hi,name in ((0,0.14,"columella"),(0.28,0.57,"flesh"),(0.57,0.71,"pith band")):
    print(f"  tex15 {name:10s}"+" ".join(f"{np.mean([radial(a,lo,hi) for a in arr]):14.4f}" for _,arr in cols))
sp=[[spokes(a) for a in arr] for _,arr in cols]
print(f"  {'spokes@NAZ':16s}"+" ".join(f"{np.mean([v[0] for v in x]):14.3f}" for x in sp))
print(f"  {'az chroma':16s}"+" ".join(f"{np.mean([v[1] for v in x]):14.4f}" for x in sp))
