# Fidelity against the REAL photographs, measured on the flesh only.
#
# The raw Laplacian of the O-Voxel field (0.2197 on a slice) is not sharpness -- it is voxel
# noise, twice what a real photograph carries.  Every method that "lost sharpness" was removing
# that.  So the reference is the photographs' own statistics, and distance to them is the score.
import os,sys,glob,numpy as np,torch
from PIL import Image
from scipy.ndimage import uniform_filter
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get("DATA",os.path.join(os.path.dirname(ROOT),"data","orange"))
def photos(fam):   # every photograph of one family, training and held out alike
    import glob as _g; return sorted(_g.glob(f"{DATA}/spl_{fam}/*.png"))+sorted(_g.glob(f"{DATA}/hld_{fam}/*.png"))

def canon(p,res):
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255
    m=a.min(2)<0.92; ys,xs=np.where(m)
    cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
    half=max(ys.max()-ys.min(),xs.max()-xs.min())/2/0.82
    y0,y1,x0,x1=int(cy-half),int(cy+half),int(cx-half),int(cx+half)
    pad=max(0,-y0,-x0,y1-a.shape[0],x1-a.shape[1])
    if pad: a=np.pad(a,((pad,pad),(pad,pad),(0,0)),constant_values=1.0); y0+=pad;y1+=pad;x0+=pad;x1+=pad
    return np.asarray(Image.fromarray((a[y0:y1,x0:x1]*255).astype(np.uint8)).resize((res,res),Image.LANCZOS)).astype(np.float32)/255
def flesh(a):
    h,w,_=a.shape; yy,xx=np.mgrid[0:h,0:w]; r=np.sqrt((yy-h/2)**2+(xx-w/2)**2)/(h/2)
    return (r<0.78)&(a.min(2)<0.92)
def stats(a):
    m=flesh(a); g=a.mean(2)
    lap=np.abs(g[1:-1,1:-1]*4-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:])
    def lstd(k): return np.sqrt(np.maximum(uniform_filter(g*g,k)-uniform_filter(g,k)**2,0))
    return dict(lap=lap[m[1:-1,1:-1]].mean(), sat=(a[...,0]-a[...,2])[m].mean(),
                tex5=lstd(5)[m].mean(), tex15=lstd(15)[m].mean())
KEYS=["lap","sat","tex5","tex15"]
def real(fam,res=128):
    d="secref_orraw_vsep" if fam=="long" else "secref_orraw_hsep"
    ph=[stats(canon(p,res)) for p in sorted(glob.glob(f"{DATA}/{d}/*.png"))]
    return {k:float(np.mean([x[k] for x in ph])) for k in KEYS}
def dist(d,ref): return float(np.sqrt(np.mean([((d[k]-ref[k])/max(abs(ref[k]),1e-6))**2 for k in KEYS])))
def radial(a,rlo,rhi):
    h,w,_=a.shape; yy,xx=np.mgrid[0:h,0:w]; r=np.sqrt((yy-h/2)**2+(xx-w/2)**2)/(h/2)
    m=(r>=rlo)&(r<rhi)&(a.min(2)<0.92); g=a.mean(2)
    return float(np.sqrt(np.maximum(uniform_filter(g*g,15)-uniform_filter(g,15)**2,0))[m].mean())
BANDS=((0,0.14,"columella"),(0.28,0.57,"flesh"),(0.57,0.71,"pith band"))
if __name__=="__main__":
    # Every candidate is pulled through vol.long_slice into the SAME canonical frame the photographs
    # are canonicalised to.  A grid slice and a canonical plane put the same radial band on
    # different rings, and comparing them once cost a wrong conclusion.
    sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
    from planes import Vol
    import math
    dv="cuda:1" if torch.cuda.is_available() else "cpu"
    vol=Vol(sys.argv[1],dv); THS=[0.7,1.4,2.1]
    def canon_of(X):
        out=[]
        for t in THS:
            a=((vol.long_slice(X.to(dv),t).clamp(-1,1)*0.5+0.5)).permute(1,2,0).cpu().numpy()
            out.append(np.asarray(Image.fromarray((a*255).astype(np.uint8)).resize((128,128),Image.LANCZOS)).astype(np.float32)/255)
        return out
    photos=[canon(p,128) for p in photos("long")]
    cols=[("real",photos),("O-Voxel",canon_of(vol.V0))]
    for p in sys.argv[2:]:
        cols.append((os.path.basename(os.path.dirname(p)),canon_of(torch.load(p,map_location="cpu")["X"].float())))
    print("  tex15 by radius, canonical frame:")
    print("  "+" ".join(f"{l[:14]:>14s}" for l,_ in cols))
    for rlo,rhi,name in BANDS:
        print(f"  {name:10s}"+" ".join(f"{np.mean([radial(a,rlo,rhi) for a in arr]):14.4f}" for _,arr in cols))
