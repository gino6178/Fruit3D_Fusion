# 3-D SinDiffusion from two slice families, in cylindrical coordinates.  The state is ONE array
# A[r, phi, z]; a longitudinal plane at phi_k is the exact (s,z) slice made of columns k and
# k+NPHI/2, a transverse plane at z_j is the exact (r,phi) polar slice.  Both families denoise the
# SAME x_t, no interpolation anywhere in the chain, and the product prior's step is one line:
# x0 = w(r) x0_L + (1-w(r)) x0_T, the longitudinal leading at the axis (columella) and the
# transverse away from it (membranes, which in polar coordinates are vertical stripes a patch
# prior can learn).  O-Voxel: shell never written; its low-pass is pinned per longitudinal plane
# (ILVR).  The asset is written once at the end (cylinder -> voxels).
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
from PIL import Image
dv=os.environ.get("DEV","cuda:1"); vol=Vol(os.environ["GRID"],dv,res=256); N=vol.N; E=vol.EXT; c=vol.c
NR,NPHI,NZ=128,int(os.environ.get("NPHI","512")),256; K=NPHI//2
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
T0H=float(os.environ.get("T0H","0.5")); T0V=float(os.environ.get("T0V","0.9")); NSTEP=int(os.environ.get("NSTEP","100"))
R0=float(os.environ.get("R0","0.15")); WFAR=float(os.environ.get("WFAR","0.3")); DCFIX=float(os.environ.get("DCFIX","16")); BS=int(os.environ.get("BS","8"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
def load(p,mult):
    m=UNet2D(64,mult).to(dv); d=torch.load(p,map_location=dv); m.load_state_dict(d if "sd" not in d else d["sd"]); m.eval(); return m
HAVE_L=os.environ.get("CKV","none")!="none"; HAVE_T=os.environ.get("CKH","none")!="none"   # a family without photographs is inactive
MV=load(os.environ["CKV"],tuple(int(x) for x in os.environ.get("MULTV","1,2").split(","))) if HAVE_L else None
MT=load(os.environ["CKH"],tuple(int(x) for x in os.environ.get("MULTH","1,2,4").split(","))) if HAVE_T else None
g=torch.Generator(device=dv).manual_seed(0)
# ---- cylinder geometry, sampled from the voxel grid once ----
r=(torch.arange(NR,device=dv)+0.5)/NR*E; ph=torch.arange(NPHI,device=dv)/NPHI*2*math.pi; z=-E+(torch.arange(NZ,device=dv)+0.5)/NZ*2*E
RR,PP,ZZ=torch.meshgrid(r,ph,z,indexing="ij")
p0=(RR*torch.cos(PP)+c).reshape(1,-1); p1=(ZZ+c).reshape(1,-1); p2=(RR*torch.sin(PP)+c).reshape(1,-1)
A0=vol._samp(vol.V0,p0,p1,p2).reshape(3,NR,NPHI,NZ)
CORE=(vol._samp(vol.CORE.expand(3,-1,-1,-1).contiguous(),p0,p1,p2).reshape(3,NR,NPHI,NZ)[:1]>0.5).float()
print(f"  cylinder {NR}x{NPHI}x{NZ}, interior {int(CORE.sum()):,} cells",flush=True)
wL=(WFAR+(1-WFAR)*torch.exp(-((RR[:,:,:]/E)/R0)**2))[None]     # longitudinal weight: 1 on the axis -> WFAR far out
# ---- exact slicing ----
def toL(x):  # (3,NR,NPHI,NZ) -> (K,3,NZ,2NR): rows z, cols s=-E..E   (matches vol.long_slice layout)
    a=x[:,:,:K,:]; b=x[:,:,K:,:].flip(1)                       # s>0 from column k, s<0 from column k+K reversed
    return torch.cat([b,a],1).permute(2,3,1,0)                 # (K, 3, NZ, 2NR) -- wait: dims (3,2NR,K,NZ)->(K,NZ,2NR,3)? fix below
def toL(x):
    a=x[:,:,:K,:]; b=x[:,:,K:,:].flip(1); s=torch.cat([b,a],1)   # (3, 2NR, K, NZ)
    return s.permute(2,0,3,1).contiguous()                      # (K, 3, NZ, 2NR)
def fromL(L):  # inverse
    s=L.permute(1,3,0,2)                                        # (3, 2NR, K, NZ)
    return torch.cat([s[:,NR:,:,:], s[:,:NR,:,:].flip(1)],2)    # columns k (s>0) then k+K (s<0 reversed)
def toT(x): return x.permute(3,0,1,2).contiguous()              # (NZ, 3, NR, NPHI)
def fromT(Tt): return Tt.permute(1,2,3,0)
assert torch.equal(fromL(toL(A0)),A0) and torch.equal(fromT(toT(A0)),A0)
def lowpassL(L,sig):
    k=int(sig*3)*2+1; w=torch.exp(-(torch.arange(k,device=dv,dtype=L.dtype)-k//2)**2/(2*sig*sig)); w=w/w.sum()
    out=[]
    for i in range(0,len(L),64):
        x=L[i:i+64]; x=F.conv2d(F.pad(x,(k//2,k//2,0,0),mode="reflect"),w.view(1,1,1,k).repeat(3,1,1,1),groups=3)
        out.append(F.conv2d(F.pad(x,(0,0,k//2,k//2),mode="reflect"),w.view(1,1,k,1).repeat(3,1,1,1),groups=3))
    return torch.cat(out)
LPL=lowpassL(toL(A0),DCFIX) if DCFIX>0 else None
def eps(model,x,t):
    out=[]
    for k in range(0,len(x),BS):
        with torch.no_grad(): out.append(model(x[k:k+BS],torch.full((len(x[k:k+BS]),),t,device=dv,dtype=torch.long)))
    return torch.cat(out)
# ---- the chain on one shared state ----
iT=max(int((T0V if HAVE_L else T0H)*(T-1)),1); iTH=int(T0H*(T-1)); ts=[int(round(v)) for v in np.linspace(iT,0,NSTEP+1)]
efix=torch.randn(A0.shape,device=dv,generator=g)                # the outside (shell, background) is noised with a fixed epsilon, never written
x=ab[iT].sqrt()*A0+(1-ab[iT]).sqrt()*efix; t0=time.time()
for i,(tc,tn) in enumerate(zip(ts[:-1],ts[1:])):
    aT=HAVE_T and tc<=iTH; aL=HAVE_L
    if aL: eL=eps(MV,toL(x),tc); x0L=fromL((toL(x)-(1-ab[tc]).sqrt()*eL)/ab[tc].sqrt())
    if aT: eT=eps(MT,toT(x),tc); x0T=fromT((toT(x)-(1-ab[tc]).sqrt()*eT)/ab[tc].sqrt())
    if aL and aT: x0=wL*x0L+(1-wL)*x0T
    elif aL: x0=x0L
    elif aT: x0=x0T
    else: x0=A0
    if LPL is not None: x0=fromL(toL(x0)-lowpassL(toL(x0),DCFIX)+LPL)     # ILVR on the longitudinal planes: the O-Voxel's layout and colour
    x0=CORE*x0+(1-CORE)*A0
    if tn<=0: break
    ebar=(x-ab[tc].sqrt()*x0)/(1-ab[tc]).sqrt()
    x=ab[tn].sqrt()*x0+(1-ab[tn]).sqrt()*ebar
    x=CORE*x+(1-CORE)*(ab[tn].sqrt()*A0+(1-ab[tn]).sqrt()*efix)
    if i%10==0: print(f"  step {i}/{NSTEP} t={tc} T={aT} {time.time()-t0:.0f}s",flush=True)
A=(CORE*x0+(1-CORE)*A0).clamp(-1,1)
# ---- write the asset once: voxels <- cylinder (trilinear in r,phi,z with phi wrap) ----
u0,u1,v=vol.u0,vol.u1,vol.v; rad=torch.sqrt(u0**2+u1**2); ang=torch.atan2(u1,u0)%(2*math.pi)
Ap=torch.cat([A,A[:,:,:1,:]],2)                                                     # phi wrap
gr=(rad/E*NR-0.5)/(NR-1)*2-1; gp=(ang/(2*math.pi)*NPHI)/(NPHI)*2-1; gz=((v+E)/(2*E)*NZ-0.5)/(NZ-1)*2-1
m=vol.CORE[0]>0.5; grid=torch.stack([gp[m],gz[m],gr[m]],-1)[None,None,None]         # grid_sample 3D takes (x=W, y=H, z=D) = (phi, z, r)
val=F.grid_sample(Ap.permute(0,1,3,2)[None],grid,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0,:]
X=vol.V0.clone(); X[:,m]=val
torch.save({"X":X.cpu(),"A":A.cpu()},OUT+"/state.pt")
def im(t): t=(t.squeeze(0)+1)/2; return Image.fromarray((t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
hs=vol.occupied_heights(); tiles=[im(vol.long_slice(X,th)) for th in (0.7,1.4,2.1)]+[im(vol.trans_slice(X,float(hs[len(hs)//2])))]
W=tiles[0].width; s=Image.new("RGB",(4*W,W)); [s.paste(t,(k*W,0)) for k,t in enumerate(tiles)]; s.save(OUT+"/xfill.png")
print("  wrote",OUT,f"{time.time()-t0:.0f}s",flush=True)
