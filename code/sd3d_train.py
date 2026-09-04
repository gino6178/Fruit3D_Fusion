# SinDiffusion on ONE 3-D object: the single sample is the volume, and every training example is
# a plane cut out of it -- a random azimuth for the longitudinal family, a random height for the
# transverse one.  The network is 2-D and its receptive field is deliberately smaller than the
# plane, so it learns the object's patch statistics rather than memorising a section.
import os,sys,math,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD)
exec(open(SPD+"/sd2d_net.py").read())
from planes import Vol
dv=os.environ.get("DEV","cuda:1")
FAM=os.environ.get("FAM","long")
vol=Vol(os.environ["GRID"],dv,res=int(os.environ.get("RES","256")))
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
MCH=int(os.environ.get("MCH","64"))
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2").split(","))
STEPS=int(os.environ.get("STEPS","50")); BS=int(os.environ.get("BS","4"))
OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
m=UNet2D(MCH,MULT).to(dv)
print(f"  {FAM}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params, mult {MULT}, "
      f"plane {vol.res}^2",flush=True)
opt=torch.optim.AdamW(m.parameters(),float(os.environ.get("LR","2e-4")))
g=torch.Generator(device=dv).manual_seed(0)
H=vol.occupied_heights()
# The shell is observed, so at inference it is never noised: the prior is handed a plane whose
# rind is clean and whose interior is noise.  Train it on exactly that, or the input is out of
# distribution -- measured: a prior trained on fully-noised planes restores a masked one to flat
# brown, while the same prior handles a fully-noised plane at the same level cleanly.
MASKED=int(os.environ.get("MASKED","1"))
# The O-Voxel is a FIT, so a prior trained only on its planes can never look more real than the
# fit does -- it reproduces the fit's smoothness.  The photographs are the only real appearance
# there is, so mix them in: PMIX of each batch is drawn from them, canonicalised to the same
# framing, with a shell mask derived from the radius the volume's own skin sits at (0.81 of R).
import glob as _glob
from PIL import Image as _Image
PMIX=float(os.environ.get("PMIX","0.5"))
PDIR=os.environ.get("PDIR","")
SHELL_R=float(os.environ.get("SHELL_R","0.78"))
def _load_photos(d,res,target=0.82):
    out=[]
    for p in sorted(_glob.glob(d+"/*.png")):
        a=np.asarray(_Image.open(p).convert("RGB")).astype(np.float32)/255
        m=a.min(2)<0.92
        ys,xs=np.where(m)
        if len(ys)<100: continue
        cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
        half=max(ys.max()-ys.min(),xs.max()-xs.min())/2/target
        y0,y1,x0,x1=int(cy-half),int(cy+half),int(cx-half),int(cx+half)
        pad=max(0,-y0,-x0,y1-a.shape[0],x1-a.shape[1])
        if pad:
            a=np.pad(a,((pad,pad),(pad,pad),(0,0)),constant_values=1.0)
            y0+=pad;y1+=pad;x0+=pad;x1+=pad
        im=np.asarray(_Image.fromarray((a[y0:y1,x0:x1]*255).astype(np.uint8))
                      .resize((res,res),_Image.LANCZOS)).astype(np.float32)/255
        t=torch.from_numpy(im).permute(2,0,1).to(dv)*2-1
        # interior = inside the section and inside SHELL_R of its radius
        gg=torch.linspace(-1,1,res,device=dv)
        Y,X=torch.meshgrid(gg,gg,indexing="ij")
        rr=torch.sqrt(X**2+Y**2)/target
        sec=(t.min(0).values<0.84)
        mk=((rr<SHELL_R)&sec).float()[None].expand(3,-1,-1).contiguous()
        out.append((t,mk))
    return out
PHOTOS=_load_photos(PDIR,vol.res) if PDIR else []
print(f"  photographs mixed in: {len(PHOTOS)} from {PDIR or '(none)'}, PMIX={PMIX}",flush=True)
C3=vol.CORE.expand(3,-1,-1,-1).contiguous()
def draw():
    im,mk=[],[]
    for _ in range(BS):
        if PHOTOS and float(torch.rand(1,device=dv,generator=g))<PMIX:
            t,k=PHOTOS[int(torch.randint(len(PHOTOS),(1,),device=dv,generator=g))]
            if float(torch.rand(1,device=dv,generator=g))<0.5: t,k=t.flip(-1),k.flip(-1)
            im.append(t); mk.append(k)
        elif FAM=="long":
            th=float(torch.rand(1,device=dv,generator=g))*math.pi
            im.append(vol.long_slice(vol.V0,th)); mk.append((vol.long_slice(C3,th)>0.5).float())
        else:
            j=H[int(torch.randint(len(H),(1,),device=dv,generator=g))]
            im.append(vol.trans_slice(vol.V0,j)); mk.append((vol.trans_slice(C3,j)>0.5).float())
    return torch.stack(im),torch.stack(mk)
t0=time.time()
for it in range(1,STEPS+1):
    x0,MK=draw()
    if not MASKED: MK=torch.ones_like(MK)
    t=torch.randint(T,(BS,),device=dv,generator=g)
    eps=torch.randn(x0.shape,device=dv,generator=g); a_=ab[t].view(-1,1,1,1)
    xt=x0*(1-MK)+(a_.sqrt()*x0+(1-a_).sqrt()*eps)*MK      # clean outside the mask
    pred=m(xt,t)
    # only the masked region carries a noise target; outside it there is no epsilon to predict
    loss=((pred-eps)**2*MK).sum()/MK.sum().clamp(min=1)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if it%max(1,STEPS//5)==0 or it==1:
        print(f"    {it}/{STEPS}  loss {float(loss.detach()):.4f}  {time.time()-t0:.0f}s",flush=True)
torch.save(m.state_dict(),f"{OUT}/model.pt")
print("TRAIN_DONE",flush=True)
