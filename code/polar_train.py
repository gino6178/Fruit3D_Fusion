import os,sys,glob,time,numpy as np,torch,torch.nn as nn
SPD=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SPD); os.chdir(SPD)
exec(open(SPD+"/sd2d_net.py").read())
from PIL import Image
dv=os.environ.get("DEV","cuda:1"); PDIR=os.environ["PDIR"]; OUT=os.environ["OUT"]; os.makedirs(OUT,exist_ok=True)
MULT=tuple(int(x) for x in os.environ.get("MULT","1,2,4").split(",")); STEPS=int(os.environ.get("STEPS","8000")); BS=int(os.environ.get("BS","4")); LR=float(os.environ.get("LR","5e-4"))
T=1000; ab=torch.cumprod(1-torch.linspace(1e-4,0.02,T,device=dv),0)
imgs=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/127.5-1).permute(2,0,1) for p in sorted(glob.glob(PDIR+"/*.png"))]).to(dv)
print("  polar strips",tuple(imgs.shape),flush=True)
m=UNet2D(64,MULT).to(dv); opt=torch.optim.AdamW(m.parameters(),LR); t0=time.time()
for it in range(1,STEPS+1):
    idx=torch.randint(0,len(imgs),(BS,),device=dv); x0=imgs[idx]
    x0=torch.stack([torch.roll(x,int(torch.randint(0,x.shape[-1],(1,))),dims=-1) for x in x0])      # phi-roll
    if torch.rand(1)<0.5: x0=x0.flip(-1)                                                                # mirror in phi
    t=torch.randint(0,T,(BS,),device=dv); e=torch.randn_like(x0); xt=ab[t].sqrt()[:,None,None,None]*x0+(1-ab[t]).sqrt()[:,None,None,None]*e
    loss=((m(xt,t)-e)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if it%1000==0 or it==1: print(f"    {it}/{STEPS}  loss {float(loss):.4f}  {time.time()-t0:.0f}s",flush=True)
torch.save(m.state_dict(),f"{OUT}/model.pt"); print("POLAR_DONE",flush=True)
