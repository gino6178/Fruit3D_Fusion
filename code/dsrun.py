# run in the score env: DreamSim to the nearest held-out photograph, per family, per arm
import sys,os,glob,torch
from PIL import Image
from dreamsim import dreamsim
SP=os.path.dirname(os.path.abspath(__file__)); dev="cuda:1"
OD=os.environ.get("OBJDIR",SP)
model,pre=dreamsim(pretrained=True,device=dev,cache_dir=os.environ.get("DREAMSIM_CACHE",f"{OD}/dreamsim_ckpt"))
def ds(refs,paths):
    if not refs or not paths: return float('nan')
    R=torch.cat([pre(Image.open(p).convert("RGB")).to(dev) for p in refs]); out=[]
    for p in paths:
        X=pre(Image.open(p).convert("RGB")).to(dev)
        with torch.no_grad(): out.append(min(float(model(X,R[j:j+1])) for j in range(len(R))))
    return sum(out)/len(out)
arms=open(f"{OD}/ds_faces/arms.txt").read().split("\n")
print(f"  {'held-out DreamSim':22s} {'long':>8s} {'trans':>8s} {'mean':>8s}   (lower is better)")
# the real floor: spl photographs against the held-out ones, same path
for fam,sd in (("long","spl_long"),("trans","spl_trans")):
    pass
fl=ds(sorted(glob.glob(f"{OD}/ds_faces/_ref/long/*.png")),sorted(glob.glob(f"{OD}/spl_long/*.png")))
ft=ds(sorted(glob.glob(f"{OD}/ds_faces/_ref/trans/*.png")),sorted(glob.glob(f"{OD}/spl_trans/*.png")))
print(f"  {'real (spl vs hld)':22s} {fl:8.4f} {ft:8.4f} {(fl+ft)/2:8.4f}")
for a in arms:
    l=ds(sorted(glob.glob(f"{OD}/ds_faces/_ref/long/*.png")),sorted(glob.glob(f"{OD}/ds_faces/{a}/long/*.png")))
    t=ds(sorted(glob.glob(f"{OD}/ds_faces/_ref/trans/*.png")),sorted(glob.glob(f"{OD}/ds_faces/{a}/trans/*.png")))
    print(f"  {a:22s} {l:8.4f} {t:8.4f} {(l+t)/2:8.4f}")
