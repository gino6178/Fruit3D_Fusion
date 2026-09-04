# One program for every object: find its photographs, canonicalise on a white background (the
# background colour is read from the corners, so black-background photographs work), declare the
# held-out split (>=4 per family: 3 train, the rest held out; fewer: all train, flagged), and
# unwrap the transverse family to polar strips.  Root-level photographs without a family folder
# are declared once here by what they are: bread and cake slices are longitudinal, the doughnut
# slice is transverse.  No other per-object setting exists.
import os,sys,glob,shutil,numpy as np,torch,torch.nn.functional as F
from PIL import Image
SP=os.path.dirname(os.path.abspath(__file__))
# the raw photograph corpus: one directory per object, each with vertical/ and horizontal/
R=os.environ.get("PHOTOS",os.path.join(os.path.dirname(SP),"photographs"))
ROOT_FAMILY={"bread":"long","cake":"long","donut":"trans","doughnut":"trans"}
obj=sys.argv[1]; src="donut" if obj=="doughnut" else obj
O=os.environ.get("OUT",os.path.join(os.path.dirname(SP),"data",obj))
def canon_bg(p,res=512,target=0.82):
    a=np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255; h,w,_=a.shape
    corners=np.concatenate([a[:8,:8].reshape(-1,3),a[:8,-8:].reshape(-1,3),a[-8:,:8].reshape(-1,3),a[-8:,-8:].reshape(-1,3)]); bg=np.median(corners,0)
    m=np.abs(a-bg).max(2)>0.12
    ys,xs=np.where(m)
    if len(ys)<100: return None
    a=np.where(m[...,None],a,1.0)                                                # background -> white
    cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2; half=max(ys.max()-ys.min(),xs.max()-xs.min())/2/target
    y0,y1,x0,x1=int(cy-half),int(cy+half),int(cx-half),int(cx+half); pad=max(0,-y0,-x0,y1-h,x1-w)
    if pad: a=np.pad(a,((pad,pad),(pad,pad),(0,0)),constant_values=1.0); y0+=pad;y1+=pad;x0+=pad;x1+=pad
    return np.asarray(Image.fromarray((a[y0:y1,x0:x1]*255).astype(np.uint8)).resize((res,res),Image.LANCZOS)).astype(np.float32)/255
def polar(a,NR=128,NPHI=512):
    H=a.shape[0]; Rr=H/2; t=torch.from_numpy(a).permute(2,0,1)[None].float()
    r=(torch.arange(NR)+0.5)/NR*Rr; ph=torch.arange(NPHI)/NPHI*2*np.pi; rr,pp=torch.meshgrid(r,ph,indexing="ij")
    x=(H/2+rr*torch.cos(pp))/(H-1)*2-1; y=(H/2+rr*torch.sin(pp))/(H-1)*2-1
    return F.grid_sample(t,torch.stack([x,y],-1)[None],mode="bilinear",padding_mode="border",align_corners=True)[0].permute(1,2,0).numpy()
fams={"long":sorted(glob.glob(f"{R}/{src}/vertical/*.png")),"trans":sorted(glob.glob(f"{R}/{src}/horizontal/*.png"))}
root=[p for p in sorted(glob.glob(f"{R}/{src}/*.png"))]
if root and obj in ROOT_FAMILY: fams[ROOT_FAMILY[obj]]+=root
fams={k:[p for p in v if "depth" not in p] for k,v in fams.items()}
shutil.rmtree(O,ignore_errors=True); note=[]
for fam,ps in fams.items():
    for d in (f"{O}/spl_{fam}",f"{O}/hld_{fam}"): os.makedirs(d,exist_ok=True)
    imgs=[(p,canon_bg(p)) for p in ps]; imgs=[(p,a) for p,a in imgs if a is not None]
    if len(imgs)==0: note.append(f"{fam}: no photographs"); continue
    if len(imgs)>=4: spl,hld=imgs[:3],imgs[3:]; note.append(f"{fam}: {len(spl)} train, {len(hld)} held out")
    else: spl,hld=imgs,imgs; note.append(f"{fam}: {len(spl)} train, NO held-out (scored against the training photographs)")
    for k,(p,a) in enumerate(spl): Image.fromarray((a*255).astype(np.uint8)).save(f"{O}/spl_{fam}/{k:02d}.png")
    for k,(p,a) in enumerate(hld): Image.fromarray((a*255).astype(np.uint8)).save(f"{O}/hld_{fam}/{k:02d}.png")
    if fam=="trans":
        os.makedirs(f"{O}/polar_spl_trans",exist_ok=True)
        for k,(p,a) in enumerate(spl): Image.fromarray((polar(a)*255).astype(np.uint8)).save(f"{O}/polar_spl_trans/{k:02d}.png")
open(f"{O}/SPLIT.txt","w").write("\n".join(note)+"\n"); print(obj,"|",", ".join(note))
