# Voxelise OUR O-Voxel result -- trained/orange.ply from the v1-inputs release, 1,162,387 cells --
# not the FruitNinja released model that build_orange/ was quantised from.  Colour is the trained
# f_dc here, and cell_level marks the skin (level 1, 480,287 cells).
import os,numpy as np,torch,torch.nn.functional as F
SP=os.path.dirname(os.path.abspath(__file__))
PLY=os.environ.get("PLY",SP+"/wt/trained/orange.ply")
META=os.environ.get("META",SP+"/wt/lattice_meta/orange")
N=int(os.environ.get("N","128")); dv=os.environ.get("DEV","cuda:1")
C0=0.28209479177387814
f=open(PLY,"rb"); hdr=b""
while b"end_header" not in hdr: hdr+=f.readline()
names=[l.split()[-1].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
n=int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
a=np.frombuffer(f.read(n*len(names)*4),dtype="<f4").reshape(n,len(names))
xyz=torch.from_numpy(a[:,[names.index(k) for k in ("x","y","z")]].copy()).to(dv)
rgb=torch.from_numpy((0.5+C0*a[:,[names.index(f"f_dc_{i}") for i in range(3)]]).clip(0,1).copy()).to(dv)
lv=torch.load(META+"/cell_level.pt",map_location=dv)
shell=(lv==1)
print(f"  {n:,} cells: {int((~shell).sum()):,} interior, {int(shell.sum()):,} skin",flush=True)
lo=xyz.min(0).values; hi=xyz.max(0).values; ctr=(lo+hi)/2; ext=float((hi-lo).max())*1.02
idx=(((xyz-ctr)/ext+0.5)*(N-1)).round().long().clamp(0,N-1)
flat=(idx[:,0]*N+idx[:,1])*N+idx[:,2]
acc=torch.zeros(N**3,3,device=dv); cnt=torch.zeros(N**3,1,device=dv)
acc.index_add_(0,flat,rgb); cnt.index_add_(0,flat,torch.ones(len(flat),1,device=dv))
shl=torch.zeros(N**3,1,device=dv); shl.index_add_(0,flat,shell[:,None].float())
occ=(cnt[:,0]>0)
V=torch.ones(N**3,3,device=dv); V[occ]=acc[occ]/cnt[occ]
SH=torch.zeros(N**3,device=dv); SH[occ]=(shl[occ,0]/cnt[occ,0])
V=V.reshape(N,N,N,3).permute(3,0,1,2).contiguous()
OCC=occ.reshape(1,N,N,N).float()
for _ in range(int(os.environ.get("FILL","3"))):
    nb=F.avg_pool3d(OCC[None],3,1,1)[0]; cv=F.avg_pool3d((V*OCC)[None],3,1,1)[0]
    add=((OCC<0.5)&(nb>0.35)).float()
    V=torch.where(add>0, cv/nb.clamp(min=1e-6), V); OCC=(OCC+add).clamp(max=1)
SHELL=(SH.reshape(1,N,N,N)>0.5).float()*OCC
SHELL=(F.max_pool3d(SHELL[None],3,1,1)[0]>0.5).float()*OCC
CORE=OCC*(1-SHELL)
print(f"  grid {N}^3: occupied {int(OCC.sum()):,} ({float(OCC.mean())*100:.1f}%), "
      f"shell {int(SHELL.sum()):,} ({float(SHELL.sum()/OCC.sum())*100:.1f}% of solid), "
      f"interior {int(CORE.sum()):,}",flush=True)
torch.save({"V":(V*2-1).clamp(-1,1).cpu(),"OCC":OCC.cpu(),"SHELL":SHELL.cpu(),"CORE":CORE.cpu(),
            "ctr":ctr.cpu(),"ext":ext,"N":N}, os.environ["OUT"])
from PIL import Image
Image.fromarray((V[:,N//2].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))\
     .resize((420,420),Image.NEAREST).save(os.environ["OUT"]+".png")
print("  saved",os.environ["OUT"],flush=True)
