# A 3DGS point cloud (FruitNinja, GaussianFluent) through the same path as our carrier: every
# Gaussian centre lands in a voxel with its f_dc colour, voxels average, holes close by the same
# rule.  No shell is known for a Gaussian model, so SHELL is empty and CORE = OCC; the grid is
# only read for cut faces.  Centres with opacity below OPA are dropped (they paint nothing).
import os,sys,numpy as np,torch,torch.nn.functional as F
PLY=sys.argv[1]; OUT=sys.argv[2]; N=int(os.environ.get("N","128")); dv=os.environ.get("DEV","cuda:1"); OPA=float(os.environ.get("OPA","0.1")); C0=0.28209479177387814
f=open(PLY,"rb"); hdr=b""
while b"end_header" not in hdr: hdr+=f.readline()
names=[l.split()[-1].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
n=int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
a=np.frombuffer(f.read(n*len(names)*4),dtype="<f4").reshape(n,len(names))
xyz=torch.from_numpy(a[:,[names.index(k) for k in ("x","y","z")]].copy()).to(dv)
rgb=torch.from_numpy((0.5+C0*a[:,[names.index(f"f_dc_{i}") for i in range(3)]]).clip(0,1).copy()).to(dv)
if "opacity" in names:
    op=torch.sigmoid(torch.from_numpy(a[:,names.index("opacity")].copy()).to(dv)); keep=op>OPA; xyz,rgb=xyz[keep],rgb[keep]
print(f"  {n:,} gaussians, {len(xyz):,} kept (opacity>{OPA})",flush=True)
lo=xyz.min(0).values; hi=xyz.max(0).values; ctr=(lo+hi)/2; ext=float((hi-lo).max())*1.02
idx=(((xyz-ctr)/ext+0.5)*(N-1)).round().long().clamp(0,N-1); flat=(idx[:,0]*N+idx[:,1])*N+idx[:,2]
acc=torch.zeros(N**3,3,device=dv); cnt=torch.zeros(N**3,1,device=dv); acc.index_add_(0,flat,rgb); cnt.index_add_(0,flat,torch.ones(len(flat),1,device=dv))
occ=cnt[:,0]>0; V=torch.ones(N**3,3,device=dv); V[occ]=acc[occ]/cnt[occ]
V=V.reshape(N,N,N,3).permute(3,0,1,2).contiguous(); OCC=occ.reshape(1,N,N,N).float()
for _ in range(int(os.environ.get("FILL","3"))):
    nb=F.avg_pool3d(OCC[None],3,1,1)[0]; cv=F.avg_pool3d((V*OCC)[None],3,1,1)[0]; add=((OCC<0.5)&(nb>0.35)).float()
    V=torch.where(add>0,cv/nb.clamp(min=1e-6),V); OCC=(OCC+add).clamp(max=1)
print(f"  grid {N}^3: occupied {int(OCC.sum()):,} ({float(OCC.mean())*100:.1f}%)",flush=True)
torch.save({"V":(V*2-1).clamp(-1,1).cpu(),"OCC":OCC.cpu(),"SHELL":torch.zeros_like(OCC).cpu(),"CORE":OCC.cpu(),"ctr":ctr.cpu(),"ext":ext,"N":N},OUT)
