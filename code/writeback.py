# Write the lifted interior back into the O-Voxel asset.  Every interior cell (cell_level != 1)
# takes the volume's colour at its centre, trilinear; every skin cell keeps its trained colour.
# The PLY is rewritten byte-for-byte except f_dc_0..2 of interior cells, so anything that reads
# the original (the renderer, the lattice loader) reads this one.
import os,sys,numpy as np,torch,torch.nn.functional as F
SP=os.path.dirname(os.path.abspath(__file__)); C0=0.28209479177387814
PLY,META,GRID,STATE,OUT=sys.argv[1:6]; dv="cpu"
f=open(PLY,"rb"); hdr=b""
while b"end_header" not in hdr: hdr+=f.readline()
names=[l.split()[-1].decode() for l in hdr.split(b"\n") if l.startswith(b"property")]
n=int([l for l in hdr.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
a=np.frombuffer(f.read(n*len(names)*4),dtype="<f4").reshape(n,len(names)).copy()
xyz=torch.from_numpy(a[:,[names.index(k) for k in ("x","y","z")]])
lv=torch.load(META+"/cell_level.pt",map_location=dv); interior=(lv!=1)
g=torch.load(GRID,map_location=dv); N=g["N"]; ctr=g["ctr"]; ext=g["ext"]
X=torch.load(STATE,map_location=dv)["X"].float()                       # (3,N,N,N) in [-1,1], same grid as voxelize_ov.py
idx=((xyz-ctr)/ext+0.5)*(N-1)                                          # continuous grid index, the inverse of the voxeliser's rounding
gg=torch.stack([idx[:,2],idx[:,1],idx[:,0]],-1)/(N-1)*2-1              # grid_sample wants (x=W=dim2, y=H=dim1, z=D=dim0)
rgb=F.grid_sample(X[None],gg[None,None,None],mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,0,:].T
rgb=(rgb.clamp(-1,1)+1)/2
fdc=((rgb-0.5)/C0).numpy().astype("<f4")
cols=[names.index(f"f_dc_{i}") for i in range(3)]
m=interior.numpy()
before=a[m][:,cols].copy(); a[m,cols[0]]=fdc[m,0]; a[m,cols[1]]=fdc[m,1]; a[m,cols[2]]=fdc[m,2]
open(OUT,"wb").write(hdr+a.astype("<f4").tobytes())
print(f"  {n:,} cells, {int(m.sum()):,} interior rewritten, {int((~m).sum()):,} skin untouched; mean |dcolour| {float(np.abs(a[m][:,cols]-before).mean()*C0):.4f}",flush=True)
