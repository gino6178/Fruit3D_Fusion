# Held-out DreamSim on canonical-frame cut faces.  Every candidate goes through the same path: grid
# slice (vol.long_slice / trans_slice) at RES, canonicalised exactly like the photographs (fidelity.canon
# on a white background), saved as PNG, scored as mean over faces of the DreamSim distance to the
# NEAREST held-out photograph (realism._dreamsim).  Grid slices, not the O-Voxel renderer: the same
# deviation from the remote protocol for every candidate, so the ranking is fair, the absolute
# numbers are not comparable to the remote table.
import sys,os,glob,math,numpy as np,torch
from PIL import Image
from planes import Vol
from fidelity import canon
SP=os.path.dirname(os.path.abspath(__file__)); RES=512
OD=os.environ.get("OBJDIR",SP)   # where spl_*/hld_* live; faces go under it
vol=Vol(sys.argv[1],"cpu",res=RES); g=torch.load(sys.argv[1],map_location="cpu")
THS=[k*math.pi/6 for k in range(6)]; hs=vol.occupied_heights(); HS=[float(hs[int(len(hs)*q)]) for q in (0.3,0.4,0.5,0.6,0.7)]
def face(t):
    t=(t.squeeze(0).clamp(-1,1)+1)/2; return Image.fromarray((t.permute(1,2,0).numpy()*255).astype(np.uint8))
def dump(name,X):
    d=f"{OD}/ds_faces/{name}"; os.makedirs(d+"/long",exist_ok=True); os.makedirs(d+"/trans",exist_ok=True)
    for i,th in enumerate(THS): face(vol.long_slice(X,th)).save(f"{d}/long/{i}.png")
    for i,h in enumerate(HS): face(vol.trans_slice(X,h)).save(f"{d}/trans/{i}.png")
    for fam in ("long","trans"):   # canonicalise like the photographs
        for p in glob.glob(f"{d}/{fam}/*.png"): Image.fromarray((canon(p,RES)*255).astype(np.uint8)).save(p)
    return d
arms=[("O-Voxel",g["V"].float())]+[(os.path.basename(os.path.dirname(p)) if p.endswith("state.pt") else os.path.basename(p).replace(".pt",""),
        torch.load(p,map_location="cpu")[ "X" if p.endswith("state.pt") else "V"]) for p in sys.argv[2:]]
dirs=[(n,dump(n,X)) for n,X in arms]
for fam,hd in (("long","hld_long"),("trans","hld_trans")):
    os.makedirs(f"{OD}/ds_faces/_ref/{fam}",exist_ok=True)
    for p in glob.glob(f"{OD}/{hd}/*.png"): Image.fromarray((canon(p,RES)*255).astype(np.uint8)).save(f"{OD}/ds_faces/_ref/{fam}/"+os.path.basename(p))
open(f"{OD}/ds_faces/arms.txt","w").write("\n".join(n for n,_ in dirs))
print("faces written for",[n for n,_ in dirs])
