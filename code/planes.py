# One implementation of "cut a plane out of the volume" and "write a plane back", shared by the
# trainer and the sampler.  Keeping two copies is how the last animation of this method ended up
# showing something the method never produced.
import math,torch,torch.nn.functional as F
class Vol:
    AXD=1                       # polar axis is grid dim 1 (lattice up = world Y)
    def __init__(s,path,dev,res=256,target=0.82):
        G=torch.load(path,map_location=dev)   # grids ship as fp16 + bool; everything downstream is float32
        s.V0=G["V"].to(dev).float(); s.OCC=G["OCC"].to(dev).float()
        s.SHELL=G["SHELL"].to(dev).float(); s.CORE=G["CORE"].to(dev).float()
        s.N=G["N"]; s.dev=dev; s.res=res
        N=s.N; lin=torch.arange(N,device=dev).float(); c=(N-1)/2
        A,B,C=torch.meshgrid(lin,lin,lin,indexing="ij")
        import os as _o; axd=int(_o.environ.get("AXD",str(s.AXD)))   # polar axis = this grid dim (data declaration)
        s.AXD=axd; ax=[A-c,B-c,C-c]; s.v=ax[axd]; s.u0,s.u1=[ax[d] for d in range(3) if d!=axd]; s.c=c
        rad=torch.sqrt(s.u0**2+s.u1**2)
        m=s.OCC[0]>0.5
        s.EXT=max(float(rad[m].max()),float(s.v[m].abs().max()))/target
        gl=torch.linspace(-s.EXT,s.EXT,res,device=dev)
        s.GV,s.GU=torch.meshgrid(gl,gl,indexing="ij")
    def _samp(s,X,p0,p1,p2):
        N=s.N
        q={s.AXD:p1}; rest=[d for d in range(3) if d!=s.AXD]; q[rest[0]]=p0; q[rest[1]]=p2
        gg=torch.stack([q[2]/(N-1)*2-1,q[1]/(N-1)*2-1,q[0]/(N-1)*2-1],-1)[None,None]
        return F.grid_sample(X[None],gg,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0]
    def long_slice(s,X,th):
        c_,s_=math.cos(th),math.sin(th)
        return s._samp(X, s.GU*c_+s.c, s.GV+s.c, s.GU*s_+s.c)
    def trans_slice(s,X,h):
        return s._samp(X, s.GU+s.c, torch.full_like(s.GU,float(h)), s.GV+s.c)
    def _write(s,X,dist,uu,vv,img,w,slabw):
        m=(dist<3*slabw)&(s.CORE[0]>0.5)
        if int(m.sum())==0: return X
        # grid_sample takes (x, y) and the gather put RADIAL on the image's columns, so the
        # in-plane coordinate must come first.  Reversed, it writes the section transposed --
        # invisible in a round-trip on a near-symmetric object, caught by a ramp test.
        import os as _o
        gg=(torch.stack([uu[m]/s.EXT,vv[m]/s.EXT],-1) if _o.environ.get('WORDER','uv')=='uv'
            else torch.stack([vv[m]/s.EXT,uu[m]/s.EXT],-1))[None,None]
        val=F.grid_sample(img[None],gg,mode="bilinear",padding_mode="border",align_corners=True)[0,:,0,:]
        wt=(w*torch.exp(-(dist[m]/slabw)**2)).clamp(0,1)
        X[:,m]=X[:,m]*(1-wt)+val*wt
        return X
    def write_long(s,X,th,img,w,slabw=2.0):
        c_,s_=math.cos(th),math.sin(th)
        return s._write(X,(-s_*s.u0+c_*s.u1).abs(), c_*s.u0+s_*s.u1, s.v, img,w,slabw)
    def write_trans(s,X,h,img,w,slabw=2.0):
        return s._write(X,(s.v-(h-s.c)).abs(), s.u0, s.u1, img,w,slabw)
    def occupied_heights(s):
        return [j for j in range(s.N) if float(s.OCC[0].select(s.AXD,j).sum())>50]
