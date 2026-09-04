import math,torch,torch.nn as nn,torch.nn.functional as F
def gn(c):
    for k in (32,16,8,4,2,1):
        if c%k==0: return nn.GroupNorm(k,c)
def emb(t,d):
    h=d//2; f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/h)
    a=t.float()[:,None]*f[None]; return torch.cat([a.sin(),a.cos()],1)
class RB(nn.Module):
    def __init__(s,i,o,e):
        super().__init__()
        s.n1=gn(i); s.c1=nn.Conv2d(i,o,3,padding=1); s.e=nn.Linear(e,o)
        s.n2=gn(o); s.c2=nn.Conv2d(o,o,3,padding=1)
        s.sk=nn.Conv2d(i,o,1) if i!=o else nn.Identity()
        nn.init.zeros_(s.c2.weight); nn.init.zeros_(s.c2.bias)
    def forward(s,x,e):
        h=s.c1(F.silu(s.n1(x)))+s.e(e)[:,:,None,None]
        return s.sk(x)+s.c2(F.silu(s.n2(h)))
class UNet2D(nn.Module):
    def __init__(s,c=64,mult=(1,2,4)):
        super().__init__()
        e=c*4; s.c=c; s.e=nn.Sequential(nn.Linear(c,e),nn.SiLU(),nn.Linear(e,e))
        chs=[c*m for m in mult]; s.inp=nn.Conv2d(3,chs[0],3,padding=1)
        s.down=nn.ModuleList(); s.ds=nn.ModuleList(); prev=chs[0]
        for i,ch in enumerate(chs):
            s.down.append(RB(prev,ch,e)); prev=ch
            s.ds.append(nn.Conv2d(ch,ch,3,stride=2,padding=1) if i<len(chs)-1 else nn.Identity())
        s.up=nn.ModuleList(); s.us=nn.ModuleList()
        for i,ch in reversed(list(enumerate(chs))):
            s.us.append(nn.Identity() if i==len(chs)-1 else nn.Upsample(scale_factor=2,mode="nearest"))
            s.up.append(RB(prev+ch,ch,e)); prev=ch
        s.on=gn(prev); s.out=nn.Conv2d(prev,3,3,padding=1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s,x,t):
        e=s.e(emb(t,s.c)); h=s.inp(x); sk=[]
        for blk,d in zip(s.down,s.ds): h=blk(h,e); sk.append(h); h=d(h)
        for u,blk in zip(s.us,s.up):
            h=u(h); k=sk.pop()
            if h.shape[-1]!=k.shape[-1]: h=F.interpolate(h,size=k.shape[-2:],mode="nearest")
            h=blk(torch.cat([h,k],1),e)
        return s.out(F.silu(s.on(h)))
