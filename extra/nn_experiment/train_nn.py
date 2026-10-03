"""Small causal Transformer SE model, trained on a NOIZEUS train/test split
that is disjoint by sentence (not just by noise realisation), so that the
Theorem-1 context-truncation test below is a genuine held-out evaluation.

Split: sentences 1-20 (all 4 SNRs, all 7 noise types) = train (560 utt.).
       sentences 21-30 (SNR=15 dB, all 7 noise types)  = test  (70 utt.).
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import measure as m

SR=8000; NFFT=256; HOP=128; NF=NFFT//2+1
DEV=torch.device('cpu'); torch.manual_seed(0); np.random.seed(0)
CKPT=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt.pt')
HIST=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'history.json')

def stft_t(x):
    x=torch.as_tensor(x,dtype=torch.float32)
    win=torch.hann_window(NFFT)
    return torch.stft(x,NFFT,HOP,window=win,return_complex=True,center=False).T  # (T,NF)

class TinyCausalTF(nn.Module):
    def __init__(self,d=64,h=2,L=2,ff=128):
        super().__init__()
        self.proj=nn.Linear(NF,d)
        enc=nn.TransformerEncoderLayer(d_model=d,nhead=h,dim_feedforward=ff,
                                        dropout=0.1,batch_first=True)
        self.tf=nn.TransformerEncoder(enc,num_layers=L)
        self.out=nn.Linear(d,NF)
    def forward(self,x,C=None):
        B,T,_=x.shape
        m_=torch.full((T,T),float('-inf'))
        idx=torch.arange(T)
        causal = idx[None,:] <= idx[:,None]
        if C is None:
            allow = causal
        else:
            allow = causal & (idx[None,:] > idx[:,None]-C)
        m_=torch.where(allow,torch.zeros(T,T),torch.full((T,T),float('-inf')))
        h=self.proj(x); h=self.tf(h,mask=m_)
        return torch.sigmoid(self.out(h))

def load_split():
    """Sentence-disjoint 3-way split (no sentence appears in more than one
    split): fit 1-16, val 17-20, test 21-30 (held out, matches Table I SNR)."""
    fit=[]; val=[]; test=[]
    for nt in m.TYPES:
        for k in range(1,17):
            for snr in [0,5,10,15]:
                fit.append((snr,k,nt))
        for k in range(17,21):
            val.append((15,k,nt))
        for k in range(21,31):
            test.append((15,k,nt))
    return fit, val, test

def get_xyz(snr,k,nt):
    s,x,v=m.load(snr,f'sp{k:02d}',nt)
    S=stft_t(s); X=stft_t(x)
    Ps=(S.abs()**2); Pn=((X-S).abs()**2)
    g=(Ps/(Ps+Pn+1e-8)).clamp(0,1)
    lm=torch.log(X.abs()+1e-8)
    return lm, g, X, s

def istft_np(Y, length):
    """Overlap-add ISTFT matching the numpy convention used by the other
    baselines (Hann analysis window, no synthesis window, weight-summed)."""
    win = np.hanning(NFFT).astype(np.float32)
    T = Y.shape[0]
    out = np.zeros(T*HOP + NFFT, np.float32)
    wsum = np.zeros_like(out)
    for i in range(T):
        seg = (np.fft.irfft(Y[i], NFFT).astype(np.float32)) * win
        s0 = i*HOP
        out[s0:s0+NFFT] += seg
        wsum[s0:s0+NFFT] += win**2
    wsum[wsum < 1e-8] = 1.0
    return (out/wsum)[:length]

def evaluate(model,items,C=None):
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
    from segsnr_metrics import compute_delta_segsnr as cds
    model.eval(); out=[]
    with torch.no_grad():
        for snr,k,nt in items:
            lm,g,X,s = get_xyz(snr,k,nt)
            mask=model(lm.unsqueeze(0),C=C).squeeze(0)
            Y=(X*mask).numpy()
            y=istft_np(Y,len(s))
            n=min(len(s),len(y))
            d=cds(s[:n],m.load(snr,f'sp{k:02d}',nt)[1][:n],y[:n],fs=SR)
            out.append((nt,k,d))
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--val_every',type=int,default=5)
    args=ap.parse_args()
    fit,val,test=load_split()
    model=TinyCausalTF().to(DEV)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    start=1; hist=[]; best_val=-1e9
    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=DEV,weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start=ck['epoch']+1; hist=ck['hist']; best_val=ck.get('best_val',-1e9)
        print(f'resumed at epoch {start}, best_val so far {best_val:.2f}')
    nparam=sum(p.numel() for p in model.parameters())
    print('params', nparam)
    crit=nn.MSELoss()
    for ep in range(start, start+args.epochs):
        model.train(); t0=time.time(); losses=[]
        order=np.random.permutation(len(fit))
        for idx in order:
            snr,k,nt=fit[idx]
            lm,g,X,s=get_xyz(snr,k,nt)
            opt.zero_grad()
            pred=model(lm.unsqueeze(0)).squeeze(0)
            loss=crit(pred,g)
            loss.backward(); opt.step()
            losses.append(loss.item())
        ml=float(np.mean(losses))
        rec=dict(epoch=ep,loss=ml,time=time.time()-t0)
        if ep%args.val_every==0 or ep==start:
            vres=evaluate(model,val,C=None)
            vmean=float(np.mean([r[2] for r in vres]))
            rec['val_dsegsnr']=vmean
            print(f'epoch {ep:3d} loss {ml:.5f}  val {vmean:+.2f}dB  {time.time()-t0:.1f}s')
            if vmean>best_val:
                best_val=vmean
                torch.save(dict(model=model.state_dict(),epoch=ep,val=vmean),'best.pt')
                print('  [new best]')
        else:
            print(f'epoch {ep:3d} loss {ml:.5f}  {time.time()-t0:.1f}s')
        hist.append(rec)
        torch.save(dict(model=model.state_dict(),opt=opt.state_dict(),epoch=ep,hist=hist,best_val=best_val),CKPT)
    json.dump(hist,open(HIST,'w'),indent=1)
