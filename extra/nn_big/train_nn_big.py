"""Larger-scale version of the trained-causal-network experiment
(Section V-G), using MS-SNSD instead of NOIZEUS as the training corpus:
~2700 training mixtures (150 speech clips x 9 noise types x 2 SNRs,
each cropped to ~3s) versus the original 448, and a somewhat larger
model (d=96, 3 layers, ~230k params versus 83k).

Split is by speech-clip content (sentence-disjoint), same principle as
train_nn.py: fit / val / test clips are three non-overlapping subsets of
MS-SNSD's clean_test pool (fixed seed).
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'msnsd_validate'))
import soundfile as sf
import msnsd_validate as mv

SR = 16000; NFFT = 512; HOP = 256; NF = NFFT // 2 + 1
DEV = torch.device('cpu'); torch.manual_seed(0); np.random.seed(0)
CROP_S = 3.0
CROP_N = int(CROP_S * SR)
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, 'ckpt_big.pt')

N_FIT, N_VAL, N_TEST = 150, 20, 20
TRAIN_SNRS = [0, 15]
EVAL_SNR = 15


def split_clips(seed=0):
    import glob
    files = sorted(glob.glob(os.path.join(mv.ROOT, 'clean_test', 'clnsp*.wav')))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    fit = [files[i] for i in idx[:N_FIT]]
    val = [files[i] for i in idx[N_FIT:N_FIT + N_VAL]]
    test = [files[i] for i in idx[N_FIT + N_VAL:N_FIT + N_VAL + N_TEST]]
    return fit, val, test


def stft_t(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    win = torch.hann_window(NFFT)
    return torch.stft(x, NFFT, HOP, window=win, return_complex=True, center=False).T


def istft_np(Y, length):
    win = np.hanning(NFFT).astype(np.float32)
    T = Y.shape[0]
    out = np.zeros(T * HOP + NFFT, np.float32)
    wsum = np.zeros_like(out)
    for i in range(T):
        seg = (np.fft.irfft(Y[i], NFFT).astype(np.float32)) * win
        s0 = i * HOP
        out[s0:s0 + NFFT] += seg
        wsum[s0:s0 + NFFT] += win ** 2
    wsum[wsum < 1e-8] = 1.0
    return (out / wsum)[:length]


class CausalTF(nn.Module):
    def __init__(self, d=96, h=4, L=3, ff=192):
        super().__init__()
        self.proj = nn.Linear(NF, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=ff,
                                          dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=L)
        self.out = nn.Linear(d, NF)

    def forward(self, x, C=None):
        B, T, _ = x.shape
        idx = torch.arange(T)
        causal = idx[None, :] <= idx[:, None]
        allow = causal if C is None else (causal & (idx[None, :] > idx[:, None] - C))
        mask = torch.where(allow, torch.zeros(T, T), torch.full((T, T), float('-inf')))
        h_ = self.proj(x); h_ = self.tf(h_, mask=mask)
        return torch.sigmoid(self.out(h_))


def det_pair_seed(sp_id, cat, snr, crop_i):
    return mv.det_seed(sp_id, cat, snr, crop_i)


def make_mixture(sf_path, cat, pools, snr_db, crop_seed, crop=True):
    s, fs = sf.read(sf_path)
    assert fs == SR
    sp_id = os.path.basename(sf_path).replace('.wav', '')
    x, v = mv.mix_at_snr(s, pools[cat], snr_db, seed=mv.det_seed(sp_id, cat, str(snr_db)))
    if crop and len(s) > CROP_N:
        rng = np.random.default_rng(crop_seed)
        start = rng.integers(0, len(s) - CROP_N)
        s = s[start:start + CROP_N]; x = x[start:start + CROP_N]; v = v[start:start + CROP_N]
    return s, x, v, sp_id


def get_xyz(sf_path, cat, pools, snr_db, crop_seed, crop=True):
    s, x, v, sp_id = make_mixture(sf_path, cat, pools, snr_db, crop_seed, crop)
    S = stft_t(s); X = stft_t(x)
    Ps = S.abs() ** 2; Pn = (X - S).abs() ** 2
    g = (Ps / (Ps + Pn + 1e-8)).clamp(0, 1)
    lm = torch.log(X.abs() + 1e-8)
    return lm, g, X, s, x, v, sp_id


def build_index(clips, cats, snrs, crop=True):
    idx = []
    for sp_path in clips:
        for cat in cats:
            for snr in snrs:
                idx.append((sp_path, cat, snr))
    return idx


def evaluate(model, clips, pools, cats, snr, C=None):
    sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
    from segsnr_metrics import compute_delta_segsnr
    model.eval(); out = []
    with torch.no_grad():
        for sp_path in clips:
            for cat in cats:
                lm, g, X, s, x, v, sp_id = get_xyz(sp_path, cat, pools, snr, crop_seed=0, crop=False)
                mask = model(lm.unsqueeze(0), C=C).squeeze(0)
                Y = (X * mask).numpy()
                y = istft_np(Y, len(s))
                n = min(len(s), len(y))
                d = compute_delta_segsnr(s[:n], x[:n], y[:n], fs=SR)
                out.append((cat, sp_id, d))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--val_every', type=int, default=2)
    args = ap.parse_args()

    pools = mv.load_noise_pools()
    fit, val, test = split_clips(seed=0)
    train_index = build_index(fit, mv.CATS, TRAIN_SNRS)
    print('train mixtures:', len(train_index), 'val clips:', len(val), 'test clips:', len(test))

    model = CausalTF().to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print('params', nparam)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start = 1; hist = []; best_val = -1e9
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start = ck['epoch'] + 1; hist = ck['hist']; best_val = ck.get('best_val', -1e9)
        print(f'resumed at epoch {start}, best_val so far {best_val:.2f}')
    crit = nn.MSELoss()
    for ep in range(start, start + args.epochs):
        model.train(); t0 = time.time(); losses = []
        order = np.random.permutation(len(train_index))
        for ii, idx in enumerate(order):
            sp_path, cat, snr = train_index[idx]
            lm, g, X, s, x, v, sp_id = get_xyz(sp_path, cat, pools, snr, crop_seed=ep * 100000 + idx)
            opt.zero_grad()
            pred = model(lm.unsqueeze(0)).squeeze(0)
            loss = crit(pred, g)
            loss.backward(); opt.step()
            losses.append(loss.item())
        ml = float(np.mean(losses))
        rec = dict(epoch=ep, loss=ml, time=time.time() - t0)
        if ep % args.val_every == 0 or ep == start:
            vres = evaluate(model, val, pools, mv.CATS, EVAL_SNR, C=None)
            vmean = float(np.mean([r[2] for r in vres]))
            rec['val_dsegsnr'] = vmean
            print(f'epoch {ep:3d} loss {ml:.5f}  val {vmean:+.2f}dB  {time.time() - t0:.1f}s')
            if vmean > best_val:
                best_val = vmean
                torch.save(dict(model=model.state_dict(), epoch=ep, val=vmean), os.path.join(HERE, 'best_big.pt'))
                print('  [new best]')
        else:
            print(f'epoch {ep:3d} loss {ml:.5f}  {time.time() - t0:.1f}s')
        hist.append(rec)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep,
                         hist=hist, best_val=best_val), CKPT)
    json.dump(hist, open(os.path.join(HERE, 'history_big.json'), 'w'), indent=1)
