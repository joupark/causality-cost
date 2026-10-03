"""Controlled data-scaling study.

Design: hold architecture, hyperparameters, noise categories, SNRs, and
the validation/test speech clips ALL FIXED across runs, and vary ONLY the
number of training speech clips (16, 32, 64, 100, 150). This isolates
data amount as the single manipulated variable, unlike the paper's two
trained-network comparison (Section V-G), which confounds architecture,
training corpus, and data amount simultaneously.

Held out: val (20 clips) and test (20 clips) are the SAME clips at every
training-set size, drawn once with a fixed seed and never touched by any
training run, so results across sizes are directly comparable.
"""
import os, sys, time, json, hashlib, numpy as np, torch, torch.nn as nn, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'msnsd_validate'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
import msnsd_validate as mv
from segsnr_metrics import compute_delta_segsnr

SR = 16000; NFFT = 512; HOP = 256; NF = NFFT // 2 + 1
DEV = torch.device('cpu'); torch.manual_seed(0); np.random.seed(0)
CROP_N = int(3.0 * SR)
HERE = os.path.dirname(os.path.abspath(__file__))

TRAIN_SNRS = [0, 15]; EVAL_SNR = 15
N_VAL, N_TEST = 20, 20
SIZES = [16, 32, 64, 100, 150]  # training speech-clip counts to compare


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
    """Same family as the paper's networks; d=64, nhead=4 (even, avoids
    the nested-tensor slowdown found for odd nhead), 2 layers -- small
    enough to train many configurations within budget."""
    def __init__(self, d=64, h=4, L=2, ff=128):
        super().__init__()
        self.proj = nn.Linear(NF, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=ff,
                                          dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=L)
        self.out = nn.Linear(d, NF)

    def forward(self, x):
        B, T, _ = x.shape
        idx = torch.arange(T)
        causal = idx[None, :] <= idx[:, None]
        mask = torch.where(causal, torch.zeros(T, T), torch.full((T, T), float('-inf')))
        h_ = self.proj(x); h_ = self.tf(h_, mask=mask)
        return torch.sigmoid(self.out(h_))


def fixed_splits(seed=0):
    import glob
    files = sorted(glob.glob(os.path.join(mv.ROOT, 'clean_test', 'clnsp*.wav')))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    val = [files[i] for i in idx[:N_VAL]]
    test = [files[i] for i in idx[N_VAL:N_VAL + N_TEST]]
    pool = [files[i] for i in idx[N_VAL + N_TEST:N_VAL + N_TEST + max(SIZES)]]
    return pool, val, test


def det_crop_seed(*parts):
    """Deterministic seed (Python's hash() is process-randomised for
    strings, which silently broke exact reproducibility here and in an
    earlier version of msnsd_validate.py -- see reproducibility_package
    changelog)."""
    h = hashlib.md5('_'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def make_mixture(sf_path, cat, pools, snr_db, crop_seed, crop=True):
    s, fs = sf.read(sf_path)
    assert fs == SR
    sp_id = os.path.basename(sf_path).replace('.wav', '')
    x, v = mv.mix_at_snr(s, pools[cat], snr_db, seed=mv.det_seed(sp_id, cat, str(snr_db)))
    if crop and len(s) > CROP_N:
        rng = np.random.default_rng(crop_seed)
        start = rng.integers(0, len(s) - CROP_N)
        s = s[start:start + CROP_N]; x = x[start:start + CROP_N]
    return s, x


def get_xyz(sf_path, cat, pools, snr_db, crop_seed, crop=True):
    s, x = make_mixture(sf_path, cat, pools, snr_db, crop_seed, crop)
    S = stft_t(s); X = stft_t(x)
    Ps = S.abs() ** 2; Pn = (X - S).abs() ** 2
    g = (Ps / (Ps + Pn + 1e-8)).clamp(0, 1)
    lm = torch.log(X.abs() + 1e-8)
    return lm, g, X, s, x


def evaluate(model, clips, pools, cats, snr):
    model.eval(); out = []
    with torch.no_grad():
        for sf_path in clips:
            for cat in cats:
                lm, g, X, s, x = get_xyz(sf_path, cat, pools, snr, crop_seed=0, crop=False)
                pred = model(lm.unsqueeze(0)).squeeze(0)
                Y = (X * pred).numpy()
                y = istft_np(Y, len(s))
                n = min(len(s), len(y))
                out.append(compute_delta_segsnr(s[:n], x[:n], y[:n], fs=SR))
    return float(np.mean(out))


def train_one(n_clips, pool, val, pools, epochs, seed=0, ckpt_path=None):
    torch.manual_seed(seed); np.random.seed(seed)
    train_clips = pool[:n_clips]
    index = [(sp, cat, snr) for sp in train_clips for cat in mv.CATS for snr in TRAIN_SNRS]
    model = CausalTF()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    start_ep = 1; best_val = -1e9; best_state = None
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start_ep = ck['epoch'] + 1; best_val = ck['best_val']; best_state = ck['best_state']
        print(f'  resumed at epoch {start_ep}, best_val so far {best_val:.3f}')
    for ep in range(start_ep, epochs + 1):
        model.train()
        order = np.random.permutation(len(index))
        for idx in order:
            sp, cat, snr = index[idx]
            lm, g, X, s, x = get_xyz(sp, cat, pools, snr, crop_seed=det_crop_seed(ep, sp, cat, snr))
            opt.zero_grad()
            pred = model(lm.unsqueeze(0)).squeeze(0)
            loss = crit(pred, g)
            loss.backward(); opt.step()
        vmean = evaluate(model, val, pools, mv.CATS, EVAL_SNR)
        print(f'  epoch {ep}/{epochs}  val {vmean:+.3f}')
        if vmean > best_val:
            best_val = vmean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        start_ep = ep + 1
        if ckpt_path:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep,
                             best_val=best_val, best_state=best_state), ckpt_path)
    model.load_state_dict(best_state)
    return model, best_val, (start_ep > epochs)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()

    pools = mv.load_noise_pools()
    pool, val, test = fixed_splits(seed=0)
    print(f'=== training on {args.size} clips ({args.size*9*2} mixtures), '
          f'val/test fixed at {N_VAL}/{N_TEST} clips ===')
    t0 = time.time()
    ckpt_path = os.path.join(HERE, f'ckpt_size{args.size}.pt')
    model, best_val, done = train_one(args.size, pool, val, pools, args.epochs, ckpt_path=ckpt_path)
    print(f'  ({time.time()-t0:.0f}s this call)')
    if not done:
        print('  not finished -- rerun the same command to continue')
    else:
        test_mean = evaluate(model, test, pools, mv.CATS, EVAL_SNR)
        print(f'size={args.size}  best_val={best_val:.3f}  test={test_mean:.3f}')
        res_path = os.path.join(HERE, 'scaling_results.json')
        results = json.load(open(res_path)) if os.path.exists(res_path) else {}
        results[str(args.size)] = dict(val=best_val, test=test_mean, epochs=args.epochs,
                                        n_mixtures=args.size * 9 * 2)
        json.dump(results, open(res_path, 'w'), indent=1)
        print('saved to', res_path)
