"""MS-SNSD (16 kHz) counterparts of oracle_null.py and controlled_general.py.

  null   stationary phase-randomised surrogate of each mixture's noise:
         T_real / T_null for K-frame oracles and the leave-out oracle
  ctrl   gamma-envelope modulation (64 ms segments) of each of the nine
         noise categories: T vs c0*nu_eff and the exact Jensen gap, B_c, B_s

Mixtures are the letter's (same clips, crops and seeds as msnsd_validate.py).
Usage: MSNSD_DIR=... python msnsd_extra.py null|ctrl [snr]
"""
import os, sys, numpy as np, pandas as pd, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'msnsd_validate')); sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import msnsd_validate as mv
from mmse_wiener_baseline import stft
from scipy.linalg import toeplitz, solve
from scipy.ndimage import uniform_filter1d
from scipy.stats import gamma as gamma_dist

NFFT = 512; HOP = 256; SEG = 1024; L = 128
KS = [1, 3, 5, 9, 17]; NSUR = 5
NUS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def surrogate(v, seed):
    V = np.fft.rfft(v)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(V)); ph[0] = 0
    return np.fft.irfft(np.abs(V) * np.exp(1j * ph), len(v))


def gaps(s, v):
    S = stft(s, NFFT, HOP); V = stft(v, NFFT, HOP); X = S + V
    Ss = np.mean(np.abs(S) ** 2, 0); P = np.abs(V) ** 2; Sb = P.mean(0)
    mse_lti = np.mean(np.abs(S - Ss / (Ss + Sb + 1e-12) * X) ** 2)
    out = {}
    for K in KS:
        Sl = uniform_filter1d(P, K, axis=0)
        out[f'K{K}'] = 10 * np.log10(mse_lti / np.mean(np.abs(S - Ss / (Ss + Sl + 1e-12) * X) ** 2))
    Pp = np.pad(P, ((2, 2), (0, 0)), mode='edge'); Sl = 0.5 * (Pp[:-4] + Pp[4:])
    out['loo'] = 10 * np.log10(mse_lti / np.mean(np.abs(S - Ss / (Ss + Sl + 1e-12) * X) ** 2))
    return out


def null(s, v, sp, cat, snr):
    if len(v) % 2: s = s[:-1]; v = v[:-1]
    row = dict(nsi=mv.nsi_std(v))
    for k, val in gaps(s, v).items(): row[f'T_{k}'] = val
    nul = [gaps(s, surrogate(v, mv.det_seed(sp, cat, snr, j))) for j in range(NSUR)]
    for k in nul[0]: row[f'N_{k}'] = float(np.mean([n[k] for n in nul]))
    return [row]


def ctrl(s, v0, sp, cat, snr):
    b = v0 / np.sqrt(np.mean(v0 ** 2)); win = np.hanning(NFFT)
    S = stft(s, NFFT, HOP); Ss = np.mean(np.abs(S) ** 2, 0); rows = []
    nseg = int(np.ceil(len(s) / SEG))
    u = np.random.default_rng(mv.det_seed(sp, 'A')).uniform(1e-6, 1 - 1e-6, nseg)
    for nu in NUS:
        a = np.ones(nseg) if nu == 0 else gamma_dist.ppf(u, a=1.0 / nu, scale=nu)
        a = np.repeat(a, SEG)[:len(s)]; a = a / a.mean()
        v = np.sqrt(a) * b
        v = v * np.sqrt(np.mean(s ** 2) / 10 ** (snr / 10) / np.mean(v ** 2)); x = s + v
        g2 = np.mean(v ** 2) / np.mean(a * b ** 2)
        X = stft(x, NFFT, HOP); B = stft(np.sqrt(g2) * b, NFFT, HOP); T = S.shape[0]
        Sb = np.mean(np.abs(B) ** 2, 0)
        at = np.array([np.sum(win ** 2 * a[i*HOP:i*HOP+NFFT] * b[i*HOP:i*HOP+NFFT] ** 2) /
                       (np.sum(win ** 2 * b[i*HOP:i*HOP+NFFT] ** 2) + 1e-20) for i in range(T)])
        aS = at[:, None] * Sb[None, :]; Pn = np.mean(np.abs(S) ** 2)
        mse_c = np.mean(np.abs(S - Ss / (Ss + Sb) * X) ** 2) / Pn
        mse_nc = np.mean(np.abs(S - Ss[None, :] / (Ss[None, :] + aS) * X) ** 2) / Pn
        c0 = float(np.mean(Ss ** 2 * Sb ** 2 / (Ss + Sb) ** 3) / np.mean(Ss))
        nu_eff = float(np.var(at) / np.mean(at) ** 2)
        rxx = mv.xcorr(x, x, L); rsx = mv.xcorr(s, x, L)
        w = solve(toeplitz(rxx) + np.eye(L) * 1e-8 * rxx[0], rsx, assume_a='pos')
        rows.append(dict(nu=nu, nu_eff=nu_eff, nsi=mv.nsi_std(v), T=mse_c - mse_nc, T_2nd=c0 * nu_eff,
                         T_exact=float((np.mean(Ss * Sb / (Ss + Sb)) - np.mean(Ss[None, :] * aS / (Ss[None, :] + aS))) / np.mean(Ss)),
                         c0=c0, bs=mse_c, bc=float(np.mean(s ** 2) - rsx @ w) / np.mean(s ** 2)))
    return rows


if __name__ == '__main__':
    mode = sys.argv[1]; snr = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    pools = mv.load_noise_pools(); rows = []
    for i, path in enumerate(mv.pick_speech(mv.N_SPEECH, seed=0)):
        s, fs = sf.read(path); sp = os.path.basename(path).replace('.wav', '')
        for cat in mv.CATS:
            x, v = mv.mix_at_snr(s, pools[cat], snr, seed=mv.det_seed(sp, cat))
            for r in (null if mode == 'null' else ctrl)(s, v, sp, cat, snr):
                rows.append(dict(speaker=sp, noise_type=cat, snr=snr, dB_letter=mv.delta_b(s, x, v), **r))
        if (i + 1) % 10 == 0: print(i + 1, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f'results_msnsd_{mode}_snr{snr}.csv'), index=False)
