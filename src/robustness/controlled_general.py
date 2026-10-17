"""Generality of the controlled-modulation result (Sec. III-B, Part A).

The letter's Part A uses one base noise (car), one envelope law (gamma) and
one segment length (64 ms).  Lemma 1's leading term uses only the envelope
variance, and Proposition 1(i) uses only the averaged spectrum, so both
should hold unchanged across:
  base     the seven NOIZEUS noise types
  law      gamma / lognormal / two-state (on-off-like) unit-mean envelopes
  seg      32, 64, 128, 256 ms segments
One factor is varied at a time around the letter's configuration.

Besides the second-order prediction c0*nu_eff, the exact Jensen gap for the
realised envelope is evaluated,
  T_exact = mean_f[g_f(Sb)] - mean_{t,f}[g_f(a_t Sb)],
which needs no small-nu assumption.

Usage: NOIZEUS_DIR=... CM_SNR=15 python controlled_general.py
"""
import os, sys, hashlib, numpy as np, pandas as pd
from scipy.linalg import toeplitz, solve
from scipy.stats import gamma as gamma_dist, norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import measure as m
from mmse_wiener_baseline import stft

FS = 8000; L = 128; NFFT = 256; HOP = 128
SNR_DB = float(os.environ.get('CM_SNR', 15))
NUS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def dseed(*p):
    return int(hashlib.md5('_'.join(map(str, p)).encode()).hexdigest()[:8], 16)


def envelope(n, nu, k, law, seg):
    nseg = int(np.ceil(n / seg))
    u = np.random.default_rng(dseed(k, 'A')).uniform(1e-6, 1 - 1e-6, nseg)
    if nu == 0:
        a = np.ones(nseg)
    elif law == 'gamma':
        a = gamma_dist.ppf(u, a=1.0 / nu, scale=nu)
    elif law == 'lognormal':
        s2 = np.log1p(nu); a = np.exp(np.sqrt(s2) * norm.ppf(u) - s2 / 2)
    elif law == 'twostate':                       # P(a=hi)=p, a=lo otherwise; mean 1, var nu
        p = 0.2; hi = 1 + np.sqrt(nu * (1 - p) / p); lo = 1 - np.sqrt(nu * p / (1 - p))
        a = np.where(u < p, hi, max(lo, 1e-3))
    return np.repeat(a, seg)[:n]


def run(k, base, law, seg):
    s, x0, v0 = m.load(15, f'sp{k:02d}', base)
    b = v0 / np.sqrt(np.mean(v0 ** 2))
    win = np.hanning(NFFT)
    S = stft(s, NFFT, HOP); Ss = np.mean(np.abs(S) ** 2, 0)
    rows = []
    for nu in NUS:
        a = envelope(len(s), nu, k, law, seg); a = a / a.mean()
        v = np.sqrt(a) * b
        v = v * np.sqrt(np.mean(s ** 2) / 10 ** (SNR_DB / 10) / np.mean(v ** 2)); x = s + v
        g2 = np.mean(v ** 2) / np.mean(a * b ** 2)
        X = stft(x, NFFT, HOP); B = stft(np.sqrt(g2) * b, NFFT, HOP)
        T = min(S.shape[0], X.shape[0], B.shape[0])
        Sb = np.mean(np.abs(B[:T]) ** 2, 0)
        at = np.array([np.sum(win ** 2 * a[i*HOP:i*HOP+NFFT] * b[i*HOP:i*HOP+NFFT] ** 2) /
                       np.sum(win ** 2 * b[i*HOP:i*HOP+NFFT] ** 2) for i in range(T)])
        Gc = Ss / (Ss + Sb)
        Gnc = Ss[None, :] / (Ss[None, :] + at[:, None] * Sb[None, :])
        Pn = np.mean(np.abs(S[:T]) ** 2)
        mse_c = np.mean(np.abs(S[:T] - Gc * X[:T]) ** 2) / Pn
        mse_nc = np.mean(np.abs(S[:T] - Gnc * X[:T]) ** 2) / Pn
        c0 = float(np.mean(Ss ** 2 * Sb ** 2 / (Ss + Sb) ** 3) / np.mean(Ss))
        aS = at[:, None] * Sb[None, :]
        t_exact = float((np.mean(Ss * Sb / (Ss + Sb)) - np.mean(Ss[None, :] * aS / (Ss[None, :] + aS))) / np.mean(Ss))
        rxx = m.xcorr(x, x, L); rsx = m.xcorr(s, x, L)
        w = solve(toeplitz(rxx) + np.eye(L) * 1e-8 * rxx[0], rsx, assume_a='pos')
        rows.append(dict(base=base, law=law, seg_ms=seg // 8, sentence=k, nu=nu,
                         nu_eff=float(np.var(at) / np.mean(at) ** 2), nsi=m.nsi_std(v),
                         T=mse_c - mse_nc, T_2nd=c0 * float(np.var(at) / np.mean(at) ** 2),
                         T_exact=t_exact, c0=c0, bs=mse_c,
                         bc=float(np.mean(s ** 2) - rsx @ w) / np.mean(s ** 2)))
    return rows


if __name__ == '__main__':
    cfgs = [(b, 'gamma', 512) for b in m.TYPES]
    cfgs += [('car', law, 512) for law in ['lognormal', 'twostate']]
    cfgs += [('car', 'gamma', seg) for seg in [256, 1024, 2048]]
    rows = []
    for base, law, seg in cfgs:
        for k in range(1, 31):
            rows += run(k, base, law, seg)
        print(base, law, seg, 'done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f'results_general_snr{int(SNR_DB)}.csv'), index=False)
