"""Controlled modulation experiment (tests Propositions 1-2 and Lemma 1
by manipulating non-stationarity directly instead of observing it).

PREDICTIONS, written before running:
 (A) Stationary base noise (NOIZEUS car) is given a mean-one power envelope
     a(k) ~ Gamma(1/nu, nu) on 64 ms segments, so the averaged spectrum is
     fixed while nu varies.
     A1: tracking gap T (frequency-domain LTI gain vs frame oracle, linear MSE)
         grows with the effective envelope variance nu_eff with slope c0
         (Lemma 1), c0 computed from the averaged PSDs alone.
     A2: B_c (FIR Wiener-Hopf, L=128, MSE) does not change with nu
         (Proposition 1(i)).
     A3: if B_c's DeltaSegSNR still falls with nu, the SegSNR decrease is a
         metric effect (MSE held fixed by construction).
     A4 (added 2026-09-11, before re-running): the non-causal LTI bound B_s
         (MSE of the time-invariant gain Ss/(Ss+Sv_bar)) is also unchanged
         with nu, so the causality cost C = B_c - B_s is unchanged.
 (B) v = w_L*b_L + w_H*b_H, with b_L = car noise low-passed at 1 kHz and
     b_H = white noise high-passed at 2 kHz; one component is multiplied
     by sqrt(min(a,1)), so its average power falls as nu grows; total
     noise power is renormalised (A2').
     B1 (config 'mod_low'): power leaves the low (speech) band -> B_c
         improves (dB_c/dnu < 0).
     B2 (config 'mod_high'): power leaves the high band, relative
         low-band power rises -> B_c degrades (dB_c/dnu > 0).
     B3: the Proposition 2 quantity w^T H w, computed from w at nu_j and
         the NOISE-ONLY correlation change to nu_{j+1}, predicts the sign
         and size of the observed (B_c(nu_{j+1}) - B_c(nu_j)).
"""
import os, sys, hashlib, json, numpy as np, pandas as pd
from scipy.linalg import toeplitz, solve
from scipy.signal import fftconvolve, butter, sosfiltfilt
from scipy.stats import gamma as gamma_dist, ttest_1samp
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import measure as m
from segsnr_metrics import compute_delta_segsnr
from mmse_wiener_baseline import stft

FS = 8000; L = 128; NFFT = 256; HOP = 128; SEG = 512
SNR_DB = float(os.environ.get('CM_SNR', 15))
NUS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


def dseed(*p):
    return int(hashlib.md5('_'.join(map(str, p)).encode()).hexdigest()[:8], 16)


def envelope(n, nu, k, tag):
    nseg = int(np.ceil(n / SEG))
    u = np.random.default_rng(dseed(k, tag)).uniform(1e-6, 1 - 1e-6, nseg)
    a = np.ones(nseg) if nu == 0 else gamma_dist.ppf(u, a=1.0 / nu, scale=nu)
    return np.repeat(a, SEG)[:n]


def wiener(s, x):
    rxx = m.xcorr(x, x, L); rsx = m.xcorr(s, x, L)
    R = toeplitz(rxx) + np.eye(L) * 1e-8 * rxx[0]
    w = solve(R, rsx, assume_a='pos')
    return w, float(np.mean(s ** 2) - rsx @ w), fftconvolve(x, w)[:len(x)]


def scale_to_snr(s, v):
    return v * np.sqrt(np.mean(s ** 2) / 10 ** (SNR_DB / 10) / np.mean(v ** 2))


def part_a(k):
    s, x0, v0 = m.load(15, f'sp{k:02d}', 'car')
    b = v0 / np.sqrt(np.mean(v0 ** 2))
    win = np.hanning(NFFT)
    S = stft(s, NFFT, HOP); Ss = np.mean(np.abs(S) ** 2, 0)
    rows = []
    for nu in NUS:
        a = envelope(len(s), nu, k, 'A'); a = a / a.mean()
        v = scale_to_snr(s, np.sqrt(a) * b); x = s + v
        g2 = np.mean(v ** 2) / np.mean(a * b ** 2)          # v = sqrt(g2*a)*b
        X = stft(x, NFFT, HOP); B = stft(np.sqrt(g2) * b, NFFT, HOP)
        T = min(S.shape[0], X.shape[0], B.shape[0])
        Sb = np.mean(np.abs(B[:T]) ** 2, 0)
        at = np.array([np.sum(win ** 2 * a[i*HOP:i*HOP+NFFT] * b[i*HOP:i*HOP+NFFT] ** 2) /
                       np.sum(win ** 2 * b[i*HOP:i*HOP+NFFT] ** 2) for i in range(T)])
        nu_eff = float(np.var(at) / np.mean(at) ** 2)
        Gc = Ss / (Ss + Sb)
        Gnc = Ss[None, :] / (Ss[None, :] + at[:, None] * Sb[None, :])
        Pn = np.mean(np.abs(S[:T]) ** 2)
        mse_c = np.mean(np.abs(S[:T] - Gc * X[:T]) ** 2) / Pn
        mse_nc = np.mean(np.abs(S[:T] - Gnc * X[:T]) ** 2) / Pn
        c0 = float(np.mean(Ss ** 2 * Sb ** 2 / (Ss + Sb) ** 3) / np.mean(Ss))
        w, bc_mse, shat = wiener(s, x)
        rows.append(dict(sentence=k, nu=nu, nu_eff=nu_eff, nsi=m.nsi_std(v),
                          dB_lin=mse_c - mse_nc, dB_pred=c0 * nu_eff, c0=c0,
                          bs_norm=mse_c, oracle_norm=mse_nc,
                          bc_mse_norm=bc_mse / np.mean(s ** 2),
                          bc_dsnr=10 * np.log10(np.mean(v ** 2) / np.mean((s - shat) ** 2)),
                          bc_dsegsnr=compute_delta_segsnr(s, x, shat, fs=FS)))
    return rows


def part_b(k):
    s, x0, v0 = m.load(15, f'sp{k:02d}', 'car')
    bL = sosfiltfilt(butter(6, 1000, 'low', fs=FS, output='sos'), v0)
    wn = np.random.default_rng(dseed(k, 'white')).standard_normal(len(s))
    bH = sosfiltfilt(butter(6, 2000, 'high', fs=FS, output='sos'), wn)
    bL /= np.sqrt(np.mean(bL ** 2)); bH /= np.sqrt(np.mean(bH ** 2))
    rows = []
    for cfg in ['mod_low', 'mod_high']:
        prev = None
        for nu in NUS:
            c = np.minimum(envelope(len(s), nu, k, 'B'), 1.0)
            v = (np.sqrt(c) * bL + bH) if cfg == 'mod_low' else (bL + np.sqrt(c) * bH)
            v = scale_to_snr(s, v); x = s + v
            w, bc_mse, _ = wiener(s, x)
            Rvv = toeplitz(m.xcorr(v, v, L))
            row = dict(sentence=k, config=cfg, nu=nu, bc_mse=bc_mse, low_share=float(
                np.mean(sosfiltfilt(butter(6, 1500, 'low', fs=FS, output='sos'), v) ** 2) / np.mean(v ** 2)))
            if prev is not None:
                dnu = nu - prev['nu']
                row.update(obs=(bc_mse - prev['bc_mse']) / dnu,
                           pred=float(prev['w'] @ (Rvv - prev['Rvv']) @ prev['w']) / dnu)
            rows.append(row)
            prev = dict(nu=nu, bc_mse=bc_mse, w=w, Rvv=Rvv)
    return rows


if __name__ == '__main__':
    which = sys.argv[1]
    rows = []
    for k in range(1, 31):
        rows += part_a(k) if which == 'A' else part_b(k)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f'results_part{which}_snr{int(SNR_DB)}.csv'), index=False)
    print('wrote part', which, len(rows), 'rows')
