"""Pooled-correlation FIR-order sweep: fits the Wiener-Hopf solution from
autocorrelations pooled across all 30 sentences of each noise type
(~84s of audio, vs ~2.8s for a single utterance), instead of one
utterance at a time.

Rationale: fir_sweep_extended.py (per-utterance) shows B_c continuing to
improve sharply out to L=4096 with no sign of saturating. This is
consistent with finite-sample overfitting of the sample autocorrelation
matrix at high lag order when only ~22000 samples (one utterance) are
available to estimate it -- not genuine convergence toward the
population Wiener bound. This script tests that hypothesis directly: if
true, pooling much more data per noise type should sharply flatten the
L-dependence at the same L values.
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import measure as m
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
from segsnr_metrics import compute_delta_segsnr
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve


def xcorr_pooled(a_list, b_list, L):
    num = np.zeros(L); den = 0
    for a, b in zip(a_list, b_list):
        N = len(a)
        A = np.fft.rfft(a, 2*N); B = np.fft.rfft(b, 2*N)
        num += np.fft.irfft(A * np.conj(B))[:L]
        den += N
    return num / den


if __name__ == '__main__':
    Ls = [128, 256, 512, 1024, 2048, 4096]
    rows = []
    t0 = time.time()
    for nt in m.TYPES:
        xs, ss, vs = [], [], []
        for k in range(1, 31):
            s, x, v = m.load(15, f'sp{k:02d}', nt)
            xs.append(x); ss.append(s); vs.append(v)
        for L in Ls:
            rxx = xcorr_pooled(xs, xs, L); rsx = xcorr_pooled(ss, xs, L)
            rxx = rxx.copy(); rxx[0] += 1e-8 * rxx[0]
            w = solve_toeplitz(rxx, rsx)
            dsegs, dsnrs = [], []
            for s, x, v in zip(ss, xs, vs):
                shat = fftconvolve(x, w)[:len(x)]
                dsegs.append(compute_delta_segsnr(s, x, shat, fs=8000))
                dsnrs.append(10*np.log10(np.mean(v**2)/np.mean((s-shat)**2)))
            rows.append(dict(noise_type=nt, L=L, dsegsnr=np.mean(dsegs), dsnr=np.mean(dsnrs)))
        print(nt, 'done', round(time.time() - t0, 1), 's', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_fir_pooled.csv'), index=False)
    print(d.pivot(index='L', columns='noise_type', values='dsegsnr').round(3))
    print()
    print('mean across types:')
    print(d.groupby('L')[['dsegsnr', 'dsnr']].mean().round(4))
