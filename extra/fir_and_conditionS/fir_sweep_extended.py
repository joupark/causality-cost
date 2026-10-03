"""Extended FIR-order sensitivity of B_c, up to L=4096, using
scipy.linalg.solve_toeplitz (Levinson-Durbin-type, O(L^2)) so large L is
tractable -- the original fir_sweep.py used a dense solve (O(L^3)), which
does not scale past L~1024.

Question: does B_c saturate as L grows, and if not within a feasible
range, what is the extrapolated asymptotic value (L -> infinity) and how
far is L=128 (the value used throughout the paper) from it?
"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import measure as m
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
from segsnr_metrics import compute_delta_segsnr


def bc_signal_toeplitz(s, x, L):
    rxx = m.xcorr(x, x, L)
    rsx = m.xcorr(s, x, L)
    rxx = rxx.copy(); rxx[0] += 1e-8 * rxx[0]
    w = solve_toeplitz(rxx, rsx)
    shat = fftconvolve(x, w)[:len(x)]
    return shat


if __name__ == '__main__':
    Ls = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows = []
    t0 = time.time()
    for nt in m.TYPES:
        for k in range(1, 31, 3):  # 70-utterance subset, matches original fir_sweep.py
            s, x, v = m.load(15, f'sp{k:02d}', nt)
            for L in Ls:
                shat = bc_signal_toeplitz(s, x, L)
                dsegsnr = compute_delta_segsnr(s, x, shat, fs=8000)
                dsnr = 10 * np.log10(np.mean(v**2) / np.mean((s - shat)**2))
                rows.append(dict(noise_type=nt, sentence=k, L=L,
                                  dsegsnr=dsegsnr, dsnr=dsnr))
        print(nt, 'done', round(time.time() - t0, 1), 's', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_fir_sweep_extended.csv'), index=False)
    print(d.groupby('L')[['dsegsnr', 'dsnr']].mean().round(4))
