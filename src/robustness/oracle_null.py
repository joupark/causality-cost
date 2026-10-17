"""Null test for the recorded-noise tracking-gap estimate (Sec. III-C).

The operational oracle uses a K-frame local periodogram of the noise that
includes the current frame, so it follows the noise *realisation* as well as
any change in the noise *statistics*.  To separate the two, each utterance's
noise is replaced by a stationary surrogate with the identical magnitude
spectrum (phase randomisation), for which the true tracking gap is zero.

Per utterance, for K in KS:
  T_real   10 log10(MSE_LTI / MSE_oracle), recorded noise
  T_null   same, phase-randomised surrogate (mean of NSUR draws)
and, as an oracle that cannot see the current frame's realisation,
  T_loo    local PSD from frames t-2 and t+2 (no sample overlap with t)

Usage: NOIZEUS_DIR=... python3 oracle_null.py   -> results_oracle_null.csv
"""
import os, sys, hashlib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import measure as m
from mmse_wiener_baseline import stft
from scipy.ndimage import uniform_filter1d

KS = [1, 3, 5, 9, 17]
NSUR = 5
SNRS = [0, 15]


def dseed(*p):
    return int(hashlib.md5('_'.join(map(str, p)).encode()).hexdigest()[:8], 16)


def surrogate(v, seed):
    V = np.fft.rfft(v)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(V))
    ph[0] = 0
    return np.fft.irfft(np.abs(V) * np.exp(1j * ph), len(v))


def gaps(s, v):
    S = stft(s, 256, 128); V = stft(v, 256, 128); X = S + V
    Ss = np.mean(np.abs(S) ** 2, 0); P = np.abs(V) ** 2; Sb = P.mean(0)
    mse_lti = np.mean(np.abs(S - Ss / (Ss + Sb) * X) ** 2)
    out = {}
    for K in KS:
        Sl = uniform_filter1d(P, K, axis=0)
        out[f'K{K}'] = 10 * np.log10(mse_lti / np.mean(np.abs(S - Ss / (Ss + Sl) * X) ** 2))
    Pp = np.pad(P, ((2, 2), (0, 0)), mode='edge')
    Sl = 0.5 * (Pp[:-4] + Pp[4:])
    out['loo'] = 10 * np.log10(mse_lti / np.mean(np.abs(S - Ss / (Ss + Sl) * X) ** 2))
    return out


if __name__ == '__main__':
    rows = []
    for snr in SNRS:
        for nt in m.TYPES:
            for k in range(1, 31):
                sp = f'sp{k:02d}'
                s, x, v = m.load(snr, sp, nt)
                row = dict(snr=snr, noise_type=nt, speaker=sp, nsi=m.nsi_std(v))
                for key, val in gaps(s, v).items():
                    row[f'T_{key}'] = val
                nul = [gaps(s, surrogate(v, dseed(snr, nt, k, j))) for j in range(NSUR)]
                for key in nul[0]:
                    row[f'N_{key}'] = float(np.mean([n[key] for n in nul]))
                row['nsi_sur'] = m.nsi_std(surrogate(v, dseed(snr, nt, k, 0)))
                rows.append(row)
        print(snr, 'dB done', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'results_oracle_null.csv'), index=False)
