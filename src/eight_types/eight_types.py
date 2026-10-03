"""Robustness check: all eight NOIZEUS noise types at 0/5/10 dB.

The 15 dB analyses in the letter use seven noise types because the NOIZEUS
distribution has no 15 dB archive for suburban-train noise (train_0dB.zip,
train_5dB.zip, train_10dB.zip exist; train_15dB.zip does not). At the SNRs
where train noise is available, this script recomputes the tracking gap T and
B_c with seven and with eight types and prints both.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import measure as m
from mmse_wiener_baseline import stft
from scipy.ndimage import uniform_filter1d
from scipy import stats
from cluster_stats import cluster_slope

def measure_all():
    rows = []
    for snr in [0, 5, 10]:
        for nt in m.TYPES + ['train']:
            for k in range(1, 31):
                sp = f'sp{k:02d}'; s, x, v = m.load(snr, sp, nt); sh, _ = m.bc_signal(s, x, 128)
                S = stft(s, 256, 128); V = stft(v, 256, 128); X = stft(x, 256, 128)
                Ss = np.mean(np.abs(S) ** 2, 0); Sl = uniform_filter1d(np.abs(V) ** 2, 3, axis=0); Sb = Sl.mean(0)
                mc = np.mean(np.abs(S - Ss / (Ss + Sb) * X) ** 2); mn = np.mean(np.abs(S - Ss / (Ss + Sl) * X) ** 2)
                rows.append(dict(snr=snr, noise_type=nt, speaker=sp, nsi=m.nsi_std(v),
                                 dB=10 * np.log10(mc / mn),
                                 dsnr=10 * np.log10(np.mean(v ** 2) / np.mean((s - sh) ** 2))))
    return pd.DataFrame(rows)

def within(df, x, y):
    xd = df[x] - df.groupby('noise_type')[x].transform('mean')
    yd = df[y] - df.groupby('noise_type')[y].transform('mean')
    return cluster_slope(xd, yd, df.speaker)

if __name__ == '__main__':
    p = os.path.join(HERE, 'results_eight_types.csv')
    d = pd.read_csv(p) if os.path.exists(p) else measure_all()
    d.to_csv(p, index=False)
    out = open(os.path.join(HERE, 'numbers_eight_types.txt'), 'w')
    def P(*a): print(*a); print(*a, file=out)
    P('train-noise mean NSI by SNR:', d[d.noise_type == 'train'].groupby('snr').nsi.mean().round(2).to_dict())
    for y, lab in [('dB', 'T'), ('dsnr', 'B_c (global dSNR)')]:
        for snr in [0, 5, 10]:
            for name, sub in [('7 types', d[(d.snr == snr) & (d.noise_type != 'train')]), ('8 types', d[d.snr == snr])]:
                r = within(sub, 'nsi', y); g = sub.groupby('noise_type')[['nsi', y]].mean(); b = stats.linregress(g.nsi, g[y])
                P(f'{lab:20s} {snr:2d} dB {name}: within slope {r["slope"]:+.3f} (p {r["p_cluster"]:.2e}); '
                  f'type means slope {b.slope:+.3f} (r {b.rvalue:+.2f}, p {b.pvalue:.3f})')
    out.close()
