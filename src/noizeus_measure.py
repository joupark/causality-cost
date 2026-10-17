"""NOIZEUS measurements for the letter: tracking gap T, B_c, NSI.

Computes, per utterance and from the audio alone (no cached CSVs):
  NSI            standard deviation of the 20 ms log-energy of v = x - s (dB)
  T              10*log10(MSE_LTI / MSE_oracle) with the gains of Eq. (2):
                 the utterance-averaged (non-causal LTI) gain and an oracle
                 gain from a three-frame local noise periodogram
  bc_dsnr        global Delta SNR of the causal Wiener filter (L = 128), i.e.
                 B_c in the MSE domain (higher is better)
  bc_dsegsnr     Delta SegSNR of the same filter (not monotone in the MSE)

and prints the statistics quoted in Sec. III-C and Table II: within-noise-type
slopes versus NSI with cluster-robust standard errors (cluster = sentence),
the between-type slopes, and the sentence-identity ANOVA.

Usage:  export NOIZEUS_DIR=/path/to/noizeus ;  python3 noizeus_measure.py
Writes results_noizeus.csv and numbers_noizeus.txt next to this script;
spl_figs.py draws Fig. 2 of the letter from them.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'baselines'))
import measure as m
from mmse_wiener_baseline import stft
from segsnr_metrics import compute_delta_segsnr
from scipy.ndimage import uniform_filter1d
from scipy import stats
from cluster_stats import cluster_slope

SNRS = [0, 5, 10, 15]


def measure():
    rows = []
    for snr in SNRS:
        for nt in m.TYPES:
            for k in range(1, 31):
                sp = f'sp{k:02d}'
                s, x, v = m.load(snr, sp, nt)
                sh, _ = m.bc_signal(s, x, 128)
                S = stft(s, 256, 128); V = stft(v, 256, 128); X = stft(x, 256, 128)
                Ss = np.mean(np.abs(S) ** 2, 0)
                Sl = uniform_filter1d(np.abs(V) ** 2, 3, axis=0)
                Sb = Sl.mean(0)
                mse_lti = np.mean(np.abs(S - Ss / (Ss + Sb) * X) ** 2)
                mse_orc = np.mean(np.abs(S - Ss / (Ss + Sl) * X) ** 2)
                rows.append(dict(
                    snr=snr, noise_type=nt, speaker=sp,
                    nsi=m.nsi_std(v),
                    T=10 * np.log10(mse_lti / mse_orc),
                    bc_dsnr=10 * np.log10(np.mean(v ** 2) / np.mean((s - sh) ** 2)),
                    bc_dsegsnr=compute_delta_segsnr(s, x, sh, fs=8000)))
        print(f'  {snr} dB done', flush=True)
    return pd.DataFrame(rows)


def within(d, x, y):
    xd = d[x] - d.groupby('noise_type')[x].transform('mean')
    yd = d[y] - d.groupby('noise_type')[y].transform('mean')
    return cluster_slope(xd, yd, d.speaker)


def figure(d15, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9))
    g = d15.groupby('noise_type').agg(
        nsi=('nsi', 'mean'), T=('T', 'mean'),
        se=('T', lambda v: v.std(ddof=1) / np.sqrt(len(v)))).sort_values('nsi')
    ax[0].errorbar(g.nsi, g['T'], yerr=g.se, fmt='o', ms=3, capsize=2)
    xs = np.linspace(g.nsi.min(), g.nsi.max(), 10)
    ax[0].plot(xs, np.polyval(np.polyfit(g.nsi, g['T'], 1), xs), 'k--', lw=1)
    ax[0].set(xlabel='NSI (dB)', ylabel='$T$ (dB)', title='(a) Type means')
    xd = d15.nsi - d15.groupby('noise_type').nsi.transform('mean')
    yd = d15['T'] - d15.groupby('noise_type')['T'].transform('mean')
    ax[1].scatter(xd, yd, s=3, alpha=.4, color='gray')
    xs = np.linspace(xd.min(), xd.max(), 10)
    ax[1].plot(xs, np.polyval(np.polyfit(xd, yd, 1), xs), 'k-', lw=1)
    ax[1].set(xlabel='NSI $-$ type mean', ylabel='$T-$ type mean', title='(b) Within type')
    for a in ax:
        a.grid(alpha=.3); a.tick_params(labelsize=6)
        a.xaxis.label.set_size(7); a.yaxis.label.set_size(7); a.title.set_size(7)
    fig.tight_layout(); fig.savefig(path)


if __name__ == '__main__':
    csv = os.path.join(HERE, 'results_noizeus.csv')
    if os.path.exists(csv):
        d = pd.read_csv(csv); print('using existing', csv, '(delete it to re-measure)')
    else:
        d = measure(); d.to_csv(csv, index=False)
    out = open(os.path.join(HERE, 'numbers_noizeus.txt'), 'w')
    def P(*a): print(*a); print(*a, file=out)

    d15 = d[d.snr == 15]
    P(f'NOIZEUS 15 dB, n={len(d15)}: mean T {d15["T"].mean():+.2f} dB; '
      f'positive in {(d15.groupby("noise_type")["T"].mean() > 0).sum()} of 7 noise types')
    for y, lab in [('T', 'Tracking gap T'), ('bc_dsnr', 'B_c, global dSNR'),
                   ('bc_dsegsnr', 'B_c, dSegSNR')]:
        r = within(d15, 'nsi', y)
        g = d15.groupby('noise_type')[['nsi', y]].mean(); b = stats.linregress(g.nsi, g[y])
        P(f'{lab:20s}: within-type slope {r["slope"]:+.3f} dB/dB (cluster-robust p {r["p_cluster"]:.1e}, '
          f'df {r["df"]}, R2 {r["r2"]:.2f}) | type means slope {b.slope:+.3f}, r {b.rvalue:+.2f}, p {b.pvalue:.3f}')
    P('T versus NSI at the other SNRs (within noise type):')
    for snr in [0, 5, 10]:
        r = within(d[d.snr == snr], 'nsi', 'T')
        P(f'  {snr:2d} dB: slope {r["slope"]:+.3f} (p {r["p_cluster"]:.1e})')
    # Sentence-identity ANOVA, on B_c in the SegSNR domain (the quantity the
    # letter quotes); the same test on B_c in MSE gives 24%, F=1.96.
    for col, lab in [('bc_dsegsnr', 'dSegSNR'), ('bc_dsnr', 'MSE')]:
        res = d15[col] - d15.groupby('noise_type')[col].transform('mean')
        groups = [res[d15.speaker == sp].values for sp in d15.speaker.unique()]
        F, p = stats.f_oneway(*groups)
        ss_b = sum(len(g) * (g.mean() - res.mean()) ** 2 for g in groups)
        P(f'Sentence identity explains {100 * ss_b / np.sum((res - res.mean()) ** 2):.0f}% of the '
          f'noise-type-detrended variance of B_c in {lab} (F={F:.2f}, p={p:.1e})')
    out.close()
    print('wrote results_noizeus.csv, numbers_noizeus.txt; run spl_figs.py for Fig. 2')
