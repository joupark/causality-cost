"""Reproduces every MS-SNSD number in the letter from
results_msnsd_a.csv + results_msnsd_b.csv (written by msnsd_validate.py)."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_stats import cluster_slope, holm_bonferroni

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
out = open(os.path.join(HERE, 'numbers_msnsd.txt'), 'w')
def P(*a):
    print(*a); print(*a, file=out)

a = pd.read_csv(os.path.join(HERE, 'results_msnsd_a.csv'))
b = pd.read_csv(os.path.join(HERE, 'results_msnsd_b.csv'))
d = pd.concat([a, b], ignore_index=True)
d.to_csv(os.path.join(HERE, 'results_msnsd.csv'), index=False)
for m in ['nlms', 'mmse', 'omlsa']:
    d['gap_' + m] = d.bc_sig - d[m]

P(f'== MS-SNSD (16 kHz, n={len(d)} = {d.speaker.nunique()} speech x '
  f'{d.noise_type.nunique()} noise types), SNR=15dB ==')
P('NSI range:', round(d.nsi.min(), 2), '-', round(d.nsi.max(), 2), 'dB')
P('means:'); P(d[['bc_sig', 'nlms', 'mmse', 'omlsa']].mean().round(3).to_string())


def within(df, x, y):
    xd = df[x] - df.groupby('noise_type')[x].transform('mean')
    yd = df[y] - df.groupby('noise_type')[y].transform('mean')
    return cluster_slope(xd, yd, df.speaker)


P('\n== Bc vs NSI (SegSNR domain) -- replicates NOIZEUS slope -0.25')
r = within(d, 'nsi', 'bc_sig')
P(f"slope {r['slope']:+.4f}  p_cluster {r['p_cluster']:.3e}  t={r['t']:.3f} df={r['df']}")

P('\n== Tracking gap T vs NSI (Section III-C)')
r = within(d, 'nsi', 'dB')
P(f"slope {r['slope']:+.4f}  p_cluster {r['p_cluster']:.3e}  R2={r['r2']:.3f}")

P('\n== Gap vs NSI, cluster-robust + Holm-Bonferroni across 3 methods')
res = {}
for m in ['nlms', 'mmse', 'omlsa']:
    r = within(d, 'nsi', 'gap_' + m); res[m] = r
    P(f"{m:6s} slope {r['slope']:+.3f}  p_cluster {r['p_cluster']:.3e}")
holm = holm_bonferroni([res[m]['p_cluster'] for m in ['nlms', 'mmse', 'omlsa']],
                        ['nlms', 'mmse', 'omlsa'])
for lab, praw, padj, rej in holm:
    P(f'  Holm: {lab:6s} p_holm {padj:.3e}  reject@0.05 {rej}')

P('\n== Bc global dSNR (MSE domain) vs NSI -- direction check vs NOIZEUS (+0.12)')
mse_path = os.path.join(HERE, 'results_msnsd_mse.csv')
if os.path.exists(mse_path):
    dm = pd.read_csv(mse_path)
    r = within(dm, 'nsi', 'dsnr')
    P(f"slope {r['slope']:+.4f}  p_cluster {r['p_cluster']:.3e}  mean_dsnr={dm.dsnr.mean():.3f}")
else:
    P('  (results_msnsd_mse.csv not found -- run msnsd_mse.py first)')

# ---- Figure 4 (3 panels)
cats = sorted(d.noise_type.unique())
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
xd = d.nsi - d.groupby('noise_type').nsi.transform('mean')
yd = d.bc_sig - d.groupby('noise_type').bc_sig.transform('mean')
r_bc = within(d, 'nsi', 'bc_sig')
ax[0].scatter(xd, yd, s=10, alpha=.45, color='gray')
xx = np.linspace(xd.min(), xd.max(), 50)
ax[0].plot(xx, r_bc['slope']*xx, 'k-', lw=2,
           label=f"{r_bc['slope']:+.3f} dB/dB\n$p$={r_bc['p_cluster']:.1e} (cluster)")
ax[0].axhline(0, ls=':', c='gray'); ax[0].axvline(0, ls=':', c='gray')
ax[0].set(xlabel='NSI - type mean (dB)', ylabel=r'$\Delta$SegSNR of $B_c$ - type mean',
           title='(a) $B_c$ vs NSI (SegSNR)')
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

yd2 = d.dB - d.groupby('noise_type').dB.transform('mean')
r_db = within(d, 'nsi', 'dB')
ax[1].scatter(xd, yd2, s=10, alpha=.45, color='gray')
ax[1].plot(xx, r_db['slope']*xx, 'k-', lw=2,
           label=f"{r_db['slope']:+.3f} dB/dB\n$p$={r_db['p_cluster']:.3f} (cluster)")
ax[1].axhline(0, ls=':', c='gray'); ax[1].axvline(0, ls=':', c='gray')
ax[1].set(xlabel='NSI - type mean (dB)', ylabel=r'$T$ - type mean (dB)',
           title='(b) Tracking gap $T$')
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

g = d.groupby('noise_type').mean(numeric_only=True).loc[cats]
for m, c, mk, lab in [('nlms', 'tab:purple', '^', 'NLMS'), ('mmse', 'tab:blue', 'o', 'MMSE-W'),
                        ('omlsa', 'tab:green', 's', 'OMLSA')]:
    slg, icg, rg, pg, seg = stats.linregress(g.nsi, g['gap_' + m])
    ax[2].scatter(g.nsi, g['gap_' + m], s=40, edgecolors=c,
                   facecolors='none' if m == 'nlms' else c, marker=mk,
                   label=f'{lab}: {slg:+.2f} dB/dB')
    ax[2].plot(g.nsi, icg + slg*g.nsi, '--', c=c)
ax[2].axhline(0, c='k', ls=':')
ax[2].set(xlabel='NSI (dB)', ylabel=r'Gap to $B_c$ (dB)', title='(c) Gap vs NSI, type means')
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout()
figdir = os.path.join(HERE, '..', 'figures')
os.makedirs(figdir, exist_ok=True)
fig.savefig(os.path.join(figdir, 'fig_msnsd.pdf'))
P('\nwrote', os.path.relpath(os.path.join(figdir, 'fig_msnsd.pdf'), HERE))
out.close()
