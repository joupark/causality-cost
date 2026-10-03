"""Statistics (numbers_controlled.txt) and Fig. 1 (fig_controlled.pdf) for the
controlled modulation experiment, from results_partA_snr{15,0}.csv and
results_partB_snr15.csv written by controlled_modulation.py."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, pearsonr
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
out = open(os.path.join(HERE, 'numbers_controlled.txt'), 'w')
def P(*a): print(*a); print(*a, file=out)
def ci(v): se = v.std(ddof=1) / np.sqrt(len(v)); return v.mean(), v.mean() - 1.96*se, v.mean() + 1.96*se
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
for snr, col in [(0, 'tab:blue'), (15, 'tab:red')]:
    d = pd.read_csv(os.path.join(HERE, f'results_partA_snr{snr}.csv'))
    P(f'== Part A, {snr} dB: NSI by nu', d.groupby('nu').nsi.mean().round(2).to_dict())
    for sub in [0.5, None]:
        sl = np.array([np.polyfit(s[s.nu <= (sub or 9)].nu_eff, s[s.nu <= (sub or 9)].dB_lin, 1)[0]
                       for _, s in d.groupby('sentence')])
        c0 = d.groupby('sentence').c0.mean().values
        mn, lo, hi = ci(sl)
        P(f'  Delta B slope (nu<={sub}): {mn:.5f} [{lo:.5f},{hi:.5f}], p={ttest_1samp(sl,0).pvalue:.1e}; '
          f'predicted c0 {c0.mean():.5f}; median ratio {np.median(sl/c0):.2f}')
    d['C_norm'] = d.bc_mse_norm - d.bs_norm
    for y in ['bs_norm', 'C_norm', 'oracle_norm']:
        sl = np.array([np.polyfit(s.nu, s[y], 1)[0] for _, s in d.groupby('sentence')])
        P(f'  {y} slope per unit nu: {sl.mean():+.6f} ({100*sl.mean()/d[d.nu==0][y].mean():+.2f}% per unit nu), p={ttest_1samp(sl,0).pvalue:.1e}')
    for y in ['bc_mse_norm', 'bc_dsnr', 'bc_dsegsnr']:
        sl = np.array([np.polyfit(s.nu, s[y], 1)[0] for _, s in d.groupby('sentence')])
        extra = f' ({100*sl.mean()/d[d.nu==0][y].mean():+.2f}% per unit nu)' if y == 'bc_mse_norm' else ''
        P(f'  {y} slope per unit nu: {sl.mean():+.5f}{extra}, p={ttest_1samp(sl,0).pvalue:.1e}')
    g = d.groupby('nu').mean(numeric_only=True)
    c0m = d.c0.mean()
    ax[0].plot(g.nu_eff, g.dB_lin / c0m, 'o-', color=col, label=f'{snr} dB')
    base = g.loc[0.0]
    ax[1].plot(g.index, g.bc_dsnr - base.bc_dsnr, 'o-', color=col, label=f'global $\\Delta$SNR, {snr} dB')
    ax[1].plot(g.index, g.bc_dsegsnr - base.bc_dsegsnr, 's--', color=col, label=f'$\\Delta$SegSNR, {snr} dB')
xx = np.linspace(0, 3.6, 10)
ax[0].plot(xx, xx, 'k:', label='Lemma 1: $\\Delta B/c_0=\\nu$')
ax[0].set(xlabel='effective envelope variance $\\nu$', ylabel='$\\Delta B / c_0$', title='(a) Causality cost vs. prediction')
ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].axhline(0, c='k', lw=.8)
ax[1].set(xlabel='nominal $\\nu$', ylabel='change of $B_c$ from $\\nu{=}0$ (dB)', title='(b) $B_c$: MSE fixed, SegSNR falls')
ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
b = pd.read_csv(os.path.join(HERE, 'results_partB_snr15.csv'))
full = b.copy(); b = b.dropna(subset=['obs'])
P('== Part B (Prop. 3), 15 dB')
for cfg, col, lab in [('mod_low', 'tab:green', 'low band modulated'), ('mod_high', 'tab:purple', 'high band modulated')]:
    s = b[b.config == cfg]; t = full[full.config == cfg].groupby('nu').bc_mse.mean()
    P(f'  {cfg}: B_c MSE change nu 0->4 {100*(t.iloc[-1]/t.iloc[0]-1):+.1f}%; sign agreement '
      f'{100*(np.sign(s.obs)==np.sign(s.pred)).mean():.1f}% of {len(s)}; r={pearsonr(s.pred,s.obs)[0]:.4f}; '
      f'median rel. error {np.median(np.abs(s.obs-s.pred)/np.abs(s.obs)):.3f}')
    ax[2].scatter(s.pred * 1e6, s.obs * 1e6, s=8, alpha=.5, color=col, label=lab)
lim = np.abs(b[['obs', 'pred']].values).max() * 1e6
ax[2].plot([-lim, lim], [-lim, lim], 'k:', lw=1)
ax[2].set(xlabel='predicted $\\mathbf{w}^{*T}\\mathbf{H}\\mathbf{w}^*$ ($\\times10^{-6}$)',
          ylabel='observed $\\Delta B_c/\\Delta\\nu$ ($\\times10^{-6}$)', title='(c) Retreat condition (Prop. 3)')
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, '..', 'figures', 'fig_controlled.pdf'))
out.close()
