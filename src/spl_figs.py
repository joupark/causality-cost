"""Regenerates Fig. 1 and Fig. 2 of the letter from the committed
per-utterance CSVs.  Labels use the letter's notation (tracking gap T,
sign condition (S) = Proposition 2).  Run: python3 src/spl_figs.py
Writes fig1_controlled.pdf and fig2_tracking_gap.pdf next to this file."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.abspath(__file__)); plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
for snr, col in [(0, 'tab:blue'), (15, 'tab:red')]:
    d = pd.read_csv(os.path.join(R, 'controlled_modulation', f'results_partA_snr{snr}.csv')); g = d.groupby('nu').mean(numeric_only=True)
    ax[0].plot(g.nu_eff, g.dB_lin / d.c0.mean(), 'o-', color=col, label=f'{snr} dB')
    e = pd.read_csv(os.path.join(R, 'robustness', f'results_general_snr{snr}.csv')); e = e[(e.base == 'car') & (e.law == 'gamma') & (e.seg_ms == 64)].groupby('nu').mean(numeric_only=True)
    ax[0].plot(e.nu_eff, e.T_exact / d.c0.mean(), '--', color=col, lw=1, label=f'exact Jensen gap, {snr} dB')
    b0 = g.loc[0.0]
    ax[1].plot(g.index, g.bc_dsnr - b0.bc_dsnr, 'o-', color=col, label=f'global $\\Delta$SNR, {snr} dB')
    ax[1].plot(g.index, g.bc_dsegsnr - b0.bc_dsegsnr, 's--', color=col, label=f'$\\Delta$SegSNR, {snr} dB')
    ax[1].plot(g.index, -10 * np.log10(g.bs_norm / b0.bs_norm), 'x:', color=col, label=f'$B_s$ (MSE), {snr} dB')
xx = np.linspace(0, 3.6, 10); ax[0].plot(xx, xx, 'k:', label='Lemma 1: $T/c_0=\\nu$')
ax[0].set(xlabel='effective envelope variance $\\nu$', ylabel='$T/c_0$', title='(a) Tracking gap vs. prediction'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].axhline(0, c='k', lw=.8); ax[1].set(xlabel='nominal $\\nu$', ylabel='change from $\\nu{=}0$ (dB)', title='(b) $B_c$, $B_s$ fixed; SegSNR falls'); ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
b = pd.read_csv(os.path.join(R, 'controlled_modulation', 'results_partB_snr15.csv')).dropna(subset=['obs'])
for cfg, col, lab in [('mod_low', 'tab:green', 'low band modulated'), ('mod_high', 'tab:purple', 'high band modulated')]:
    s = b[b.config == cfg]; ax[2].scatter(s.pred * 1e6, s.obs * 1e6, s=8, alpha=.5, color=col, label=lab)
lim = np.abs(b[['obs', 'pred']].values).max() * 1e6; ax[2].plot([-lim, lim], [-lim, lim], 'k:', lw=1)
ax[2].set(xlabel='predicted $\\mathbf{w}^{*T}\\mathbf{H}\\mathbf{w}^*$ ($\\times10^{-6}$)', ylabel='observed $\\Delta B_c/\\Delta\\nu$ ($\\times10^{-6}$)', title='(c) Sign condition (Prop. 2)')
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(R, 'fig1_controlled.pdf'))
a = pd.read_csv(os.path.join(R, 'results_noizeus.csv')); d = a[a.snr == 15].copy()
# the committed CSV names the tracking gap 'T'; this script's plotting code uses 'dB'
d['dB'] = d['T']
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9))
g = d.groupby('noise_type').agg(nsi=('nsi', 'mean'), T=('dB', 'mean'), se=('dB', lambda v: v.std(ddof=1) / np.sqrt(len(v)))).sort_values('nsi')
ax[0].errorbar(g.nsi, g['T'], yerr=g.se, fmt='o', ms=3, capsize=2); p = np.polyfit(g.nsi, g['T'], 1); xs = np.linspace(g.nsi.min(), g.nsi.max(), 10)
nl = pd.read_csv(os.path.join(R, 'robustness', 'results_oracle_null.csv')); nl = nl[nl.snr == 15].groupby('noise_type').agg(nsi=('nsi', 'mean'), N=('N_K3', 'mean'))
ax[0].plot(nl.nsi, nl.N, 's', ms=3, mfc='none', color='tab:orange')  # open squares: stationary-surrogate null
ax[0].plot(xs, np.polyval(p, xs), 'k--', lw=1); ax[0].set(xlabel='NSI (dB)', ylabel='$T$ (dB)', title='(a) Type means'); ax[0].grid(alpha=.3)
xd = d.nsi - d.groupby('noise_type').nsi.transform('mean'); yd = d.dB - d.groupby('noise_type').dB.transform('mean')
ax[1].scatter(xd, yd, s=3, alpha=.4, color='gray'); p = np.polyfit(xd, yd, 1); xs = np.linspace(xd.min(), xd.max(), 10)
ax[1].plot(xs, np.polyval(p, xs), 'k-', lw=1); ax[1].set(xlabel='NSI $-$ type mean', ylabel='$T-$ type mean', title='(b) Within type'); ax[1].grid(alpha=.3)
for a_ in ax: a_.tick_params(labelsize=6); a_.xaxis.label.set_size(7); a_.yaxis.label.set_size(7); a_.title.set_size(7)
fig.tight_layout(); fig.savefig(os.path.join(R, 'fig2_tracking_gap.pdf'))
print('figures written; type-mean slope %.3f' % np.polyfit(g.nsi, g['T'], 1)[0])
