"""../figures/fig_tauc.pdf: floor height and optimal context vs measured coherence time."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
tn = pd.read_csv(os.path.join(HERE, 'tauc_noizeus.csv')); ref = pd.read_csv(os.path.join(HERE, '..', 'results_context.csv'))
Cs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]; cn = [f'C{c}' for c in Cs]
ref['minex'] = ref[cn].min(axis=1); ref['Cstar'] = [Cs[int(np.argmin(r))] * 16 for r in ref[cn].values]
d = ref.merge(tn, on=['noise_type', 'speaker']); d['tau'] = d['tauc_0.1'] * 1000
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6)); types = d.groupby('noise_type').tau.mean().sort_values().index
for j, (y, lab, th) in enumerate([('minex', 'minimum excess MSE (floor height)', -2/3), ('Cstar', 'optimal context $C^\\star$ (ms)', 2/3)]):
    dd = d[d[y] > 0]
    for i, t in enumerate(types):
        s = dd[dd.noise_type == t]; c = plt.cm.viridis(i / 6)
        ax[j].scatter(s.tau, s[y], s=6, alpha=.35, color=c, marker='o')
        ax[j].scatter(np.exp(np.log(s.tau).mean()), np.exp(np.log(s[y]).mean()), s=45, color=c, edgecolor='k', label=t, zorder=3)
    g = dd.groupby('noise_type').agg(ltau=('tau', lambda v: np.log(v).mean()), ly=(y, lambda v: np.log(v).mean()))
    b, a = np.polyfit(g['ltau'], g['ly'], 1); xx = np.linspace(g['ltau'].min(), g['ltau'].max(), 10)
    ax[j].plot(np.exp(xx), np.exp(a + b * xx), 'k-', lw=1.2, label=f'type means: slope {b:+.2f}')
    ax[j].plot(np.exp(xx), np.exp(a + b * xx.mean() + th * (xx - xx.mean())), 'k:', lw=1.2, label=f'theory: slope {th:+.2f}')
    ax[j].set(xscale='log', yscale='log', xlabel='coherence time $\\tau_c$ (ms)', ylabel=lab,
              title=['(a) Floor height', '(b) Optimal context'][j]); ax[j].grid(alpha=.3, which='both')
ax[0].legend(fontsize=5.5, ncol=2, loc='lower left'); ax[1].legend(handles=ax[1].get_legend_handles_labels()[0][-2:], fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, '..', 'figures', 'fig_tauc.pdf'))
