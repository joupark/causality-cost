"""numbers_floor.txt and ../figures/fig_floor.pdf from floor_results.json."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'floor_results.json'))); ms = sorted(int(k) for k in R)
out = open(os.path.join(HERE, 'numbers_floor.txt'), 'w')
def P(*a): print(*a); print(*a, file=out)
b = np.array([R[str(m)]['beta'] for m in ms])
cols = ['lb_global', 'lb_exact', 'win_best', 'ewma_best', 'mlp_short', 'mlp_long']
A = {c: np.array([R[str(m)][c] for m in ms]) for c in cols}
best = np.minimum.reduce([A['win_best'], A['ewma_best'], A['mlp_long']])
P('m, beta, ' + ', '.join(cols) + ', best, best/lb_exact, win_best_C')
for i, m in enumerate(ms):
    P(m, f'{b[i]:.5f}', ' '.join(f'{A[c][i]:.3e}' for c in cols), f'{best[i]:.3e}',
      f'{best[i]/A["lb_exact"][i]:.2f}', R[str(m)]['win_best_C'])
for c in ['lb_exact', 'win_best', 'ewma_best', 'mlp_long', 'mlp_short']:
    P(f'log-log slope vs beta, {c}: {np.polyfit(np.log(b), np.log(A[c]), 1)[0]:.3f}')
P(f'log-log slope vs beta, best achievable: {np.polyfit(np.log(b), np.log(best), 1)[0]:.3f}')
P('all estimators >= lb_exact:', bool(all((A[c] >= A['lb_exact']).all() for c in cols[2:])))
P('window/lb_exact range:', np.round(A['win_best'] / A['lb_exact'], 2), '; mlp_long/lb_exact:', np.round(A['mlp_long'] / A['lb_exact'], 2))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for c, mk, lab in [('lb_global', 'k--', 'bound (T1)–(T3) constants'), ('lb_exact', 'k-', 'two-point bound $E[d^2]/8$'),
                   ('win_best', 'o-', 'best sliding window'), ('ewma_best', 'x:', 'best exponential smoothing'),
                   ('mlp_short', 's-', 'MLP, context $m_\\beta$'), ('mlp_long', 'D-', 'MLP, context $8m_\\beta$')]:
    ax[0].loglog(b, A[c], mk, label=lab, ms=4)
ax[0].loglog(b, A['lb_exact'][0] * (b / b[0]) ** (2 / 3) * 0.6, color='gray', lw=4, alpha=.3, label='slope 2/3')
ax[0].set(xlabel='drift rate $\\beta$', ylabel='risk $E\\|\\hat w-w^*\\|^2$', title='(a) Floor vs. drift rate')
ax[0].legend(fontsize=6); ax[0].grid(alpha=.3, which='both')
for m in ms:
    cur = R[str(m)]['win_curve']; C = np.array([int(k) for k in cur]); r = np.array([cur[k] for k in cur])
    o = np.argsort(C); ax[1].loglog(C[o] / m, r[o] / R[str(m)]['lb_exact'], label=f'$m_\\beta={m}$')
ax[1].axhline(4, color='k', ls=':', lw=1, label='limit $C\\to\\infty$ (4$\\times$ bound)')
ax[1].axhline(1, color='k', lw=1)
ax[1].set(xlabel='window length $C/m_\\beta$', ylabel='risk / two-point bound', title='(b) More context', ylim=(0.7, 3000))
ax[1].legend(fontsize=6); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, '..', 'figures', 'fig_floor.pdf')); out.close()
