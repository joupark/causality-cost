"""Tests Q1-Q4 (see tauc.py) -> numbers_tauc.txt."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from cluster_stats import cluster_slope, holm_bonferroni
out = open(os.path.join(HERE, 'numbers_tauc.txt'), 'w')
def P(*a): print(*a); print(*a, file=out)
tn = pd.read_csv(os.path.join(HERE, 'tauc_noizeus.csv')); tm = pd.read_csv(os.path.join(HERE, 'tauc_msnsd.csv'))
ref = pd.read_csv(os.path.join(HERE, '..', 'results_context.csv'))
Cs = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]; cn = [f'C{c}' for c in Cs]
ref['minex'] = ref[cn].min(axis=1); ref['Cstar'] = [Cs[int(np.argmin(r))] * 0.016 for r in ref[cn].values]
def plateau(path):
    d = pd.read_csv(path); d = d[d.C.isin([64, 128, 99999])]
    return d.groupby(['noise_type', 'speaker']).agg(ex=('excess_db', 'mean'), nsi=('nsi', 'mean')).reset_index()
nn1 = plateau(os.path.join(HERE, '..', 'nn_experiment', 'results_context_nn_mse.csv'))
nn2 = plateau(os.path.join(HERE, '..', 'nn_big', 'results_context_big_mse.csv'))
sets = [('Q1 floor height  log(min excess) [ref-noise, NOIZEUS]', ref.merge(tn, on=['noise_type', 'speaker']), 'minex', True, -2/3),
        ('Q2 optimal context log(C*)       [ref-noise, NOIZEUS]', ref.merge(tn, on=['noise_type', 'speaker']), 'Cstar', True, 2/3),
        ('Q3a Network 1 plateau excess (dB) [NOIZEUS test]',     nn1.merge(tn, on=['noise_type', 'speaker']), 'ex', False, None),
        ('Q3b Network 2 plateau excess (dB) [MS-SNSD test]',     nn2.merge(tm, on=['noise_type', 'speaker']), 'ex', False, None)]
def within(d, x, y):
    xd = d[x] - d.groupby('noise_type')[x].transform('mean'); yd = d[y] - d.groupby('noise_type')[y].transform('mean')
    return cluster_slope(xd, yd, d.speaker)
prim = []
for eps in [0.1, 0.05, 0.2]:
    P(f'\n===== eps0 = {eps}' + ('  (PRIMARY)' if eps == 0.1 else '  (sensitivity)'))
    for lab, d, y, logy, theory in sets:
        d = d.copy()
        nonpos = int((d[y] <= 0).sum()) if logy else 0
        if logy: d = d[d[y] > 0]
        d['ltau'] = np.log(d[f'tauc_{eps}']); d['yy'] = np.log(d[y]) if logy else d[y]
        r = within(d, 'ltau', 'yy'); g = d.groupby('noise_type')[['ltau', 'yy', 'nsi']].mean()
        bt = stats.linregress(g['ltau'], g['yy']); bn = stats.linregress(g['nsi'], g['yy'])
        rn = within(d, 'nsi', 'yy')
        P(f'{lab}: n={len(d)} (non-positive dropped: {nonpos}), right-censored {100*d[f"cens_{eps}"].mean():.0f}%, at 16 ms floor {100*(d[f"tauc_{eps}"]<=0.0161).mean():.0f}%')
        P(f'   within-type  log tau_c: slope {r["slope"]:+.3f}{"" if theory is None else f" (theory {theory:+.3f})"}, '
          f'p_cluster {r["p_cluster"]:.3g}, R2 {r["r2"]:.3f} | NSI: slope {rn["slope"]:+.3f}, p {rn["p_cluster"]:.3g}, R2 {rn["r2"]:.3f}')
        P(f'   between-type (n={len(g)}) log tau_c: slope {bt.slope:+.3f}, r {bt.rvalue:+.2f}, p {bt.pvalue:.3g} | NSI: r {bn.rvalue:+.2f}, p {bn.pvalue:.3g}')
        if eps == 0.1: prim.append((lab.split()[0], r['p_cluster'], r['slope']))
P('\nHolm across the four primary within-type tests (eps0=0.1):')
for lab, pr, pa, rej in holm_bonferroni([p for _, p, _ in prim], [l for l, _, _ in prim]):
    P(f'   {lab}: p_raw {pr:.3g}  p_holm {pa:.3g}  reject {rej}')
out.close()
