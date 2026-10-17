import sys, numpy as np, pandas as pd
sys.path.insert(0, '..')
from cluster_stats import cluster_slope
from scipy.stats import ttest_1samp
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
def within(d, x, y):
    xd = d[x] - d.groupby('noise_type')[x].transform('mean')
    yd = d[y] - d.groupby('noise_type')[y].transform('mean')
    return cluster_slope(xd, yd, d.speaker)
d = pd.read_csv('results_oracle_null.csv')
print('surrogate NSI mean', d.groupby('snr').nsi_sur.mean().round(2).to_dict(), ' real NSI', d.groupby('snr').nsi.mean().round(2).to_dict())
for snr in [15, 0]:
    e = d[d.snr == snr].copy()
    print(f'\n===== ORACLE NULL, {snr} dB (n={len(e)})')
    for key in ['K1', 'K3', 'K5', 'K9', 'K17', 'loo']:
        e['D'] = e[f'T_{key}'] - e[f'N_{key}']
        r = within(e, 'nsi', f'T_{key}'); rn = within(e, 'nsi', f'N_{key}'); rd = within(e, 'nsi', 'D')
        t = ttest_1samp(e.groupby('speaker').D.mean(), 0)
        print(f'{key:4s} mean T_real {e[f"T_{key}"].mean():+.3f}  T_null {e[f"N_{key}"].mean():+.3f}  excess {e.D.mean():+.3f} (p {t.pvalue:.1e}, >0 in {(e.groupby("noise_type").D.mean()>0).sum()}/7 types)'
              f' | slope real {r["slope"]:+.3f} (p {r["p_cluster"]:.1e}, R2 {r["r2"]:.2f}); null {rn["slope"]:+.3f} (p {rn["p_cluster"]:.1e}); excess {rd["slope"]:+.3f} (p {rd["p_cluster"]:.1e}, R2 {rd["r2"]:.2f})')
    e['D3'] = e.T_K3 - e.N_K3
    print(e.groupby('noise_type')[['nsi', 'T_K3', 'N_K3', 'D3', 'T_loo', 'N_loo']].mean().sort_values('nsi').round(3))
    g = e.groupby('noise_type')[['nsi', 'D3', 'T_K3']].mean()
    from scipy.stats import linregress
    for c in ['T_K3', 'D3']:
        b = linregress(g.nsi, g[c]); print(f'  type means {c}: slope {b.slope:+.3f} r {b.rvalue:+.2f} p {b.pvalue:.3f}')

for snr in [15, 0]:
    g = pd.read_csv(f'results_general_snr{snr}.csv')
    print(f'\n===== CONTROLLED GENERALITY, {snr} dB')
    for (base, law, seg), c in g.groupby(['base', 'law', 'seg_ms'], sort=False):
        def per(f): return np.array([f(s) for _, s in c.groupby('sentence')])
        sm = per(lambda s: np.polyfit(s[s.nu <= .5].nu_eff, s[s.nu <= .5]['T'], 1)[0])
        c0 = per(lambda s: s.c0.mean())
        bc = per(lambda s: np.polyfit(s.nu, s.bc, 1)[0]); bs = per(lambda s: np.polyfit(s.nu, s.bs, 1)[0])
        full = c[c.nu > 0]
        rex = full.groupby('nu').apply(lambda s: s['T'].sum() / s.T_exact.sum())
        r2nd = full.groupby('nu').apply(lambda s: s['T'].sum() / s.T_2nd.sum())
        se = sm.std(ddof=1) / np.sqrt(len(sm))
        print(f'{base:10s} {law:9s} {seg:4d}ms NSI {c[c.nu==0].nsi.mean():4.1f}->{c[c.nu==4].nsi.mean():4.1f} | small-nu slope {sm.mean():.5f} [{sm.mean()-1.96*se:.5f},{sm.mean()+1.96*se:.5f}] vs c0 {c0.mean():.5f} | '
              f'Bc {100*bc.mean()/c[c.nu==0].bc.mean():+.2f}%/nu (p {ttest_1samp(bc,0).pvalue:.2f}) Bs {100*bs.mean()/c[c.nu==0].bs.mean():+.2f}%/nu (p {ttest_1samp(bs,0).pvalue:.2f}) | '
              f'T/T_2nd nu=.1..4: {" ".join(f"{v:.2f}" for v in r2nd.values)} | T/T_exact: {" ".join(f"{v:.2f}" for v in rex.values)}')
