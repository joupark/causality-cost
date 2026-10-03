import os
"""Context-truncation experiment for the trained causal NN (Theorem 1 test).

Evaluated on the held-out test set (sentences 21-30, 15 dB, 7 noise types,
n=70 = 10 sentences x 7 noise types) that never appeared in training or
model-selection (val used sentences 17-20). At inference, the model's
causal self-attention is restricted to the last C frames (no retraining);
C is swept over {1,2,4,8,16,32,64,128,None=unbounded}.

Statistical policy (see cluster_stats.py): the same 10 test sentences
appear once per noise type, so residuals are not independent across the
70 rows; all within-type slopes below are cluster-robust (cluster =
sentence, df = 9). Two claims are tested:
  (a) variance term  -- formal evidence: paired Wilcoxon signed-rank test,
      monotonicity in C, aggregated to one point per sentence (n=10,
      fully independent units, no clustering needed for a paired design).
  (b) non-stationarity floor -- ONE summary test (chosen after inspecting
      the per-C results for THIS network, i.e. post hoc; the same test was
      then fixed in advance for nn_big/, where it gave p=0.055): the
      gap-to-Bc-vs-NSI slope, averaged over the plateau region C in
      {64,128,unbounded} (where panel (a) shows the variance term has
      already saturated), cluster-robust. The slope at each individual C
      (Fig. 5c bars) is reported for transparency but is secondary /
      descriptive: with 9 non-independent tests on only 10 sentence
      clusters, Holm-Bonferroni correction does not let any single C
      survive on its own (reported explicitly below).
"""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import train_nn as t
from scipy import stats
from cluster_stats import cluster_slope, holm_bonferroni, paired_monotone_test

Cs = [1, 2, 4, 8, 16, 32, 64, 128, None]
PLATEAU = [64, 128, 99999]  # region where the variance term has saturated (Fig. 5a/b)

def spectral_mse(model, snr, k, nt, C):
    """Raw (linear, not dB) spectral MSE of the model's output at context C,
    and of a 3-frame-smoothed local-oracle Wiener gain, on the same STFT
    representation -- the quantity Theorem 1 is stated about."""
    lm, g, X, s = t.get_xyz(snr, k, nt)
    with torch.no_grad():
        mask = model(lm.unsqueeze(0), C=C).squeeze(0)
    S = t.stft_t(s)
    mse_model = torch.mean(torch.abs(S - mask * X) ** 2).item()
    # local oracle: 3-frame-smoothed instantaneous Wiener gain (as in
    # prop2_check.py / context_exp.py), independent of the trained model
    Ss = (S.abs() ** 2)
    Pn = (X - S).abs() ** 2
    Pn_loc = torch.from_numpy(
        np.apply_along_axis(lambda a: np.convolve(a, np.ones(3) / 3, mode='same'),
                             0, Pn.numpy()))
    Sbar = Ss.mean(0, keepdim=True)
    g_loc = (Sbar / (Sbar + Pn_loc + 1e-8)).clamp(0, 1)
    mse_oracle = torch.mean(torch.abs(S - g_loc * X) ** 2).item()
    return mse_model, mse_oracle


def main():
    model = t.TinyCausalTF()
    ck = torch.load('best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    fit, val, test = t.load_split()

    bc = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_15_nlms.csv'))
    bc = bc.set_index(['noise_type', 'speaker'])[['bc_sig', 'nsi_std']]

    rows = []
    mrows = []
    for C in Cs:
        res = t.evaluate(model, test, C=C)
        for nt, k, d in res:
            sp = f'sp{k:02d}'
            b, nsi = bc.loc[(nt, sp)]
            rows.append(dict(C=(C if C is not None else 99999), noise_type=nt,
                              speaker=sp, nsi=nsi, bc=b, dsegsnr=d, gap=b - d))
        for snr_, k, nt in test:
            sp = f'sp{k:02d}'; nsi = bc.loc[(nt, sp), 'nsi_std']
            mse_m, mse_o = spectral_mse(model, 15, k, nt, C)
            mrows.append(dict(C=(C if C is not None else 99999), noise_type=nt,
                               speaker=sp, nsi=nsi, mse_model=mse_m, mse_oracle=mse_o,
                               excess_db=10*np.log10(mse_m/mse_o)))
        print('spectral MSE done for C =', C)
    d = pd.DataFrame(rows)
    d.to_csv('results_context_nn.csv', index=False)
    dm = pd.DataFrame(mrows)
    dm.to_csv('results_context_nn_mse.csv', index=False)

    print('\n== Excess spectral MSE over local oracle (dB), mean over n=70')
    gm = dm.groupby('C').excess_db.agg(['mean', 'sem'])
    print(gm.round(3))

    print('\n== (a) VARIANCE TERM: formal monotonicity test')
    print('Sentence-level (n=10 independent sentences; noise types averaged out),')
    print('paired Wilcoxon signed-rank, H1: performance improves with more context.')
    d_unit = d.copy(); d_unit['unit'] = d_unit.speaker  # collapse noise types per sentence below
    sent = d.groupby(['speaker', 'C']).dsegsnr.mean().unstack('C')
    Cvals_all = [c for c in Cs if c is not None] + [99999]
    mono_rows = []
    for a_, b_ in zip(Cvals_all[:-1], Cvals_all[1:]):
        diff = sent[b_] - sent[a_]
        stat, p = stats.wilcoxon(diff, alternative='greater')
        mono_rows.append(dict(a=a_, b=b_, n=len(diff), median_diff=float(diff.median()), p=p))
        print(f'  C={a_:>6} -> C={b_:>6}  n={len(diff)}  median {diff.median():+.3f} dB  p={p:.4g}')
    ov = sent[99999] - sent[1]
    stat, p_overall = stats.wilcoxon(ov, alternative='greater')
    print(f'  Overall C=1 -> unbounded: median {ov.median():+.2f} dB, all_positive={all(ov>0)}, p={p_overall:.4g}')
    json.dump(dict(steps=mono_rows, overall_p=p_overall), open('monotonicity_nn.json', 'w'), indent=1)

    print('\n== 1/C fit in the LINEAR MSE domain, with sentence-bootstrap CI on R^2')
    from scipy.optimize import curve_fit
    Cs_arr = np.array([c for c in Cs if c is not None], dtype=float)
    def fmod(C, a, floor):
        return floor + a / C
    def fit_once(dsub):
        g_ = dsub.groupby('C').mse_model.mean()
        y_ = np.array([g_.loc[c] for c in Cs_arr])
        p_, _ = curve_fit(fmod, Cs_arr, y_, p0=[y_[0]*Cs_arr[0], g_.loc[99999]], maxfev=10000)
        pred_ = fmod(Cs_arr, *p_)
        r2_ = 1 - np.sum((y_ - pred_) ** 2) / np.sum((y_ - y_.mean()) ** 2)
        return r2_, p_[0], p_[1]
    r2_full, a_full, floor_full = fit_once(dm)
    rng = np.random.default_rng(0)
    sentences = dm.speaker.unique()
    boots = []
    for _ in range(2000):
        samp = rng.choice(sentences, size=len(sentences), replace=True)
        rows_bs = pd.concat([dm[dm.speaker == sp] for sp in samp])
        try:
            r2_bs, _, _ = fit_once(rows_bs); boots.append(r2_bs)
        except Exception:
            pass
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'a={a_full:.4g}  floor={floor_full:.4g}  R2={r2_full:.3f}  '
          f'95% CI (sentence bootstrap, n=2000) [{lo:.3f}, {hi:.3f}]')

    print('\n== Mean dSegSNR(C), n=70 per C')
    g = d.groupby('C').dsegsnr.agg(['mean', 'sem'])
    print(g.round(3))

    print('\n== (b) NON-STATIONARITY FLOOR')
    print('Secondary/descriptive: gap-vs-NSI slope at each individual C')
    print('(cluster-robust, cluster=sentence, df=9; Holm-Bonferroni across the 9 levels)')
    slopes = []
    for C, s in d.groupby('C'):
        xd = s.nsi - s.groupby('noise_type').nsi.transform('mean')
        yd = s.gap - s.groupby('noise_type').gap.transform('mean')
        r = cluster_slope(xd, yd, s.speaker)
        slopes.append((C, r['slope'], r['p_cluster'], r['se']))
        print(f"C={C:>6}  slope {r['slope']:+.3f}  cluster-SE {r['se']:.3f}  "
              f"p_cluster {r['p_cluster']:.3f}  (naive p {r['p_naive']:.3f})")
    holm = holm_bonferroni([s[2] for s in slopes], [s[0] for s in slopes])
    print('Holm-Bonferroni across all 9 context levels (none individually pre-specified):')
    for lab, praw, padj, rej in holm:
        print(f'  C={lab:>6}  p_raw {praw:.3f}  p_holm {padj:.3f}  reject@0.05 {rej}')

    print('\nSUMMARY floor test (post hoc for this network; pre-specified for nn_big):')
    print('gap-vs-NSI slope, averaged over the plateau region C in {64,128,unbounded}')
    print('(one test, not part of the 9-way family above; this is the number quoted in the paper)')
    plateau = d[d.C.isin(PLATEAU)].groupby(['noise_type', 'speaker']).agg(
        gap=('gap', 'mean'), nsi=('nsi', 'mean')).reset_index()
    xd = plateau.nsi - plateau.groupby('noise_type').nsi.transform('mean')
    yd = plateau.gap - plateau.groupby('noise_type').gap.transform('mean')
    rprim = cluster_slope(xd, yd, plateau.speaker)
    print(f"  slope {rprim['slope']:+.3f}  cluster-SE {rprim['se']:.3f}  t {rprim['t']:+.2f}  "
          f"df {rprim['df']}  p_cluster {rprim['p_cluster']:.4f}  (naive p {rprim['p_naive']:.4f})")

    json.dump({str(C): dict(slope=sl, p_cluster=pc, se=se) for C, sl, pc, se in slopes},
              open('context_nn_slopes.json', 'w'), indent=1)
    json.dump(dict(slope=rprim['slope'], se=rprim['se'], t=rprim['t'], df=rprim['df'],
                    p_cluster=rprim['p_cluster'], p_naive=rprim['p_naive']),
              open('primary_floor_test.json', 'w'), indent=1)

    # Monotonicity of the mean vs C (descriptive; formal test is the Wilcoxon block above)
    order = [c for c in Cs]
    means = [g.loc[(c if c is not None else 99999), 'mean'] for c in order]
    print('\nmeans in C order:', [round(mm, 2) for mm in means])
    diffs = np.diff(means)
    print('monotone non-decreasing in', sum(diffs >= -0.05), '/', len(diffs), 'steps (mean-level check)')

    make_figure(d, dm, slopes, holm, rprim)


def make_figure(d, dm, slopes, holm, rprim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
    os.makedirs(outdir, exist_ok=True)
    Cvals = [c for c in Cs if c is not None]  # 1..128, x-axis of the curves

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    g_mse = dm.groupby('C').excess_db.agg(['mean', 'sem'])
    ys = [g_mse.loc[c, 'mean'] for c in Cvals]
    es = [g_mse.loc[c, 'sem'] for c in Cvals]
    ax[0].errorbar(Cvals, ys, yerr=es, fmt='o-', color='tab:blue', capsize=3,
                    label='truncated context')
    ax[0].axhline(g_mse.loc[99999, 'mean'], color='tab:red', ls='--',
                   label=f"unbounded context ({g_mse.loc[99999, 'mean']:.2f} dB)")
    ax[0].set_xscale('log', base=2)
    ax[0].set_xlabel(r'Context $C$ (frames, 16 ms each)')
    ax[0].set_ylabel('Excess spectral MSE\nover local oracle (dB)')
    ax[0].set_title('(a) Variance term')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    g_seg = d.groupby('C').dsegsnr.agg(['mean', 'sem'])
    ys2 = [g_seg.loc[c, 'mean'] for c in Cvals]
    es2 = [g_seg.loc[c, 'sem'] for c in Cvals]
    ax[1].errorbar(Cvals, ys2, yerr=es2, fmt='s-', color='tab:green', capsize=3)
    ax[1].axhline(g_seg.loc[99999, 'mean'], color='tab:red', ls='--',
                   label=f"unbounded ({g_seg.loc[99999, 'mean']:.2f} dB)")
    ax[1].set_xscale('log', base=2)
    ax[1].set_xlabel(r'Context $C$ (frames)')
    ax[1].set_ylabel(r'$\Delta$SegSNR (dB)')
    ax[1].set_title('(b) Held-out test performance\n(Wilcoxon $p$<0.005 every step, $n{=}10$ sentences)')
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

    holm_by_C = {lab: padj for lab, praw, padj, rej in holm}
    labs = [str(c) for c in Cvals] + [r'$\infty$']
    vals = [sl[1] for sl in slopes]
    padj_ordered = [holm_by_C[sl[0]] for sl in slopes]
    plateau_mask = [sl[0] in PLATEAU for sl in slopes]
    colors = ['tab:red' if m else 'lightcoral' for m in plateau_mask]
    ax[2].bar(range(len(vals)), vals, color=colors)
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set_xticks(range(len(vals))); ax[2].set_xticklabels(labs, fontsize=7)
    ax[2].set_xlabel(r'Context $C$ (frames)')
    ax[2].set_ylabel('Gap-vs-NSI slope (dB/dB)')
    ax[2].set_title('(c) Individual $C$: descriptive only\n(dark = plateau, the primary test)', fontsize=8.5)
    ax[2].text(0.03, 0.95, f"primary test (plateau-averaged,\ncluster-robust): $p$={rprim['p_cluster']:.3f}",
               transform=ax[2].transAxes, fontsize=6.5, va='top')
    ax[2].grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_nn_context.pdf'))
    print('wrote', os.path.join(outdir, 'fig_nn_context.pdf'))

if __name__ == '__main__':
    main()
