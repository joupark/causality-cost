"""Context-truncation experiment (Theorem 1 test) for the larger,
MS-SNSD-trained causal network (274k params, ~2700 training mixtures),
mirroring nn_experiment/eval_context.py's methodology exactly so results
are directly comparable to the small NOIZEUS-trained network:

  - held-out test set: 20 MS-SNSD speech clips x 9 noise categories
    (n=180), never used in training or model selection.
  - context truncated at inference over C in {1,2,4,...,128,unbounded}
    (frames of 16ms at 16kHz, i.e. HOP=256 samples).
  - variance term: paired Wilcoxon signed-rank test, one point per
    speech clip (n=20 independent units, noise categories averaged out).
  - non-stationarity floor: one pre-specified confirmatory test
    (gap-to-Bc-vs-NSI slope averaged over the plateau region where the
    variance term has saturated), cluster-robust (cluster=speech clip).
"""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'msnsd_validate'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import soundfile as sf
import train_nn_big as t
from cluster_stats import cluster_slope, holm_bonferroni
from scipy import stats

Cs = [1, 2, 4, 8, 16, 32, 64, 128, None]
PLATEAU = [64, 128, 99999]


def spectral_mse(model, sf_path, cat, pools, C):
    lm, g, X, s, x, v, sp_id = t.get_xyz(sf_path, cat, pools, t.EVAL_SNR, crop_seed=0, crop=False)
    with torch.no_grad():
        mask = model(lm.unsqueeze(0), C=C).squeeze(0)
    S = t.stft_t(s)
    mse_model = torch.mean(torch.abs(S - mask * X) ** 2).item()
    Ss = S.abs() ** 2
    Pn = (X - S).abs() ** 2
    from scipy.ndimage import uniform_filter1d
    Pn_loc = torch.from_numpy(uniform_filter1d(Pn.numpy(), 3, axis=0))
    Sbar = Ss.mean(0, keepdim=True)
    g_loc = (Sbar / (Sbar + Pn_loc + 1e-8)).clamp(0, 1)
    mse_oracle = torch.mean(torch.abs(S - g_loc * X) ** 2).item()
    return mse_model, mse_oracle, sp_id


def main(Cs_subset=None, out_suffix=''):
    Cs_run = Cs_subset if Cs_subset is not None else Cs
    model = t.CausalTF()
    ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_big.pt'),
                     map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    print('loaded checkpoint: epoch', ck['epoch'], 'val', ck['val'])
    fit, val, test = t.split_clips(seed=0)
    pools = t.mv.load_noise_pools()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fir_and_conditionS'))
    from scipy.linalg import solve_toeplitz
    from scipy.signal import fftconvolve

    def bc_signal(s, x, L=128):
        N = len(x)
        A = np.fft.rfft(x, 2*N); B = np.fft.rfft(s, 2*N)
        rxx = np.fft.irfft(A*np.conj(A))[:L] / N
        rsx = np.fft.irfft(B*np.conj(A))[:L] / N
        rxx = rxx.copy(); rxx[0] += 1e-8*rxx[0]
        w = solve_toeplitz(rxx, rsx)
        return fftconvolve(x, w)[:len(x)]

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
    from segsnr_metrics import compute_delta_segsnr

    rows, mrows = [], []
    for C in Cs_run:
        for sf_path in test:
            for cat in t.mv.CATS:
                s, fs = sf.read(sf_path)
                sp_id = os.path.basename(sf_path).replace('.wav', '')
                x, v = t.mv.mix_at_snr(s, pools[cat], t.EVAL_SNR, seed=t.mv.det_seed(sp_id, cat, str(t.EVAL_SNR)))
                with torch.no_grad():
                    lm = torch.log(t.stft_t(x).abs() + 1e-8)
                    mask = model(lm.unsqueeze(0), C=C).squeeze(0)
                X = t.stft_t(x)
                Y = (X * mask).numpy()
                y = t.istft_np(Y, len(s))
                n = min(len(s), len(y))
                nn_dseg = compute_delta_segsnr(s[:n], x[:n], y[:n], fs=t.SR)
                shat = bc_signal(s, x, 128)
                bc = compute_delta_segsnr(s, x, shat, fs=t.SR)
                nsi = t.mv.nsi_std(v)
                rows.append(dict(C=(C if C is not None else 99999), noise_type=cat,
                                  speaker=sp_id, nsi=nsi, bc=bc, dsegsnr=nn_dseg, gap=bc - nn_dseg))
                # reuse the mask/X already computed above instead of a second forward pass
                Ss = X.abs() ** 2 * 0  # placeholder, replaced below
                S = t.stft_t(s)
                mse_model = torch.mean(torch.abs(S - mask * X) ** 2).item()
                Ssp = S.abs() ** 2
                Pn = (X - S).abs() ** 2
                from scipy.ndimage import uniform_filter1d
                Pn_loc = torch.from_numpy(uniform_filter1d(Pn.numpy(), 3, axis=0))
                Sbar = Ssp.mean(0, keepdim=True)
                g_loc = (Sbar / (Sbar + Pn_loc + 1e-8)).clamp(0, 1)
                mse_oracle = torch.mean(torch.abs(S - g_loc * X) ** 2).item()
                mrows.append(dict(C=(C if C is not None else 99999), noise_type=cat,
                                   speaker=sp_id, nsi=nsi, mse_model=mse_model, mse_oracle=mse_oracle,
                                   excess_db=10*np.log10(mse_model/mse_oracle)))
        print('C =', C, 'done', flush=True)
    d = pd.DataFrame(rows); dm = pd.DataFrame(mrows)
    HERE = os.path.dirname(os.path.abspath(__file__))
    d.to_csv(os.path.join(HERE, f'results_context_big{out_suffix}.csv'), index=False)
    dm.to_csv(os.path.join(HERE, f'results_context_big_mse{out_suffix}.csv'), index=False)
    print('wrote batch', out_suffix, len(d), 'rows')


def analyze():
    HERE = os.path.dirname(os.path.abspath(__file__))
    d = pd.read_csv(os.path.join(HERE, 'results_context_big.csv'))
    dm = pd.read_csv(os.path.join(HERE, 'results_context_big_mse.csv'))
    out = open(os.path.join(HERE, 'numbers_big.txt'), 'w')
    def P(*a): print(*a); print(*a, file=out)

    P('== Excess spectral MSE over local oracle (dB), n=180 per C')
    P(dm.groupby('C').excess_db.agg(['mean', 'sem']).round(3).to_string())

    P('\n== (a) VARIANCE TERM: paired Wilcoxon, n=20 speech clips (noise types averaged out)')
    sent = d.groupby(['speaker', 'C']).dsegsnr.mean().unstack('C')
    Cvals_all = [c for c in Cs if c is not None] + [99999]
    for a_, b_ in zip(Cvals_all[:-1], Cvals_all[1:]):
        diff = sent[b_] - sent[a_]
        stat, p = stats.wilcoxon(diff, alternative='greater')
        P(f'  C={a_:>6} -> C={b_:>6}  n={len(diff)}  median {diff.median():+.3f} dB  p={p:.4g}')
    ov = sent[99999] - sent[1]
    stat, p_overall = stats.wilcoxon(ov, alternative='greater')
    P(f'  Overall C=1 -> unbounded: median {ov.median():+.2f} dB, all_positive={all(ov>0)}, p={p_overall:.4g}')

    P('\n== (b) NON-STATIONARITY FLOOR: gap-vs-NSI slope per C (cluster-robust, cluster=speaker)')
    slopes = []
    for C, s_ in d.groupby('C'):
        xd = s_.nsi - s_.groupby('noise_type').nsi.transform('mean')
        yd = s_.gap - s_.groupby('noise_type').gap.transform('mean')
        r = cluster_slope(xd, yd, s_.speaker)
        slopes.append((C, r['slope'], r['p_cluster'], r['se']))
        P(f"C={C:>6}  slope {r['slope']:+.3f}  p_cluster {r['p_cluster']:.3f}  (naive p {r['p_naive']:.3f})")
    holm = holm_bonferroni([s[2] for s in slopes], [s[0] for s in slopes])
    P('Holm-Bonferroni across all 9 context levels:')
    for lab, praw, padj, rej in holm:
        P(f'  C={lab:>6}  p_holm {padj:.3f}  reject@0.05 {rej}')

    P('\nPRIMARY pre-specified confirmatory test for the floor (plateau C in {64,128,unbounded}):')
    plateau = d[d.C.isin(PLATEAU)].groupby(['noise_type', 'speaker']).agg(
        gap=('gap', 'mean'), nsi=('nsi', 'mean')).reset_index()
    xd = plateau.nsi - plateau.groupby('noise_type').nsi.transform('mean')
    yd = plateau.gap - plateau.groupby('noise_type').gap.transform('mean')
    rprim = cluster_slope(xd, yd, plateau.speaker)
    P(f"  slope {rprim['slope']:+.3f}  t {rprim['t']:+.2f}  df {rprim['df']}  "
      f"p_cluster {rprim['p_cluster']:.4f}  (naive p {rprim['p_naive']:.4f})")
    json.dump(dict(slope=rprim['slope'], p_cluster=rprim['p_cluster'], df=rprim['df']),
              open(os.path.join(HERE, 'primary_floor_test_big.json'), 'w'), indent=1)
    out.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=str, default='')  # comma-separated C values, or 'analyze'
    args = ap.parse_args()
    if args.batch == 'analyze':
        analyze()
    elif args.batch:
        vals = [None if v == 'None' else int(v) for v in args.batch.split(',')]
        suffix = '_' + '_'.join(str(v) for v in vals)
        main(Cs_subset=vals, out_suffix=suffix)
    else:
        main()
