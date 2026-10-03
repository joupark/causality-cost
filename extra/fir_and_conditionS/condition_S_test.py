"""Direct test of Proposition 3's sign condition (S):

    dB_c/dnu  =  sum_f |W*(f)|^2 * d(Sbar_v(f))/dnu   (condition S)

is positive (a retreat) iff increasing NSI moves averaged noise power
into the passband of the causal Wiener filter |W*(f)|^2.

So far the paper only checks the SIGN of this quantity indirectly, via
the sign of the observed Bc-vs-NSI slope in the MSE domain (positive on
NOIZEUS = no retreat; negative on MS-SNSD = a retreat). This script
computes the condition (S) statistic directly:

  1. For each noise type, fit the pooled (large-sample) Wiener filter
     w* (L=128) as in fir_pooled.py, and take its frequency response
     W*(f) = rfft(w*, NFFT).
  2. For each utterance/instance of that type, compute the noise
     segment's periodogram Sv(f) (Welch-averaged).
  3. Regress Sv(f) against NSI across instances, at each frequency bin,
     to get an empirical estimate of d(Sbar_v(f))/dnu.
  4. S_stat = sum_f |W*(f)|^2 * slope_f(f)  -- the sign of this IS the
     sign condition (S), evaluated directly from data.

Run once for NOIZEUS and once (reconstructing the exact same mixtures
via the deterministic seed) for MS-SNSD, and compare the sign of
S_stat against the already-reported sign of the Bc-vs-NSI slope in the
MSE domain for that corpus.
"""
import os, sys, numpy as np, pandas as pd
from scipy.linalg import solve_toeplitz
from scipy.signal import welch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import measure as m

NFFT = 512  # zero-padded frequency grid for L=128 filters


def xcorr_pooled(a_list, b_list, L):
    num = np.zeros(L); den = 0
    for a, b in zip(a_list, b_list):
        N = len(a)
        A = np.fft.rfft(a, 2*N); B = np.fft.rfft(b, 2*N)
        num += np.fft.irfft(A * np.conj(B))[:L]
        den += N
    return num / den


def pooled_wiener_freq_response(x_list, s_list, L=128, nfft=NFFT):
    rxx = xcorr_pooled(x_list, x_list, L)
    rsx = xcorr_pooled(s_list, x_list, L)
    rxx = rxx.copy(); rxx[0] += 1e-8 * rxx[0]
    w = solve_toeplitz(rxx, rsx)
    W = np.fft.rfft(w, nfft)
    return np.abs(W) ** 2


def welch_psd(v, fs, nfft=NFFT):
    f, p = welch(v, fs=fs, nperseg=min(nfft, len(v)), nfft=nfft)
    return p  # length nfft//2+1, matches np.fft.rfft grid


def condition_S_for_type(x_list, s_list, v_list, nsi_list, fs, L=128, nfft=NFFT):
    W2 = pooled_wiener_freq_response(x_list, s_list, L=L, nfft=nfft)
    P = np.array([welch_psd(v, fs, nfft=nfft) for v in v_list])  # (n_utt, n_freq)
    nsi = np.array(nsi_list)
    nsi_c = nsi - nsi.mean()
    denom = np.sum(nsi_c ** 2)
    slope_f = (P - P.mean(0)).T @ nsi_c / denom  # (n_freq,) least-squares slope per bin
    S_stat = np.sum(W2 * slope_f)
    # normalise by ||W||^2 * ||noise power|| scale so numbers are comparable
    # across corpora; report both raw and normalised.
    S_stat_norm = S_stat / (np.sum(W2) * np.mean(P))
    return S_stat, S_stat_norm, W2, slope_f


def run_noizeus():
    rows = []
    for nt in m.TYPES:
        xs, ss, vs, nsis = [], [], [], []
        for k in range(1, 31):
            s, x, v = m.load(15, f'sp{k:02d}', nt)
            xs.append(x); ss.append(s); vs.append(v); nsis.append(m.nsi_std(v))
        S_stat, S_norm, W2, slope_f = condition_S_for_type(xs, ss, vs, nsis, fs=8000)
        rows.append(dict(corpus='NOIZEUS', noise_type=nt, S_stat=S_stat, S_stat_norm=S_norm))
        print(f'NOIZEUS {nt:12s} S_stat={S_stat:+.4e}  S_norm={S_norm:+.4f}')
    return pd.DataFrame(rows)


def run_msnsd():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'msnsd_validate'))
    import msnsd_validate as mv
    pools = mv.load_noise_pools()
    speech_files = mv.pick_speech(mv.N_SPEECH, seed=0)
    rows = []
    for cat in mv.CATS:
        xs, ss, vs, nsis = [], [], [], []
        for sf_path in speech_files:
            import soundfile as sf
            s, fs = sf.read(sf_path)
            sp_id = os.path.basename(sf_path).replace('.wav', '')
            x, v = mv.mix_at_snr(s, pools[cat], mv.SNR_DB, seed=mv.det_seed(sp_id, cat))
            xs.append(x); ss.append(s); vs.append(v); nsis.append(mv.nsi_std(v))
        S_stat, S_norm, W2, slope_f = condition_S_for_type(xs, ss, vs, nsis, fs=16000, nfft=1024)
        rows.append(dict(corpus='MS-SNSD', noise_type=cat, S_stat=S_stat, S_stat_norm=S_norm))
        print(f'MS-SNSD {cat:22s} S_stat={S_stat:+.4e}  S_norm={S_norm:+.4f}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    d1 = run_noizeus()
    print()
    d2 = run_msnsd()
    d = pd.concat([d1, d2], ignore_index=True)
    d.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_condition_S.csv'), index=False)
    print()
    for corpus, g in d.groupby('corpus'):
        pos = (g.S_stat_norm > 0).sum(); n = len(g)
        print(f'{corpus}: {pos}/{n} noise types have S_stat > 0 (predicted retreat); '
              f'mean S_norm = {g.S_stat_norm.mean():+.4f}')
