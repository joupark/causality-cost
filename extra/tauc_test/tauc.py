"""Coherence time tau_c of real noise, following Definition 1.

Operationalisation (fixed before any analysis against the outcomes):
  * 32 ms Hann frames, 16 ms hop; periodogram summed into 16 equal bands
    (DC excluded) -> band spectrum S(n, b).
  * D(T) = mean_n  sum_b |S(n+T,b) - S(n,b)| / (0.5 sum_b [S(n,b)+S(n+T,b)]),
    T = 1..Tmax hops (symmetric denominator; see note in dcurve).
  * Finite-sample estimation noise: the same D(T) is computed for
    phase-randomised surrogates of the same noise (same average spectrum,
    no non-stationarity), averaged over 3 surrogates -> D_surr(T).
  * tau_c = first lag T with D(T) - D_surr(T) >= eps0 (eps0 = 0.1 primary;
    0.05 and 0.2 reported as sensitivity), times 16 ms. Tmax = 1 s (62 hops,
    or half the utterance if shorter); if never reached, tau_c = Tmax and the
    value is flagged as censored.

PREDICTIONS (written before running the analysis):
  Q1 floor height (min over C of the reference-noise excess MSE) decreases
     with tau_c; theory: log-log slope -2/3.
  Q2 optimal context C* increases with tau_c; theory: log-log slope +2/3.
  Q3 plateau-context excess MSE of both trained networks decreases with tau_c.
  Q4 tau_c explains Q1-Q3 better than the NSI.
"""
import os, sys, hashlib, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
NB, EPS = 16, [0.05, 0.1, 0.2]


def rng(*p):
    return np.random.default_rng(int(hashlib.md5('_'.join(map(str, p)).encode()).hexdigest()[:8], 16))


def band_spec(v, fs):
    N = int(0.032 * fs); H = N // 2; w = np.hanning(N)
    nfr = 1 + (len(v) - N) // H
    fr = np.stack([v[i * H:i * H + N] * w for i in range(nfr)])
    P = np.abs(np.fft.rfft(fr, axis=1))[:, 1:] ** 2
    edges = np.linspace(0, P.shape[1], NB + 1).astype(int)
    return np.stack([P[:, edges[b]:edges[b + 1]].sum(1) for b in range(NB)], 1)


def dcurve(S, Tmax):
    # symmetric normalisation: the first run divided by ||S(n)|| alone, which is
    # zero in digitally silent MS-SNSD segments; fixed before any outcome analysis
    return np.array([np.mean(np.abs(S[T:] - S[:-T]).sum(1) /
                             (0.5 * (S[T:].sum(1) + S[:-T].sum(1)) + 1e-12)) for T in range(1, Tmax + 1)])


def surrogate(v, r):
    V = np.fft.rfft(v); ph = r.uniform(0, 2 * np.pi, len(V)); ph[0] = 0
    if len(v) % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(V) * np.exp(1j * ph), n=len(v))


def tauc(v, fs, key):
    S = band_spec(v, fs); Tmax = min(62, S.shape[0] // 2)
    D = dcurve(S, Tmax)
    Ds = np.mean([dcurve(band_spec(surrogate(v, rng(key, i)), fs), Tmax) for i in range(3)], 0)
    Dst = D - Ds
    out = {}
    for e in EPS:
        hit = np.nonzero(Dst >= e)[0]
        out[f'tauc_{e}'] = (hit[0] + 1 if len(hit) else Tmax) * 0.016
        out[f'cens_{e}'] = int(len(hit) == 0)
    out['Dstruct_1s'] = float(Dst[-1])
    return out


if __name__ == '__main__':
    which = sys.argv[1]
    rows = []
    if which == 'noizeus':
        import measure as m
        for nt in m.TYPES:
            for k in range(1, 31):
                s, x, v = m.load(15, f'sp{k:02d}', nt)
                rows.append(dict(noise_type=nt, speaker=f'sp{k:02d}', **tauc(v, 8000, (nt, k))))
    else:  # MS-SNSD test mixtures of Network 2
        sys.path.insert(0, os.path.join(HERE, '..', 'nn_big'))
        import soundfile as sf, train_nn_big as t
        pools = t.mv.load_noise_pools(); _, _, test = t.split_clips(seed=0)
        for p in test:
            s, fs = sf.read(p); sp = os.path.basename(p).replace('.wav', '')
            for cat in t.mv.CATS:
                x, v = t.mv.mix_at_snr(s, pools[cat], t.EVAL_SNR, seed=t.mv.det_seed(sp, cat, str(t.EVAL_SNR)))
                rows.append(dict(noise_type=cat, speaker=sp, **tauc(v, 16000, (cat, sp))))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f'tauc_{which}.csv'), index=False)
    d = pd.DataFrame(rows)
    print(d.groupby('noise_type')[['tauc_0.1', 'cens_0.1', 'Dstruct_1s']].mean().round(3).to_string())
