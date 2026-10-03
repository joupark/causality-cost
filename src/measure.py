import os
"""
Re-measure NOIZEUS quantities with the author's released code.

Per utterance (7 noise types x 30 sentences per SNR; train noise excluded):
  nsi_paper : 10*log10(1 + Var[e]/E[e]^2), e = 20 ms frame energies of v   (paper eq. 20)
  nsi_std   : std of 10*log10(e)  (definition used in run_vbd.py, nsi_f)
  bc_sig    : signal-estimation FIR Wiener-Hopf, L=128, Toeplitz from
              time-averaged correlations, scored with segsnr_metrics
              (Hamming, 10 ms hop, VAD -20 dB)  -> paper's stated method
  bc_vbd    : run_vbd.py causal_wb (noise-cancellation form, L=256) scored
              with run_vbd.py seg_snr (no window, no VAD), 20 ms frames
  nlms, mmse, omlsa : released baselines, paper metric (15 dB only for nlms)
"""
import sys, csv, time
import numpy as np, soundfile as sf
from pathlib import Path
from scipy.linalg import toeplitz, solve
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines'))
from segsnr_metrics import compute_delta_segsnr
from nlms_baseline import nlms_line_enhancer
from mmse_wiener_baseline import mmse_wiener
from omlsa_baseline import omlsa

ROOT = Path(os.environ.get('NOIZEUS_DIR', 'noizeus'))
FS = 8000
TYPES = ['car', 'station', 'exhibition', 'airport', 'babble', 'street', 'restaurant']
FR = int(0.020 * FS)


def nsi_paper(v):
    n = len(v) // FR
    e = np.array([np.mean(v[i*FR:(i+1)*FR]**2) for i in range(n)])
    return 10*np.log10(1 + np.var(e)/np.mean(e)**2)


def nsi_std(v):
    n = len(v) // FR
    e = np.array([np.mean(v[i*FR:(i+1)*FR]**2) + 1e-12 for i in range(n)])
    return float(np.std(10*np.log10(e)))


def xcorr(a, b, L):
    """biased time-average r[k] = (1/N) sum_n a(n) b(n-k), k=0..L-1"""
    N = len(a)
    A = np.fft.rfft(a, 2*N); B = np.fft.rfft(b, 2*N)
    r = np.fft.irfft(A*np.conj(B))[:L] / N
    return r


def bc_signal(s, x, L=128):
    rxx = xcorr(x, x, L)
    rsx = xcorr(s, x, L)          # E[s(n) x(n-k)]
    R = toeplitz(rxx) + np.eye(L)*1e-8*rxx[0]
    w = solve(R, rsx, assume_a='pos')
    shat = fftconvolve(x, w)[:len(x)]
    mse = np.mean(s**2) - rsx @ w
    return shat, mse


def seg_snr_vbd(cl, no, enh, FRAME=FR):
    L = min(len(cl), len(no), len(enh))
    def _s(a, b):
        r = []
        for i in range(L // FRAME):
            st = i*FRAME; af = a[st:st+FRAME]; nf = af - b[st:st+FRAME]
            r.append(np.clip(10*np.log10((np.mean(af**2)+1e-12)/(np.mean(nf**2)+1e-12)), -10, 35))
        return float(np.mean(r)) if r else 0.
    return _s(cl[:L], enh[:L]) - _s(cl[:L], no[:L])


def causal_wb_vbd(no, cl, L=256):
    N = min(len(no), len(cl))
    x = no[:N].astype(np.float64); d = (no[:N]-cl[:N]).astype(np.float64)
    X = np.fft.rfft(x, n=2*N); D = np.fft.rfft(d, n=2*N)
    Rxx = np.fft.irfft(np.abs(X)**2)/N; Rxd = np.fft.irfft(X*np.conj(D))/N
    R = toeplitz(Rxx[:L]); r = Rxd[:L]; R += np.eye(L)*1e-6*(np.max(np.abs(R))+1e-10)
    h = solve(R, r)
    return seg_snr_vbd(cl[:N], no[:N], (no[:N]-fftconvolve(x, h, mode='full')[:N]))


def load(snr, sp, nt):
    s, _ = sf.read(ROOT/'clean'/f'{sp}.wav')
    x, _ = sf.read(ROOT/f'{snr}dB'/f'{sp}_{nt}_sn{snr}.wav')
    n = min(len(s), len(x)); s = s[:n]; x = x[:n]
    return s, x, x - s


if __name__ == '__main__':
    snrs = [int(a) for a in sys.argv[1].split(',')]
    do_nlms = len(sys.argv) > 2 and sys.argv[2] == 'nlms'
    out = Path(f'results_{"_".join(map(str,snrs))}{"_nlms" if do_nlms else ""}.csv')
    rows = []; t0 = time.time()
    for snr in snrs:
        for nt in TYPES:
            for k in range(1, 31):
                sp = f'sp{k:02d}'
                s, x, v = load(snr, sp, nt)
                shat, mse = bc_signal(s, x, 128)
                row = dict(snr=snr, noise_type=nt, speaker=sp,
                           nsi_paper=nsi_paper(v), nsi_std=nsi_std(v),
                           bc_sig=compute_delta_segsnr(s, x, shat, fs=FS),
                           bc_sig_mse_db=10*np.log10(mse/np.mean(s**2)),
                           bc_sig_vbdmetric=seg_snr_vbd(s, x, shat),
                           bc_vbd=causal_wb_vbd(x, s),
                           mmse=compute_delta_segsnr(s, x, mmse_wiener(x, v, n_fft=256, fs=FS), fs=FS),
                           omlsa=compute_delta_segsnr(s, x, omlsa(x, v, n_fft=256, fs=FS), fs=FS))
                if do_nlms:
                    enh, _ = nlms_line_enhancer(x, L=128, mu=0.05, eps=1e-3, delay=1)
                    row['nlms'] = compute_delta_segsnr(s, x, enh, fs=FS)
                rows.append(row)
        print(f'SNR {snr} done, {time.time()-t0:.0f}s', flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('wrote', out)
