"""External validation on a second corpus: MS-SNSD (Microsoft Scalable Noisy
Speech Dataset), 16 kHz, real-world noise recordings, MIT license
(github.com/microsoft/MS-SNSD). Independent of NOIZEUS (different speech
content, different noise recordings, different sampling rate).

Design mirrors the NOIZEUS analysis exactly so the same cluster-robust
methodology applies: N=40 clean speech clips (fixed seed) are each mixed,
at SNR=15 dB, with a noise segment drawn from each of 9 real noise
categories -- the same speech content reused across noise categories, as
in NOIZEUS (30 sentences x 7 types), which is why "cluster = speech clip"
inference is used identically here.
"""
import os, sys, glob, csv, hashlib, numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))
from segsnr_metrics import compute_delta_segsnr
from nlms_baseline import nlms_line_enhancer
from mmse_wiener_baseline import mmse_wiener, stft
from omlsa_baseline import omlsa
from scipy.linalg import toeplitz, solve
from scipy.signal import fftconvolve

ROOT = os.environ.get('MSNSD_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MS-SNSD'))
FS = 16000
CATS = ['AirConditioner', 'AirportAnnouncements', 'Babble', 'CopyMachine',
        'Munching', 'Neighbor', 'ShuttingDoor', 'Typing', 'VacuumCleaner']
N_SPEECH = 40
SNR_DB = 15
FR = int(0.020 * FS)


def load_noise_pools():
    pools = {}
    for cat in CATS:
        files = sorted(glob.glob(f'{ROOT}/noise_test/{cat}_*.wav'))
        segs = [sf.read(f)[0] for f in files]
        pools[cat] = np.concatenate(segs).astype(np.float64)
    return pools


def pick_speech(n, seed=0):
    files = sorted(glob.glob(f'{ROOT}/clean_test/clnsp*.wav'))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[i] for i in sorted(idx)]


def det_seed(*parts):
    """Deterministic seed across processes and machines: Python's builtin
    hash() is randomised per-process (PYTHONHASHSEED) for strings, which
    silently broke exact reproducibility of the noise crops in an earlier
    version of this script. hashlib is stable."""
    h = hashlib.md5('_'.join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def mix_at_snr(s, noise_pool, snr_db, seed):
    rng = np.random.default_rng(seed)
    L = len(s)
    start = rng.integers(0, len(noise_pool) - L)
    v = noise_pool[start:start + L].copy()
    ps = np.mean(s ** 2); pv = np.mean(v ** 2)
    target_pv = ps / (10 ** (snr_db / 10))
    v *= np.sqrt(target_pv / (pv + 1e-12))
    x = s + v
    return x, v


def nsi_std(v, fr=FR):
    n = len(v) // fr
    e = np.array([np.mean(v[i*fr:(i+1)*fr]**2) + 1e-12 for i in range(n)])
    return float(np.std(10*np.log10(e)))


def xcorr(a, b, L):
    N = len(a)
    A = np.fft.rfft(a, 2*N); B = np.fft.rfft(b, 2*N)
    return np.fft.irfft(A*np.conj(B))[:L] / N


def bc_signal(s, x, L=128):
    rxx = xcorr(x, x, L); rsx = xcorr(s, x, L)
    R = toeplitz(rxx) + np.eye(L)*1e-8*rxx[0]
    w = solve(R, rsx, assume_a='pos')
    shat = fftconvolve(x, w)[:len(x)]
    return shat


def delta_b(s, x, v):
    n_fft = 512  # 32 ms at 16 kHz, matches the 256-sample/8kHz NOIZEUS setting
    S = stft(s, n_fft, n_fft//2); X = stft(x, n_fft, n_fft//2)
    Ss = np.mean(np.abs(S)**2, 0)
    Pn = np.abs(X - S)**2
    from scipy.ndimage import uniform_filter1d
    Pn_loc = uniform_filter1d(Pn, 3, axis=0)
    Sbar = Pn.mean(0)
    Gc = Ss/(Ss+Sbar+1e-12); Gnc = Ss/(Ss+Pn_loc+1e-12)
    mse_c = np.mean(np.abs(S - Gc*X)**2); mse_nc = np.mean(np.abs(S - Gnc*X)**2)
    return 10*np.log10(mse_c/mse_nc)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=N_SPEECH)
    ap.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_msnsd.csv'))
    args = ap.parse_args()
    pools = load_noise_pools()
    speech_files = pick_speech(N_SPEECH, seed=0)[args.start:args.end]
    rows = []
    for si, sf_path in enumerate(speech_files):
        s, fs = sf.read(sf_path)
        assert fs == FS
        sp_id = os.path.basename(sf_path).replace('.wav', '')
        for cat in CATS:
            x, v = mix_at_snr(s, pools[cat], SNR_DB, seed=det_seed(sp_id, cat))
            shat = bc_signal(s, x, 128)
            n = min(len(s), len(shat))
            bc = compute_delta_segsnr(s[:n], x[:n], shat[:n], fs=FS)
            nlms_out, _ = nlms_line_enhancer(x, L=128, mu=0.05, eps=1e-3, delay=1)
            nlms_g = compute_delta_segsnr(s, x, nlms_out, fs=FS)
            mmse_out = mmse_wiener(x, v, n_fft=512, fs=FS)
            mmse_g = compute_delta_segsnr(s, x, mmse_out, fs=FS)
            omlsa_out = omlsa(x, v, n_fft=512, fs=FS)
            omlsa_g = compute_delta_segsnr(s, x, omlsa_out, fs=FS)
            dB = delta_b(s, x, v)
            rows.append(dict(speaker=sp_id, noise_type=cat, nsi=nsi_std(v),
                              bc_sig=bc, nlms=nlms_g, mmse=mmse_g, omlsa=omlsa_g, dB=dB))
        print(f'{args.start+si+1}/{N_SPEECH} done', flush=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print('wrote', args.out, len(rows), 'rows')
