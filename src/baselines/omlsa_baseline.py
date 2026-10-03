"""
OMLSA Baseline (Oracle PSD)
============================

Optimally-Modified Log-Spectral Amplitude (OMLSA)
estimator with MCRA-based speech-presence probability
(Cohen, 2003). Used in Table 1/2 of the paper as an
oracle-PSD baseline.

OMLSA improves on MMSE-Wiener by weighting the spectral
gain with a speech-presence probability p(t,f):

    G_OMLSA(t,f) = G_LSA(t,f)^p(t,f) * G_min^(1-p(t,f))

where G_LSA is the log-spectral-amplitude (Ephraim-Malah
LSA) gain and G_min is a spectral floor. This time-varying
gating relaxes the strict stationarity assumption of the
plain Wiener filter.

Paper settings:
    n_fft = 256 (8 kHz)  -> 32 ms frame, 32 ms latency
    alpha = 0.92         (a-priori SNR smoothing)
    G_min = -25 dB       (spectral floor)
    Oracle noise PSD

NOTE: OMLSA gap to B_c in Table 1 is -0.14 dB
(B_c - Delta SegSNR = -0.93 - (-0.79)); the negative gap
means OMLSA's time-varying gain slightly EXCEEDS the
static-LTI reference B_c at low NSI. Its within-noise NSI
slope is +0.27 dB/dB (Table 3) -- still positive, so it
also falls below B_c as NSI grows (Proposition 2).
"""
import numpy as np
from scipy.special import exp1  # exponential integral E_1

# Reuse STFT helpers from the MMSE-Wiener module
from mmse_wiener_baseline import stft, istft, oracle_noise_psd


def lsa_gain(xi, gamma):
    """
    Ephraim-Malah Log-Spectral Amplitude gain.

        G_LSA = (xi/(1+xi)) * exp( 0.5 * E_1(nu) )
        nu = (xi/(1+xi)) * gamma
    """
    xi = np.maximum(xi, 1e-6)
    ratio = xi / (1 + xi)
    nu = ratio * gamma
    nu = np.clip(nu, 1e-6, 500)  # numerical range for exp1
    return ratio * np.exp(0.5 * exp1(nu))


def omlsa(noisy, noise, n_fft=256, hop=None, alpha=0.92,
          g_min_db=-25.0, fs=8000):
    """
    Causal OMLSA enhancement with oracle noise PSD.

    Parameters
    ----------
    noisy : ndarray
        Noisy observation x(n) = s(n) + v(n).
    noise : ndarray
        Isolated noise v(n) for the ORACLE PSD.
    n_fft : int
        FFT / frame length. Paper uses 256 at 8 kHz (32 ms).
    hop : int or None
        Hop size. Defaults to n_fft // 2.
    alpha : float
        A-priori SNR (decision-directed) smoothing.
    g_min_db : float
        Spectral floor in dB (paper: -25 dB).
    fs : int
        Sample rate.

    Returns
    -------
    enhanced : ndarray
        Enhanced time-domain signal.
    """
    if hop is None:
        hop = n_fft // 2

    X = stft(noisy, n_fft, hop)
    S_v = oracle_noise_psd(noise, n_fft, hop) + 1e-10

    n_frames, n_bins = X.shape
    S_hat = np.zeros_like(X)
    G_min = 10 ** (g_min_db / 20)

    prev_mag2 = np.zeros(n_bins)
    # Smoothed a-priori speech-presence quantities
    q = 0.5 * np.ones(n_bins)   # a-priori speech-absence prob
    xi_smooth = np.ones(n_bins)

    for t in range(n_frames):
        X_mag2 = np.abs(X[t]) ** 2
        gamma = X_mag2 / S_v

        # Decision-directed a-priori SNR
        if t == 0:
            xi = np.maximum(gamma - 1, 0.0)
        else:
            xi = (alpha * prev_mag2 / S_v
                  + (1 - alpha) * np.maximum(gamma - 1, 0.0))
        xi = np.maximum(xi, 1e-6)

        # --- MCRA-style speech-presence probability ---
        # Likelihood ratio under speech-present vs absent
        ratio = xi / (1 + xi)
        # Clip the exponent to avoid overflow (numerically safe)
        exponent = np.clip(gamma * ratio, 0, 50)
        Lambda = (1.0 / (1 + xi)) * np.exp(exponent)
        # Smooth the a-priori speech-absence probability q
        q_inst = 1.0 / (1.0 + Lambda)
        q = 0.9 * q + 0.1 * np.clip(q_inst, 0.01, 0.99)
        # Speech-presence probability p
        neg_exp = np.clip(-gamma * ratio, -50, 0)
        p = 1.0 / (1.0 + (q / (1 - q)) * (1 + xi)
                   * np.exp(neg_exp))
        p = np.clip(p, 0.0, 1.0)

        # LSA gain, then OMLSA modification
        G_lsa = lsa_gain(xi, gamma)
        G = (G_lsa ** p) * (G_min ** (1 - p))

        S_hat[t] = G * X[t]
        prev_mag2 = np.abs(S_hat[t]) ** 2

    return istft(S_hat, n_fft, hop, len(noisy))


def omlsa_latency_ms(n_fft=256, fs=8000):
    """Algorithmic latency = one analysis frame."""
    return n_fft / fs * 1000.0


def evaluate_omlsa(clean, noisy, noise, fs=8000,
                   n_fft=256, segsnr_fn=None):
    """
    Run OMLSA and compute its Delta SegSNR.

    Parameters
    ----------
    clean, noisy, noise : ndarray
        Clean target, noisy observation, isolated noise.
    fs : int
        Sample rate.

    Returns
    -------
    delta_segsnr : float
        SegSNR(enhanced) - SegSNR(noisy) in dB.
    enhanced : ndarray
        OMLSA output.
    """
    enhanced = omlsa(noisy, noise, n_fft=n_fft, fs=fs)

    if segsnr_fn is None:
        from segsnr_metrics import segmental_snr
        segsnr_fn = segmental_snr

    n = min(len(clean), len(noisy), len(enhanced))
    seg_in = segsnr_fn(clean[:n], noisy[:n], fs=fs)
    seg_out = segsnr_fn(clean[:n], enhanced[:n], fs=fs)
    return seg_out - seg_in, enhanced


if __name__ == '__main__':
    np.random.seed(0)
    fs = 8000
    n = fs * 2
    t = np.arange(n) / fs

    clean = (np.sin(2 * np.pi * 220 * t)
             + 0.5 * np.sin(2 * np.pi * 440 * t))
    clean /= clean.std()
    clean[:int(0.15 * fs)] = 0

    sigma_v = clean.std() * 10 ** (-10 / 20)
    noise = sigma_v * np.random.randn(n)
    noisy = clean + noise

    enhanced = omlsa(noisy, noise, n_fft=256, fs=fs)

    warm = 2000
    snr_in = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((noisy[warm:] - clean[warm:]) ** 2))
    snr_out = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((enhanced[warm:] - clean[warm:]) ** 2))

    print("OMLSA (oracle PSD, n_fft=256, alpha=0.92, "
          "Gmin=-25 dB)")
    print(f"  Latency    : {omlsa_latency_ms():.1f} ms")
    print(f"  Input SNR  : {snr_in:+.2f} dB")
    print(f"  Output SNR : {snr_out:+.2f} dB")
    print(f"  SNR change : {snr_out - snr_in:+.2f} dB")
