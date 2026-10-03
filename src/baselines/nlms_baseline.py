"""
NLMS Adaptive Line Enhancer (Baseline)
=======================================

128-tap Normalized Least Mean Squares (NLMS) adaptive
line enhancer, as used in Table 1/2 of the paper.

An adaptive line enhancer (ALE) predicts the current
sample from the previous L samples of the noisy input:

    s_hat(n) = w(n)^T x(n-1:n-L)

Speech is locally predictable (highly correlated), so the
predictor passes speech while the (less correlated) noise
component is suppressed. The NLMS update is:

    e(n) = x(n) - s_hat(n)              # prediction error
    w(n+1) = w(n) + mu * e(n) * x_vec(n) / (||x_vec(n)||^2 + eps)

Reference: Haykin, "Adaptive Filter Theory" (Ch. 6),
Widrow & Stearns, "Adaptive Signal Processing".

Paper settings:
    L   = 128 taps
    mu  = 0.05  (step size)
    eps = 1e-3  (regularization)
    fs  = 8000 Hz  -> latency = L/fs = 16 ms

NOTE on the result in the paper:
    The NLMS gap to B_c in Table 1/2 is +3.85 dB
    (B_c - Delta SegSNR = -0.93 - (-4.78)).
    NLMS is a *line enhancer*, not a noise-PSD method;
    it falls well below the static-LTI reference because
    its time-varying weights cannot match the structural
    optimum under non-stationary noise (Proposition 1).
"""
import numpy as np


def nlms_line_enhancer(noisy, L=128, mu=0.05, eps=1e-3,
                       delay=1):
    """
    Apply an NLMS adaptive line enhancer to a noisy signal.

    Parameters
    ----------
    noisy : ndarray, shape (T,)
        Noisy observation x(n) = s(n) + v(n).
    L : int
        Filter length (number of taps). Paper uses 128.
    mu : float
        NLMS step size (0 < mu < 2 for stability). Paper: 0.05.
    eps : float
        Small regularization to avoid division by zero.
    delay : int
        Decorrelation delay (samples) between the reference
        and the predicted sample. delay=1 is the standard
        one-step linear predictor used in the paper.

    Returns
    -------
    enhanced : ndarray, shape (T,)
        Enhanced (speech-predicted) signal s_hat(n).
    error : ndarray, shape (T,)
        Prediction-error signal e(n) (the suppressed part).
    """
    x = np.asarray(noisy, dtype=np.float64)
    n_samples = len(x)

    w = np.zeros(L)
    enhanced = np.zeros(n_samples)
    error = np.zeros(n_samples)

    for n in range(L + delay, n_samples):
        # Regressor: L past samples, offset by the decorrelation delay
        x_vec = x[n - delay: n - delay - L: -1]  # length L, reversed

        # One-step linear prediction
        s_hat = w @ x_vec
        e = x[n] - s_hat

        # NLMS weight update (normalized by input energy)
        norm = x_vec @ x_vec + eps
        w = w + (mu * e / norm) * x_vec

        enhanced[n] = s_hat
        error[n] = e

    return enhanced, error


def nlms_latency_ms(L=128, fs=8000):
    """Algorithmic latency of the NLMS ALE in milliseconds."""
    return L / fs * 1000.0


# ----------------------------------------------------------------
# Evaluation helper: NLMS Delta SegSNR on a (clean, noisy) pair
# ----------------------------------------------------------------

def evaluate_nlms(clean, noisy, fs=8000, L=128, mu=0.05,
                  eps=1e-3, segsnr_fn=None):
    """
    Run NLMS and compute its Delta SegSNR vs the clean target.

    Parameters
    ----------
    clean, noisy : ndarray
        Clean reference and noisy observation.
    fs : int
        Sample rate (8000 for NOIZEUS, 16000 for VBD).
    segsnr_fn : callable or None
        Function (clean, processed, fs) -> SegSNR dB.
        If None, imports from segsnr_metrics (chart_code/).

    Returns
    -------
    delta_segsnr : float
        SegSNR(enhanced) - SegSNR(noisy) in dB.
    enhanced : ndarray
        The NLMS output.
    """
    enhanced, _ = nlms_line_enhancer(noisy, L=L, mu=mu, eps=eps)

    if segsnr_fn is None:
        # Expect segsnr_metrics.py to be importable
        from segsnr_metrics import segmental_snr
        segsnr_fn = segmental_snr

    n = min(len(clean), len(noisy), len(enhanced))
    seg_in = segsnr_fn(clean[:n], noisy[:n], fs=fs)
    seg_out = segsnr_fn(clean[:n], enhanced[:n], fs=fs)
    return seg_out - seg_in, enhanced


# ----------------------------------------------------------------
# Example / self-test
# ----------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)
    fs = 8000
    n = fs * 2
    t = np.arange(n) / fs

    # Clean: sum of harmonics (predictable, speech-like)
    clean = (np.sin(2 * np.pi * 200 * t)
             + 0.5 * np.sin(2 * np.pi * 400 * t))
    clean /= clean.std()

    # Additive white noise at 5 dB SNR (white = unpredictable)
    snr_db = 5.0
    sigma_v = clean.std() * 10 ** (-snr_db / 20)
    noise = sigma_v * np.random.randn(n)
    noisy = clean + noise

    enhanced, error = nlms_line_enhancer(noisy, L=128,
                                         mu=0.05, eps=1e-3)

    # Crude SNR before/after (skip warm-up region)
    warm = 2000
    snr_in = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((noisy[warm:] - clean[warm:]) ** 2))
    snr_out = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((enhanced[warm:] - clean[warm:]) ** 2))

    print(f"NLMS line enhancer (L=128, mu=0.05)")
    print(f"  Latency       : {nlms_latency_ms():.1f} ms")
    print(f"  Input SNR     : {snr_in:+.2f} dB")
    print(f"  Output SNR    : {snr_out:+.2f} dB")
    print(f"  SNR change    : {snr_out - snr_in:+.2f} dB")
    print()
    print("  (ALE passes the predictable harmonic speech and")
    print("   attenuates the white, unpredictable noise.)")
