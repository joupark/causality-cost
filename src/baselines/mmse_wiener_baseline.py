"""
MMSE-Wiener Filter Baseline (Oracle PSD)
=========================================

Causal STFT-domain MMSE estimator with decision-directed
a-priori SNR estimation (Ephraim & Malah, 1984).
Used in Table 1/2 of the paper as an oracle-PSD baseline.

The frame-wise Wiener gain is:

    G(t, f) = xi(t, f) / (1 + xi(t, f))

where xi is the a-priori SNR. With the ORACLE noise PSD
(computed from the isolated noise signal), this is the
optimal static-gain-form estimator under the stationarity
assumption.

Decision-directed a-priori SNR (Ephraim-Malah):

    xi(t,f) = alpha * |S_hat(t-1,f)|^2 / S_v(f)
            + (1 - alpha) * max(gamma(t,f) - 1, 0)

with gamma = |X|^2 / S_v the a-posteriori SNR.

Paper settings:
    n_fft = 256 (8 kHz)  -> 32 ms frame, 32 ms latency
    alpha = 0.98         (decision-directed smoothing)
    Oracle noise PSD     (from clean reference's noise)

NOTE: MMSE-Wiener gap to B_c in Table 1 is +0.91 dB
(B_c - Delta SegSNR = -0.93 - (-1.84)). The static gain
form cannot track a drifting noise PSD, so it falls
below B_c by an amount that grows with NSI (Proposition 1,
within-noise slope +0.42 dB/dB in Table 3).
"""
import numpy as np


def stft(x, n_fft, hop):
    """Short-time Fourier transform with Hann window."""
    win = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    X = np.zeros((n_frames, n_fft // 2 + 1), dtype=complex)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + n_fft] * win
        X[i] = np.fft.rfft(seg)
    return X


def istft(X, n_fft, hop, length):
    """Inverse STFT with overlap-add (Hann window)."""
    win = np.hanning(n_fft)
    x = np.zeros(length)
    wsum = np.zeros(length)
    for i in range(X.shape[0]):
        seg = np.fft.irfft(X[i], n=n_fft) * win
        start = i * hop
        end = min(start + n_fft, length)
        x[start:end] += seg[:end - start]
        wsum[start:end] += win[:end - start] ** 2
    wsum[wsum < 1e-8] = 1.0
    # Guard the overlap-add normalisation: near the signal edges the summed
    # squared window is ~0, and dividing by it produced large spikes in the
    # first/last samples (found 2026-09-11; invisible to SegSNR, which drops
    # low-energy frames, but it corrupted waveform MSE).
    ok = wsum > 1e-3 * wsum.max()
    return np.where(ok, x / np.where(ok, wsum, 1.0), x)


def oracle_noise_psd(noise, n_fft, hop):
    """
    Compute the ORACLE noise PSD from the isolated noise
    signal (time-averaged over all frames).

    This is the 'oracle' information the paper uses to
    isolate structural limitations from estimation error.
    """
    V = stft(noise, n_fft, hop)
    return np.mean(np.abs(V) ** 2, axis=0)  # average PSD


def mmse_wiener(noisy, noise, n_fft=256, hop=None,
                alpha=0.98, fs=8000):
    """
    Causal MMSE-Wiener enhancement with oracle noise PSD.

    Parameters
    ----------
    noisy : ndarray
        Noisy observation x(n) = s(n) + v(n).
    noise : ndarray
        Isolated noise v(n) for the ORACLE PSD.
    n_fft : int
        FFT / frame length. Paper uses 256 at 8 kHz (32 ms).
    hop : int or None
        Hop size. Defaults to n_fft // 2 (50% overlap).
    alpha : float
        Decision-directed smoothing factor (paper: 0.98).
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
    prev_mag2 = np.zeros(n_bins)

    for t in range(n_frames):
        X_mag2 = np.abs(X[t]) ** 2

        # A-posteriori SNR
        gamma = X_mag2 / S_v

        # Decision-directed a-priori SNR
        if t == 0:
            xi = np.maximum(gamma - 1, 0.0)
        else:
            xi = (alpha * prev_mag2 / S_v
                  + (1 - alpha) * np.maximum(gamma - 1, 0.0))
        xi = np.maximum(xi, 1e-6)

        # Wiener gain (static-gain form)
        G = xi / (1 + xi)

        S_hat[t] = G * X[t]
        prev_mag2 = np.abs(S_hat[t]) ** 2

    return istft(S_hat, n_fft, hop, len(noisy))


def mmse_wiener_latency_ms(n_fft=256, fs=8000):
    """Algorithmic latency = one analysis frame."""
    return n_fft / fs * 1000.0


def evaluate_mmse_wiener(clean, noisy, noise, fs=8000,
                         n_fft=256, segsnr_fn=None):
    """
    Run MMSE-Wiener and compute its Delta SegSNR.

    Parameters
    ----------
    clean : ndarray
        Clean reference (target).
    noisy : ndarray
        Noisy observation.
    noise : ndarray
        Isolated noise for the oracle PSD.
    fs : int
        Sample rate.

    Returns
    -------
    delta_segsnr : float
        SegSNR(enhanced) - SegSNR(noisy) in dB.
    enhanced : ndarray
        MMSE-Wiener output.
    """
    enhanced = mmse_wiener(noisy, noise, n_fft=n_fft, fs=fs)

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

    enhanced = mmse_wiener(noisy, noise, n_fft=256, fs=fs)

    warm = 2000
    snr_in = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((noisy[warm:] - clean[warm:]) ** 2))
    snr_out = 10 * np.log10(
        np.mean(clean[warm:] ** 2)
        / np.mean((enhanced[warm:] - clean[warm:]) ** 2))

    print("MMSE-Wiener (oracle PSD, n_fft=256, alpha=0.98)")
    print(f"  Latency    : {mmse_wiener_latency_ms():.1f} ms")
    print(f"  Input SNR  : {snr_in:+.2f} dB")
    print(f"  Output SNR : {snr_out:+.2f} dB")
    print(f"  SNR change : {snr_out - snr_in:+.2f} dB")
