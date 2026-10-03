"""
Segmental SNR (SegSNR) and Delta SegSNR Computation
====================================================

Reference: Hansen & Pellom (1998), "An effective quality
evaluation protocol for speech enhancement algorithms"

Standard segmental SNR is computed frame-by-frame on the
clean signal s and the difference (s - s_hat):

    SegSNR(s, s_hat) = (1/N) * Sum_n 10*log10( ||s_n||^2 / ||s_n - s_hat_n||^2 )

with each frame n clipped to [-10 dB, +35 dB] to avoid
silence frames dominating the average.

Delta SegSNR is the improvement over the noisy input:

    Delta SegSNR = SegSNR(s, s_hat) - SegSNR(s, x)

where x is the noisy observation and s_hat is the enhanced signal.

Usage:
    delta = compute_delta_segsnr(clean, noisy, enhanced, fs=8000)
"""
import numpy as np


# Standard parameters (Loizou, "Speech Enhancement: Theory and Practice")
SEG_LEN_MS = 20.0       # frame length in ms
SEG_HOP_MS = 10.0       # 50% overlap
SEGSNR_MIN = -10.0      # clip floor in dB
SEGSNR_MAX = 35.0       # clip ceiling in dB
ENERGY_FLOOR = 1e-10    # avoid log(0)


def frame_signal(x, frame_len, hop):
    """Split signal into overlapping frames using a Hamming window."""
    n_samples = len(x)
    n_frames = max(1, 1 + (n_samples - frame_len) // hop)
    window = np.hamming(frame_len)
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end <= n_samples:
            frames[i] = x[start:end] * window
        else:
            # Last frame zero-padded
            seg = x[start:n_samples]
            frames[i, :len(seg)] = seg * window[:len(seg)]
    return frames


def segmental_snr(clean, processed, fs=8000,
                  seg_len_ms=SEG_LEN_MS, hop_ms=SEG_HOP_MS,
                  clip_min=SEGSNR_MIN, clip_max=SEGSNR_MAX,
                  vad_threshold_db=-20.0):
    """
    Compute frame-averaged segmental SNR (dB).

    Parameters
    ----------
    clean : ndarray, shape (T,)
        Clean reference signal.
    processed : ndarray, shape (T,)
        Signal to be evaluated (e.g., noisy or enhanced).
    fs : int
        Sample rate in Hz.
    seg_len_ms, hop_ms : float
        Frame length and hop in milliseconds.
    clip_min, clip_max : float
        Per-frame SNR clipping bounds in dB.
    vad_threshold_db : float
        Energy threshold (dB) below which frames are excluded
        (silence frames). Relative to peak frame energy.

    Returns
    -------
    segsnr_db : float
        Mean segmental SNR in dB, averaged over voiced frames.
    """
    # Align lengths
    n = min(len(clean), len(processed))
    clean = clean[:n].astype(np.float64)
    processed = processed[:n].astype(np.float64)

    frame_len = int(seg_len_ms * fs / 1000)
    hop = int(hop_ms * fs / 1000)

    # Frame both signals
    s_frames = frame_signal(clean, frame_len, hop)
    p_frames = frame_signal(processed, frame_len, hop)
    diff = s_frames - p_frames

    # Per-frame energies
    sig_energy = np.sum(s_frames ** 2, axis=1)
    noise_energy = np.sum(diff ** 2, axis=1) + ENERGY_FLOOR

    # VAD: keep frames with energy above threshold
    max_energy = sig_energy.max() + ENERGY_FLOOR
    voiced = sig_energy >= max_energy * 10 ** (vad_threshold_db / 10)

    if voiced.sum() == 0:
        return np.nan

    # Per-frame SNR (dB)
    frame_snr = 10 * np.log10(sig_energy[voiced] / noise_energy[voiced])

    # Clip to standard range
    frame_snr = np.clip(frame_snr, clip_min, clip_max)

    return float(frame_snr.mean())


def compute_delta_segsnr(clean, noisy, enhanced, fs=8000, **kwargs):
    """
    Compute Delta SegSNR = SegSNR(enhanced) - SegSNR(noisy).

    Parameters
    ----------
    clean : ndarray
        Clean reference signal.
    noisy : ndarray
        Noisy observation (s + v).
    enhanced : ndarray
        Output of the SE method.
    fs : int
        Sample rate.

    Returns
    -------
    delta_segsnr_db : float
        Improvement in segmental SNR (dB). Positive = better.
    """
    segsnr_in = segmental_snr(clean, noisy, fs=fs, **kwargs)
    segsnr_out = segmental_snr(clean, enhanced, fs=fs, **kwargs)
    return segsnr_out - segsnr_in


# ----------------------------------------------------------------
# Convenience: batch over a list of test utterances
# ----------------------------------------------------------------

def batch_delta_segsnr(triplets, fs=8000, verbose=False):
    """
    Compute Delta SegSNR for a list of (clean, noisy, enhanced) triplets.

    Parameters
    ----------
    triplets : list of (ndarray, ndarray, ndarray)
        Each tuple contains (clean, noisy, enhanced) for one utterance.
    fs : int
        Sample rate (default 8000 for NOIZEUS).

    Returns
    -------
    deltas : ndarray, shape (n_utterances,)
        Delta SegSNR per utterance.
    summary : dict
        {'mean', 'std', 'sem', 'n'}
    """
    deltas = []
    for i, (s, x, sh) in enumerate(triplets):
        d = compute_delta_segsnr(s, x, sh, fs=fs)
        deltas.append(d)
        if verbose and (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(triplets)}]  Delta SegSNR = {d:+.2f} dB')

    deltas = np.array(deltas)
    valid = ~np.isnan(deltas)
    summary = {
        'mean': float(deltas[valid].mean()),
        'std': float(deltas[valid].std()),
        'sem': float(deltas[valid].std() / np.sqrt(valid.sum())),
        'n': int(valid.sum()),
    }
    return deltas, summary


# ----------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------

if __name__ == '__main__':
    import soundfile as sf
    from glob import glob
    from pathlib import Path

    # Example: evaluate one method on NOIZEUS holdout (SNR=15 dB)
    CLEAN_DIR = '/path/to/noizeus/clean'      # clean references
    NOISY_DIR = '/path/to/noizeus/noisy_15dB' # noisy mixtures
    ENH_DIR   = '/path/to/enhanced/conformer' # method output

    triplets = []
    for clean_path in sorted(glob(f'{CLEAN_DIR}/*.wav')):
        name = Path(clean_path).name
        noisy_path = Path(NOISY_DIR) / name
        enh_path = Path(ENH_DIR) / name

        if not (noisy_path.exists() and enh_path.exists()):
            continue

        s,  fs = sf.read(clean_path)
        x,  _  = sf.read(noisy_path)
        sh, _  = sf.read(enh_path)
        triplets.append((s, x, sh))

    deltas, summary = batch_delta_segsnr(triplets, fs=8000, verbose=True)

    print(f"\nResults across {summary['n']} utterances:")
    print(f"  mean Delta SegSNR : {summary['mean']:+.2f} dB")
    print(f"  std               : {summary['std']:.2f} dB")
    print(f"  SEM               : {summary['sem']:.3f} dB")
