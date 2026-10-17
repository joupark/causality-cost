# Controlled modulation experiment (Section III-B, Fig. 1)

Tests Propositions 1-2 and Lemma 1 causally by imposing non-stationarity on
stationary noise instead of observing it. Predictions are written in the
docstring of `controlled_modulation.py` (before the results were computed).

    export NOIZEUS_DIR=/path/to/noizeus
    CM_SNR=15 python3 controlled_modulation.py A   # fixed averaged spectrum, 15 dB
    CM_SNR=0  python3 controlled_modulation.py A   # same at 0 dB
    CM_SNR=15 python3 controlled_modulation.py B   # redistributed spectrum
    python3 controlled_numbers.py                  # numbers_controlled.txt, ../figures/fig_controlled.pdf

About 1 minute in total. Seeds are hashlib-based (deterministic).
Results are summarised in numbers_controlled.txt and Section III-B of the letter.

Added (A4): results_partA_*.csv now also record bs_norm (MSE of the time-invariant
non-causal gain, i.e. B_s) and oracle_norm (B_nc). B_s and C = B_c - B_s do not change
with nu (15 dB: +0.03%, -1.4% per unit nu, p=0.72, 0.35; 0 dB: +0.09%, -1.0%,
p=0.44, 0.10); the oracle MSE falls 4.3% and 9.4% per unit nu. All previously reported
columns reproduce exactly.
