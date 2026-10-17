# Robustness checks (Sec. III-B generality, Sec. III-C oracle null)

Two questions the main experiments leave open, each answered from the audio.

## 1. Is the recorded-noise tracking gap an artefact of the oracle?

The operational oracle of Sec. III-C uses a three-frame local periodogram that
contains the current frame, so it follows the noise *realisation* as well as
any change in the noise *statistics*. `oracle_null.py` (NOIZEUS) and
`msnsd_extra.py null` (MS-SNSD) replace each utterance's noise by a
phase-randomised stationary surrogate with the identical magnitude spectrum,
for which the true tracking gap is zero, and measure the same quantity.

| | NOIZEUS, 15 dB | MS-SNSD, 15 dB |
|---|---|---|
| mean `T`, recorded noise | +0.46 dB | +1.43 dB |
| mean `T`, stationary surrogate (bias) | +0.12 dB | +0.12 dB |
| within-type slope vs. NSI, recorded | +0.253 (p 1.2e-07) | +0.057 (p 1.3e-02) |
| within-type slope, surrogate subtracted | +0.245 (p 8.4e-08) | +0.060 (p 8.9e-03) |
| within-type slope, oracle without the current frame | +0.167 (p 5.6e-07) | +0.022 (p 9.2e-02) |

The scripts also vary the smoothing length (1 to 17 frames) and repeat
everything at 0 dB. On NOIZEUS the trend survives every variant; on MS-SNSD at
15 dB it does not survive the oracle that excludes the current frame.

## 2. Is the controlled result specific to car noise, a gamma envelope and 64 ms?

`controlled_general.py` repeats Part A of `../controlled_modulation` varying
one factor at a time: the carrier (seven NOIZEUS noises), the envelope law
(gamma, lognormal, two-state) and the segment length (32 to 256 ms), at 15 and
0 dB. `msnsd_extra.py ctrl` does the same at 16 kHz with the nine MS-SNSD
noises as carriers. Both also evaluate the exact Jensen gap for the realised
envelope, which needs no small-`nu` assumption (dashed curves in Fig. 1a).

* `B_c` changes by at most 0.33% per unit `nu` in the 24 NOIZEUS
  configurations (all p > 0.05), and by +0.06% pooled over the MS-SNSD
  carriers (p = 0.31).
* The 95% CI of the small-`nu` slope contains `c0` in all 24 NOIZEUS
  configurations.
* Limits: `B_s` drifts by +0.4 to +0.7% per unit `nu` for four of the 24
  (uncorrected p < 0.05). With impulsive MS-SNSD carriers (door shutting,
  munching, typing) at 15 dB the measured slope is 57 to 76% of `c0`: Lemma 1
  assumes a stationary carrier.

## Run

```bash
export NOIZEUS_DIR=/path/to/noizeus ; export MSNSD_DIR=/path/to/MS-SNSD
python3 oracle_null.py
CM_SNR=15 python3 controlled_general.py ; CM_SNR=0 python3 controlled_general.py
python3 robustness_numbers.py > numbers_robustness.txt
for snr in 15 0; do python3 msnsd_extra.py null $snr ; python3 msnsd_extra.py ctrl $snr ; done
python3 msnsd_numbers_extra.py > numbers_msnsd_robustness.txt
```

About 20 minutes on one core. The committed `results_*.csv` and `numbers_*.txt`
were produced with Python 3.6.8, NumPy 1.19.3, SciPy 1.4.1, pandas 1.1.5 and
statsmodels 0.12.2; the same environment reproduces every other CSV in this
repository to within 1e-11.
