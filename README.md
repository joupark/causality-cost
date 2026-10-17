# Causality cost and tracking gap under non-stationary noise

Code and per-utterance results for the letter

> J. Park, "Causality Cost and Tracking Gap of Wiener Filtering Under Non-Stationary Noise",
> submitted to *IEEE Signal Processing Letters*, 2026.

The letter separates the gap between the causal Wiener bound `B_c` and a
time-varying non-causal oracle `B_nc` into

```
B_c - B_nc  =  (B_c - B_s)  +  (B_s - B_nc)
               \_________/     \_________/
                C: causality     T: tracking
                    cost              gap
```

where `B_s` is the non-causal LTI (stationary Wiener) bound. With a fixed
time-averaged noise spectrum, `B_c` and `C` do not change with
non-stationarity and only `T` grows; `B_c` itself degrades if and only if
non-stationarity moves averaged noise power into the passband of the causal
Wiener filter (condition (S)). It can equally improve; the derivative also vanishes when the spectral
redistribution is orthogonal to the filter's squared response, which makes `B_c`
stationary in the non-stationarity parameter, not necessarily constant.

Every number and figure in the letter is produced by the scripts below, and
nothing else is included: the repository holds only what reproduces the letter
from the two public corpora.

## Requirements

```
python >= 3.10
numpy scipy pandas matplotlib soundfile statsmodels
```

`pip install -r requirements.txt`. CPU only; no GPU is needed. The scripts in
`src/` run in minutes.

The results committed here were produced with Python 3.12.3, NumPy 1.26.4,
SciPy 1.17.1, pandas 3.0.2, statsmodels 0.15.0 and matplotlib 3.10.8, on a
single CPU core (Intel Xeon, 2.1 GHz).

## Data (not redistributed here)

| Corpus | Where to get it | Environment variable |
|---|---|---|
| NOIZEUS (8 kHz) | https://ecs.utdallas.edu/loizou/speech/noizeus/ | `NOIZEUS_DIR` |
| MS-SNSD (16 kHz) | https://github.com/microsoft/MS-SNSD | `MSNSD_DIR` |

```bash
export NOIZEUS_DIR=/path/to/noizeus     # contains clean/, 0dB/, 5dB/, 10dB/, 15dB/
export MSNSD_DIR=/path/to/MS-SNSD       # contains clean_test/, noise_test/
```

NOIZEUS ships eight noise types, but has no 15 dB archive for suburban-train
noise, so the 15 dB analyses use the other seven; `src/eight_types/` rechecks
the conclusions at 0/5/10 dB with all eight.

## Reproducing the letter

| Letter item | Command (from `src/`) |
|---|---|
| Fig. 1, Sec. III-B (controlled modulation; `T` vs. the closed-form rate, `B_c`/`B_s` constancy, condition (S)) | `cd controlled_modulation && CM_SNR=15 python3 controlled_modulation.py A && CM_SNR=0 python3 controlled_modulation.py A && CM_SNR=15 python3 controlled_modulation.py B && python3 controlled_numbers.py` |
| Fig. 2, Table II NOIZEUS column, the variance decomposition behind Sec. III-A's clustering choice | `python3 noizeus_measure.py` (measures all 4 SNRs from audio; ~4 min) |
| Table II MS-SNSD column | `cd msnsd_validate && bash run_msnsd.sh` |
| Table II condition (S) row | `cd condition_S && python3 condition_S_test.py` |
| Sec. III-A eight-type check | `cd eight_types && python3 eight_types.py` |
| Sec. III-B generality (24 configurations, 16 kHz carriers), exact Jensen gap in Fig. 1a; Sec. III-C stationary-surrogate null in Fig. 2a | see `src/robustness/README.md` |

Each folder has a README with its own notes and the expected output. Results
are written next to the scripts as `results_*.csv` and `numbers_*.txt`; the
versions used in the letter are committed, so you can diff against them.

## Notes on reproducibility

* All random draws use `hashlib`-derived seeds, not Python's `hash()`, which is
  randomised per process.
* Predictions for the controlled experiments were written into the scripts'
  docstrings before the results were computed.
* `src/baselines/mmse_wiener_baseline.py` guards the overlap-add
  normalisation at the signal edges. Without the guard the first and last
  samples contain spikes; this is invisible to segmental SNR but corrupts
  waveform MSE.
* Statistics use cluster-robust standard errors (cluster = speech item),
  because both corpora reuse the same speech across noise types.

## License

Code: MIT (see `LICENSE`). The NOIZEUS and MS-SNSD corpora are distributed by
their respective owners under their own terms and are not included here.

## Figures

`src/spl_figs.py` regenerates Fig. 1 and Fig. 2 of the letter from the
committed CSVs and writes them next to itself:

```
python3 src/spl_figs.py
```
