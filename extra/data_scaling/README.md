# Controlled data-scaling study (Limitations, Section V-G)

Tests directly whether the trained networks' weak absolute performance
(Section V-G) is a training-data-scale issue, as the cited literature
predicts, rather than a training bug -- by holding everything except the
amount of training data fixed and varying only that.

## Design

Unlike the paper's two-network comparison (Section V-G), which changes
architecture, training corpus, and data amount all at once, this study
holds the following fixed across every run:

- architecture (causal Transformer, d=64, 4 heads, 2 layers)
- hyperparameters (Adam, lr=1e-3, weight decay=1e-4, 10 epochs)
- noise categories, SNRs (0, 15 dB), 3 s crops
- the validation set (20 MS-SNSD speech clips) and test set (20 clips),
  drawn once with a fixed seed and reused, unchanged, at every data size

Only the number of *training* speech clips varies: 16, 32, 64, 100, 150
(14 to 135 minutes of audio).

## Run

    export MSNSD_DIR=/path/to/MS-SNSD
    python3 scaling_study.py --size 16  --epochs 10
    python3 scaling_study.py --size 32  --epochs 10
    python3 scaling_study.py --size 64  --epochs 10
    python3 scaling_study.py --size 100 --epochs 10
    python3 scaling_study.py --size 150 --epochs 10

Each command is resumable (checkpoints to `ckpt_size<N>.pt`); rerun the
same command if it does not finish within your time budget. Results
accumulate in `scaling_results.json`.

## Result (already run; `scaling_results.json`, `scaling_curve.pdf`)

| Training clips | Minutes | Held-out $\Delta$SegSNR (dB) |
|---|---|---|
| 16  | 14.4  | $-5.55$ |
| 32  | 28.8  | $-4.33$ |
| 64  | 57.6  | $-4.60$ |
| 100 | 90.0  | $-4.38$ |
| 150 | 135.0 | $-4.23$ |

Performance is clearly worse at the smallest size (16 clips) than at any
larger size, but the differences among 32-150 clips should **not** be
read as a precise curve shape -- see the important caveat below.

## Reproducibility bug found and fixed (`reproducibility_bug_note.json`)

The original script seeded per-epoch data-augmentation (crop position)
with Python's builtin `hash()` on a tuple containing strings, which is
randomised per process (`PYTHONHASHSEED`) -- the same bug class found
and fixed in `../msnsd_validate/` earlier in this project. It is now
fixed here too (`det_crop_seed`, hashlib-based).

To quantify the impact, the 32-clip condition was rerun after the fix,
everything else identical: it gave $-4.76$\,dB versus the original
$-4.33$\,dB, a run-to-run spread of about $0.4$\,dB. **This is
comparable to or larger than the differences among the 32-150 clip
conditions in the table above.** With only one training run per size (no
repeated seeds), this scaling study supports the coarse conclusion that
very little data (16 clips, 14 min) is clearly worse than more, but
**not** a precise scaling curve or a specific plateau point -- this is
exactly how it is worded in the paper's Limitations paragraph. A proper
scaling-law fit would need multiple seeds per size.

The `16`, `64`, `100`, `150` conditions above are still the original
(pre-fix) numbers; only `32` has been regenerated with the deterministic
fix. Rerunning the current (fixed) script will reproduce the `32`
condition exactly but will give somewhat different numbers (expect a
similar $\pm0.3$-$0.5$\,dB order of difference, not a change in the
qualitative pattern) for `16`, `64`, `100`, `150`, since those were not
regenerated.

This is a single run per size (no repeated seeds), so the exact
non-monotonic step at 64 clips should be read as noise around a real
overall trend, not a precise functional form; with only 5 points the
Spearman correlation between size and test performance (rho=0.70) is not
itself significant (p=0.19). The result is reported in the paper as a
qualitative, order-of-magnitude comparison against the literature, not a
precise scaling law fit.
