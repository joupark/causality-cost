# Second trained-network test of Theorem 1 (larger scale, MS-SNSD)

A second, independent test of Theorem 1's two predicted signatures
(variance term, non-stationarity floor), using a larger causal
Transformer trained on MS-SNSD instead of NOIZEUS, to check whether the
result in `../nn_experiment/` depends on the small scale of that first
experiment.

## Model and data

- Causal Transformer: 3 layers, d=96, 4 heads, 274k parameters (versus
  83k for the NOIZEUS network).
- Training: 150 MS-SNSD speech clips x 9 noise categories x 2 SNRs
  (0, 15 dB), each cropped to 3s -> 2700 training mixtures (6x the
  NOIZEUS network's 448).
- Split is by speech-clip content, sentence-disjoint as in
  `../nn_experiment/`: 150 fit / 20 validation / 20 held-out test clips,
  fixed seed, no overlap.
- Best checkpoint (by validation, epoch 18 of 24) is shipped as
  `best_big.pt` so the reported numbers reproduce without retraining.

## Files

| File | Purpose |
|---|---|
| `train_nn_big.py` | model, data pipeline, training loop |
| `best_big.pt` | best validation checkpoint (epoch 18) |
| `eval_context_big.py` | context-truncation test on the held-out test set |
| `results_context_big.csv`, `results_context_big_mse.csv` | per-utterance results at every context length |
| `numbers_big.txt` | all statistics reported in the paper (Section V-G) |
| `primary_floor_test_big.json` | the single pre-specified floor test |
| `history_big.json` | full training curve (loss + validation per epoch) |

## Run

Training (resumable; ~130s/epoch on one CPU core, run in chunks):

    export MSNSD_DIR=/path/to/MS-SNSD
    python3 train_nn_big.py --epochs 5 --val_every 1   # repeat/resume;
                                                          # writes ckpt_big.pt
                                                          # (not shipped --
                                                          # regenerate by
                                                          # training, or use
                                                          # the shipped
                                                          # best_big.pt directly)

Context-truncation evaluation (from the shipped `best_big.pt`, no
retraining needed; run in three C-value batches to keep each call short):

    python3 eval_context_big.py --batch 1
    python3 eval_context_big.py --batch 2,4,8
    python3 eval_context_big.py --batch 16,32,64
    python3 eval_context_big.py --batch 128,None
    # merge the four results_context_big_*.csv / *_mse_*.csv into
    # results_context_big.csv / results_context_big_mse.csv, then:
    python3 eval_context_big.py --batch analyze

## Headline finding

Reported honestly, including the part that did not improve:

- **Held-out performance**: -4.09 dB on the test set (validation, used
  for checkpoint selection: -3.46 dB), *worse* than the smaller
  NOIZEUS-trained network (-2.06 dB) despite 6x more training data --
  attributed to MS-SNSD's noise categories being harder (e.g. overlapping
  speech in `AirportAnnouncements`), not to anything about context length.
- **Variance term**: replicates cleanly. Paired Wilcoxon signed-rank test
  (n=20 independent speech clips), p<1e-4 at every step from C=1 to 128,
  saturating at the same point as the first network; the final step to
  unbounded context is not significant (p=0.62), consistent with
  saturation rather than noise.
- **Non-stationarity floor**: replicates in direction but is weaker
  statistically. The same pre-specified plateau test (gap-to-Bc-vs-NSI
  slope averaged over C in {64,128,unbounded}) gives slope +0.09,
  p=0.055 (cluster-robust, df=19) -- about a third the size of, and just
  short of significant compared to, the first network's +0.25, p=0.012.

Whether the smaller floor reflects a genuinely smaller effect for a
better-converged large-C solution, or reduced power on this harder
dataset, is not resolved by this experiment and is reported as such in
the paper (Section V-G).
