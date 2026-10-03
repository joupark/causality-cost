# MS-SNSD second-corpus validation (Section V-E, Fig. 4)

## Getting the data (without a full ~2.5GB clone)

MS-SNSD (github.com/microsoft/MS-SNSD, MIT license) is a large repository
because it stores training audio directly in git. Only the `clean_test`
(403MB) and `noise_test` (73MB) folders are needed here; fetch just those
with a partial clone:

    git clone --filter=blob:none --no-checkout --depth 1 \
        https://github.com/microsoft/MS-SNSD.git
    cd MS-SNSD
    git sparse-checkout init --cone
    git sparse-checkout set clean_test noise_test
    git checkout

Point `MSNSD_DIR` at the resulting folder (it should contain `clean_test/`
and `noise_test/`), or place/symlink it as `MS-SNSD/` inside this
directory (the default `run_msnsd.sh` looks there).

## Run

    export MSNSD_DIR=/path/to/MS-SNSD   # optional; defaults to ./MS-SNSD
    ./run_msnsd.sh

Takes about 10 minutes on one CPU core (360 mixtures x four measurements).

## What it does

- `msnsd_validate.py`: mixes 40 clean speech clips (fixed seed) with each
  of 9 real noise categories at SNR=15 dB (n=360), computes NSI, $B_c$
  (FIR Wiener-Hopf, L=128, same as NOIZEUS), the causality cost $\Delta B$
  (frequency-domain idealisation of Proposition 2), and the
  NLMS/MMSE-Wiener/OMLSA gap to $B_c$.
- `msnsd_mse.py`: the same mixtures, additionally scored in the MSE domain
  (global $\Delta$SNR) to test whether $B_c$'s SegSNR-vs-MSE-domain
  divergence found on NOIZEUS (Section V-B) replicates.
- `msnsd_numbers.py`: cluster-robust (cluster = speech clip) statistics
  for all of the above, with Holm-Bonferroni correction across the three
  enhancement methods -- the numbers quoted in Section V-E -- plus Fig. 4.
- `cluster_stats.py`: copy of the root-level module (see main README),
  self-contained so this folder can be run independently.

The speech content is reused across all 9 noise categories (as NOIZEUS
reuses 30 sentences across 7 noise types), so cluster-robust inference
with cluster = speech clip applies identically to both corpora.
