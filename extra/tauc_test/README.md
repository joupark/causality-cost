# Coherence time of real noise and the floor (Section IV-B, V-G, Fig. 7)

tauc.py measures the coherence time tau_c of Definition 1 per utterance
(32 ms frames, 16 ms hop, 16 bands; relative l1 spectral change D(T) minus
the same quantity for three phase-randomised stationary surrogates; first lag
reaching eps0 = 0.1, at most 1 s; eps0 = 0.05 and 0.2 as sensitivity).
Predictions Q1-Q4 are in its docstring and were fixed before the analysis.
The denominator was symmetrised after a first run divided by zero in digitally
silent MS-SNSD segments, before any outcome analysis.

    export NOIZEUS_DIR=... MSNSD_DIR=...
    python3 tauc.py noizeus; python3 tauc.py msnsd
    python3 tauc_analysis.py      # numbers_tauc.txt
    python3 tauc_fig.py           # ../figures/fig_tauc.pdf

Result (eps0 = 0.1): reference-noise estimator on NOIZEUS -- optimal context
grows as tau_c^0.66 across noise types (r = 0.99; predicted 2/3) and tau_c^0.53
within types (p = 4.3e-6); floor height falls as tau_c^-0.29 across types
(r = -0.97) and tau_c^-0.43 within types (p = 3.6e-7), slower than the predicted
-2/3; tau_c explains more within-type variance than the NSI. Trained networks:
plateau excess MSE unrelated to tau_c (p = 0.51, 0.31).
