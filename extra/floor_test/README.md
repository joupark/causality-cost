# Floor test in the model of Theorem 1 (Section V-G, Fig. 8)

Observations x(k) ~ N(0, theta(k)), theta in [1,4], target w* = Ss/(Ss+theta),
Ss = 4, so (T1)-(T3) hold with kappa_I = 0.5, g = 0.0625. Test episodes follow
the proof's two-point construction (constant vs. ramp of slope +-beta over the
last m_beta samples). Predictions are in the docstring of floor_test.py.

    python3 floor_test.py 8; python3 floor_test.py 16
    python3 floor_test.py 32; python3 floor_test.py 64   # ~2-5 min each (CPU)
    python3 floor_numbers.py                             # numbers_floor.txt, ../figures/fig_floor.pdf

m_beta = 128 was not run (the 8*m_beta-context MLP exceeds the per-run budget).
Results: every estimator (best window, best exponential smoothing, MLP with
context m_beta or 8 m_beta trained on fresh data each epoch) stays above the
exact bound (7.6-11.7x for the MLP); window/exponential risk scales as
beta^0.66 and stays 8.6-8.9x the bound; the optimal-window prediction was not
observed in this ensemble (see paper). Because the ensemble is the proof's own
construction, the test checks tightness, not an independent physical claim.
