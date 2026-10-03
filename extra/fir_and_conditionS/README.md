# FIR order sensitivity and direct test of condition (S)

Two follow-up analyses, both referenced in Section V-D/F and the
Limitations paragraph of the paper.

## `fir_sweep_extended.py` + `fir_pooled.py` (Section V-H, FIR order)

`fir_sweep_extended.py` extends the original `fir_sweep.py` (root of the
repo) to L=4096 using `scipy.linalg.solve_toeplitz` (an O(L^2)
Levinson-type solver) instead of a dense solve, which does not scale
past L~1024. It shows B_c continuing to improve sharply out to L=4096
with no sign of saturating when fit **per utterance**.

`fir_pooled.py` tests whether this is a finite-sample artefact: it
refits the same L values from autocorrelations **pooled across all 30
sentences of each noise type** (~84s of audio instead of ~2.8s) and
finds the L-dependence almost disappears (+0.036 dB from L=128 to 4096,
averaged across noise types, versus +1.15 dB for the per-utterance fit).
This confirms L=128 is a well-converged choice once enough data is used
to estimate the correlations.

Run: `python3 fir_sweep_extended.py && python3 fir_pooled.py`
(~1 minute total; NOIZEUS only, no extra data needed beyond the root
`measure.py`/NOIZEUS setup).

## `condition_S_test.py` (Section V-E, direct test of Prop. 3)

Proposition 3 says $B_c$ retreats iff a sign condition (S) holds:
non-stationarity must move averaged noise power into the passband of
the causal Wiener filter, $\sum_f |W^*(f)|^2 \partial\bar S_v(f)/\partial\nu > 0$.
Elsewhere in the paper this is only inferred indirectly, from the sign
of the observed $B_c$-vs-NSI slope in the MSE domain. This script
measures the sum directly, per noise type, on both NOIZEUS and MS-SNSD:
it fits each noise type's pooled Wiener filter to get $|W^*(f)|^2$,
regresses each instance's noise periodogram against its NSI to estimate
$\partial \bar S_v(f)/\partial\nu$, and sums the product over frequency.

Finding: the sign is mixed across individual noise types in both
corpora (not distinguishable from chance at these small per-corpus
counts), but the **median** across types is slightly negative on
NOIZEUS and positive on MS-SNSD -- the same direction as the
corpus-level MSE finding, though with too few noise types to be
conclusive on its own. Reported as a first direct measurement, not a
resolution, of condition (S).

Run: `MSNSD_DIR=/path/to/MS-SNSD python3 condition_S_test.py`
(needs both NOIZEUS, already required elsewhere, and MS-SNSD's
`noise_test`/`clean_test` folders -- see `../msnsd_validate/README.md`
for how to fetch them without a full clone).
