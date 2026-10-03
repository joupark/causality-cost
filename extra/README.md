# extra/ — material from a longer, unsubmitted study

Not part of the SPL letter. Kept for the record. These scripts use the section
and proposition numbering of that longer draft (e.g. "Theorem 1", "Proposition 3"),
which does not match the letter.

| Folder | Contents |
|---|---|
| `nn_experiment/` | Small causal Transformer (83k params) trained on NOIZEUS; context-truncation test |
| `nn_big/` | Larger causal Transformer (274k params) trained on MS-SNSD |
| `floor_test/` | Floor of the minimax bound tested in the theorem's own model |
| `tauc_test/` | Coherence time of real noise; optimal context and floor height vs. tau_c |
| `data_scaling/` | Training-set-size study (16-150 MS-SNSD clips) |
| `fir_and_conditionS/` | FIR-order sensitivity; the condition-(S) script also lives in `src/condition_S/` |
