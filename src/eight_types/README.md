# Eight-type robustness check (letter, Sec. III-A footnote)

The letter's 15 dB analyses use seven NOIZEUS noise types: the distribution has
no 15 dB archive for suburban-train noise (train_0dB/5dB/10dB.zip exist,
train_15dB.zip does not), so train noise cannot be included at that SNR.

    export NOIZEUS_DIR=/path/to/noizeus
    python3 eight_types.py    # numbers_eight_types.txt (delete the csv to re-measure)

At 0/5/10 dB, including train noise leaves the conclusions unchanged, e.g. the
within-type slope of T versus NSI at 0 dB is +0.300 (7 types) vs +0.296 (8 types),
and the type-mean slope +0.432 (r=0.95) vs +0.394 (r=0.85).
