"""Gap to B_c in the MSE domain (global DeltaSNR) versus NSI and tau_c.

The paper's gap analysis (Table III) uses DeltaSegSNR, and the controlled
experiment (Section V-B) shows that SegSNR itself falls with NSI when MSE is
fixed.  This script repeats the gap analysis in global DeltaSNR (a monotone
function of MSE) for NLMS, MMSE-Wiener, OMLSA and DeepFilterNet3 on NOIZEUS,
15 dB, with the same calls as measure.py / dfn3_eval.py.
PREDICTION (before running): the gap grows with NSI in the MSE domain for all
four methods; if not, the SegSNR-domain result must be reported as possibly
metric-driven.
"""
import os, sys, numpy as np, pandas as pd, torch
from scipy.signal import resample_poly
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'baselines'))
import measure as m
from nlms_baseline import nlms_line_enhancer
from mmse_wiener_baseline import mmse_wiener
from omlsa_baseline import omlsa
FS = 8000
def gsnr(s, v, y):
    n = min(len(s), len(y)); return 10 * np.log10(np.mean(v[:n] ** 2) / np.mean((s[:n] - y[:n]) ** 2))
use_dfn = len(sys.argv) > 1 and sys.argv[1] == 'dfn3'
if use_dfn:
    from df.enhance import init_df, enhance
    model, st, _ = init_df(model_base_dir='DeepFilterNet3', log_level='ERROR')
rows = []
for nt in m.TYPES:
    for k in range(1, 31):
        s, x, v = m.load(15, f'sp{k:02d}', nt); r = dict(noise_type=nt, speaker=f'sp{k:02d}')
        if use_dfn:
            x48 = resample_poly(x, 6, 1).astype(np.float32)
            y48 = enhance(model, st, torch.from_numpy(x48).unsqueeze(0)).squeeze(0).numpy()
            r['dfn3'] = gsnr(s, v, resample_poly(y48, 1, 6))
        else:
            r['mmse'] = gsnr(s, v, mmse_wiener(x, v, n_fft=256, fs=FS))
            r['omlsa'] = gsnr(s, v, omlsa(x, v, n_fft=256, fs=FS))
            r['nlms'] = gsnr(s, v, nlms_line_enhancer(x, L=128, mu=0.05, eps=1e-3, delay=1)[0])
        rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(HERE, 'gap_mse_dfn3.csv' if use_dfn else 'gap_mse_classic.csv'), index=False)
print('wrote', len(rows))
