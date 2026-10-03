"""Global Delta-SNR (MSE domain) of Bc on MS-SNSD, matching the SAME
mixtures (same seeds) as msnsd_validate.py, to test whether the
MSE-domain Bc-vs-NSI direction found on NOIZEUS (positive) replicates
here.
"""
import os, sys, numpy as np, pandas as pd, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msnsd_validate as m

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=m.N_SPEECH)
    ap.add_argument('--out', type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_msnsd_mse.csv'))
    args = ap.parse_args()
    pools = m.load_noise_pools()
    speech_files = m.pick_speech(m.N_SPEECH, seed=0)[args.start:args.end]
    rows = []
    for si, sf_path in enumerate(speech_files):
        s, fs = sf.read(sf_path)
        sp_id = os.path.basename(sf_path).replace('.wav', '')
        for cat in m.CATS:
            x, v = m.mix_at_snr(s, pools[cat], m.SNR_DB, seed=m.det_seed(sp_id, cat))
            shat = m.bc_signal(s, x, 128)
            n = min(len(s), len(shat))
            dsnr = 10*np.log10(np.mean(v**2) / np.mean((s[:n]-shat[:n])**2))
            rows.append(dict(speaker=sp_id, noise_type=cat, nsi=m.nsi_std(v), dsnr=dsnr))
        print(f'{args.start+si+1}/{m.N_SPEECH} done', flush=True)
    df = pd.DataFrame(rows)
    if os.path.exists(args.out) and args.start > 0:
        prev = pd.read_csv(args.out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print('wrote', args.out, len(df), 'rows')
