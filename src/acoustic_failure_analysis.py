
from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.stats import spearmanr

ROOT = Path.cwd()
OUT = ROOT / 'results' / 'lora_generic_hindi'
BASE = OUT / 'whisper_small_test_predictions.csv'
LORA = OUT / 'whisper_small_lora_test_predictions.csv'
PARQUET_DIR = ROOT / 'data' / 'raw' / 'lahaja' / 'data'
N_BOOT = 2000
SEED = 42


def norm(s):
    s = str(s).lower().strip()
    return ' '.join(
        ''.join(c if c.isalnum() or c.isspace() else ' ' for c in s).split()
    )


def ed(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(r, h):
    a, b = norm(r).split(), norm(h).split()
    return ed(a, b) / max(1, len(a))



def decode(cell):
    raw = cell.get('bytes') if isinstance(cell, dict) else None

    if raw is None:
        raise ValueError(
            f'No embedded audio bytes; cell type={type(cell).__name__}'
        )

    x, sr = sf.read(
        BytesIO(raw),
        dtype='float32',
        always_2d=True
    )
    return x.mean(axis=1), int(sr)


def features(x, sr):
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(x)))
    frame = max(1, round(0.020 * sr))
    hop = max(1, round(0.010 * sr))

    if len(x) < frame:
        frames = x.reshape(1, -1)
    else:
        frames = np.stack([
            x[s:s + frame] for s in range(0, len(x) - frame + 1, hop)
        ])

    frms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    threshold = max(0.005, peak * 0.02)

    return {
        'audio_duration_seconds': dur,
        'sample_rate_hz': sr,
        'rms_amplitude': rms,
        'rms_dbfs': float(20 * np.log10(max(rms, 1e-12))),
        'peak_amplitude': peak,
        'clipping_fraction': float(np.mean(np.abs(x) >= 0.999)),
        'low_energy_fraction': float(np.mean(frms < threshold))
    }


def cluster_ci(df, xcol, ycol, seed):
    speakers = df.speaker_id.dropna().unique()
    if len(speakers) < 5:
        return {
            'rho': None, 'ci95_low': None, 'ci95_high': None,
            'speakers': len(speakers)
        }

    rho = spearmanr(df[xcol], df[ycol], nan_policy='omit').statistic
    rng = np.random.default_rng(seed)
    groups = {s: df[df.speaker_id == s] for s in speakers}
    vals = []

    for _ in range(N_BOOT):
        sample = pd.concat(
            [groups[s] for s in rng.choice(
                speakers, len(speakers), replace=True
            )],
            ignore_index=True
        )
        if sample[xcol].nunique() < 2 or sample[ycol].nunique() < 2:
            continue
        z = spearmanr(sample[xcol], sample[ycol], nan_policy='omit').statistic
        if np.isfinite(z):
            vals.append(float(z))

    if not vals or not np.isfinite(rho):
        return {
            'rho': None, 'ci95_low': None, 'ci95_high': None,
            'speakers': len(speakers)
        }

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        'rho': float(rho),
        'ci95_low': float(lo),
        'ci95_high': float(hi),
        'speakers': int(len(speakers)),
        'bootstrap_replicates_used': len(vals)
    }


def main():
    for p in (BASE, LORA):
        if not p.exists():
            raise FileNotFoundError(f'Missing predictions: {p}')

    if not PARQUET_DIR.exists():
        raise FileNotFoundError(f'Missing LAHAJA parquet directory: {PARQUET_DIR}')

    b = pd.read_csv(BASE).rename(columns={
        'prediction': 'baseline_prediction',
        'status': 'baseline_status'
    })
    l = pd.read_csv(LORA).rename(columns={
        'prediction': 'lora_prediction',
        'status': 'lora_status'
    })

    m = b.merge(
        l, on=['source_shard', 'source_row'],
        how='inner', suffixes=('_base', '_lora'),
        validate='one_to_one'
    )

    if len(m) != len(b) or len(m) != len(l):
        raise ValueError(
            f'Prediction alignment mismatch: {len(b)}, {len(l)}, matched {len(m)}'
        )

    if not (
        m.reference_base.fillna('').astype(str).values ==
        m.reference_lora.fillna('').astype(str).values
    ).all():
        raise ValueError('Reference mismatch')

    if not (
        m.sp_id_base.astype(str).values ==
        m.sp_id_lora.astype(str).values
    ).all():
        raise ValueError('Speaker ID mismatch')

    m = m[
        m.baseline_status.astype(str).str.lower().eq('success') &
        m.lora_status.astype(str).str.lower().eq('success')
    ].reset_index(drop=True)

    print(f'Matched successful recordings: {len(m)}; speakers: {m.sp_id_base.nunique()}')

    cache = {}
    rows = []
    failures = []

    for i, r in m.iterrows():
        shard = str(r.source_shard)

        try:
            if shard not in cache:
                p = PARQUET_DIR / Path(shard).name
                if not p.exists():
                    raise FileNotFoundError(f'Parquet shard not found: {p}')
                cache[shard] = pq.read_table(
                    p, columns=['audio_filepath']
                ).column('audio_filepath').to_pylist()

            x, sr = decode(cache[shard][int(r.source_row)])
            f = features(x, sr)

            ref = str(r.reference_base) if pd.notna(r.reference_base) else ''
            bp = str(r.baseline_prediction) if pd.notna(r.baseline_prediction) else ''
            lp = str(r.lora_prediction) if pd.notna(r.lora_prediction) else ''

            nw = len(norm(ref).split())
            bw = wer(ref, bp)
            lw = wer(ref, lp)

            f.update({
                'source_shard': shard,
                'source_row': int(r.source_row),
                'recording_key': f'{shard}::{int(r.source_row)}',
                'speaker_id': str(r.sp_id_base),
                'state': str(r.get('state_base', '')),
                'district': str(r.get('district_base', '')),
                'reference_words': nw,
                'speech_density_words_per_second':
                    nw / max(f['audio_duration_seconds'], 1e-9),
                'baseline_wer': bw,
                'lora_wer': lw,
                'wer_improvement_baseline_minus_lora': bw - lw
            })
            rows.append(f)

        except Exception as e:
            failures.append({
                'source_shard': shard,
                'source_row': str(r.source_row),
                'error': str(e)
            })

        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(m)}; usable {len(rows)}')

    # FIX: save and display errors even when every recording fails.
    if not rows:
        error_path = OUT / 'acoustic_failure_decode_errors.csv'
        pd.DataFrame(failures).to_csv(
            error_path, index=False, encoding='utf-8-sig'
        )
        print('\nNo acoustic features extracted.')
        print('First 10 errors:')
        print(pd.DataFrame(failures).head(10).to_string(index=False))
        raise RuntimeError(
            f'Audio extraction failed; errors saved to {error_path}'
        )

    d = pd.DataFrame(rows)
    d['duration_band'] = pd.cut(
        d.audio_duration_seconds,
        [0, 3, 6, 10, 20, np.inf],
        labels=['0-3s', '3-6s', '6-10s', '10-20s', '>20s'],
        right=False
    )

    qlabels = ['Q1-low', 'Q2', 'Q3', 'Q4-high']
    for col, qcol in [
        ('low_energy_fraction', 'low_energy_quartile'),
        ('rms_dbfs', 'rms_quartile'),
        ('speech_density_words_per_second', 'speech_density_quartile')
    ]:
        d[qcol] = pd.qcut(
            d[col].rank(method='first'), 4, labels=qlabels
        )

    d.to_csv(
        OUT / 'acoustic_failure_analysis.csv',
        index=False, encoding='utf-8-sig'
    )

    group_rows = []
    for feature in [
        'duration_band', 'low_energy_quartile',
        'rms_quartile', 'speech_density_quartile'
    ]:
        for group, sub in d.groupby(feature, observed=True):
            group_rows.append({
                'grouping_feature': feature,
                'group': str(group),
                'recordings': len(sub),
                'speakers': sub.speaker_id.nunique(),
                'mean_baseline_wer': sub.baseline_wer.mean(),
                'median_baseline_wer': sub.baseline_wer.median(),
                'mean_lora_wer': sub.lora_wer.mean(),
                'median_lora_wer': sub.lora_wer.median(),
                'mean_wer_improvement':
                    sub.wer_improvement_baseline_minus_lora.mean(),
                'mean_low_energy_fraction': sub.low_energy_fraction.mean(),
                'mean_rms_dbfs': sub.rms_dbfs.mean(),
                'mean_duration_seconds': sub.audio_duration_seconds.mean()
            })

    pd.DataFrame(group_rows).to_csv(
        OUT / 'acoustic_failure_group_summary.csv',
        index=False, encoding='utf-8-sig'
    )

    summary = {
        'usable_recordings': len(d),
        'failed_audio_decodes': len(failures),
        'speakers': int(d.speaker_id.nunique()),
        'interpretation_note':
            'Low-energy fraction is an amplitude-threshold proxy, not verified '
            'voice activity detection. RMS depends on recording gain. '
            'Results are associations, not causal claims.',
        'descriptives': {
            'mean_baseline_wer': float(d.baseline_wer.mean()),
            'mean_lora_wer': float(d.lora_wer.mean()),
            'mean_wer_improvement':
                float(d.wer_improvement_baseline_minus_lora.mean()),
            'median_duration_seconds':
                float(d.audio_duration_seconds.median()),
            'median_low_energy_fraction':
                float(d.low_energy_fraction.median()),
            'median_rms_dbfs': float(d.rms_dbfs.median()),
            'median_speech_density_words_per_second':
                float(d.speech_density_words_per_second.median())
        },
        'speaker_cluster_bootstrap_spearman': {}
    }

    for feature in [
        'audio_duration_seconds', 'low_energy_fraction',
        'rms_dbfs', 'speech_density_words_per_second'
    ]:
        summary['speaker_cluster_bootstrap_spearman'][feature] = {}
        for target in [
            'baseline_wer', 'lora_wer',
            'wer_improvement_baseline_minus_lora'
        ]:
            summary['speaker_cluster_bootstrap_spearman'][feature][target] = (
                cluster_ci(d, feature, target, SEED)
            )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUT / 'acoustic_failure_decode_errors.csv',
            index=False, encoding='utf-8-sig'
        )

    (OUT / 'acoustic_failure_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print('\nACOUSTIC ANALYSIS COMPLETE')
    print(json.dumps(summary['descriptives'], indent=2))
    print(
        'Outputs: acoustic_failure_analysis.csv, '
        'acoustic_failure_group_summary.csv, acoustic_failure_summary.json'
    )
    if failures:
        print(f'Audio decode/lookup failures: {len(failures)}; see acoustic_failure_decode_errors.csv')


if __name__ == '__main__':
    main()
