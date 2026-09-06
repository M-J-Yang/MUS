#!/usr/bin/env python3
"""Inspect existing metadata and results; no model inference or experiments."""
import csv
import hashlib
import json
import statistics
import sys
from functools import lru_cache
from pathlib import Path
import soundfile as sf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'src'))
from usde.text import normalize_text as _normalize_text
normalize_text = lru_cache(maxsize=None)(_normalize_text)

@lru_cache(maxsize=None)
def audio_info(path):
    info = sf.info(path)
    return info.frames, info.samplerate

def read(path):
    return json.loads((ROOT / path).read_text())

def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]

def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

audit = {'operation': 'read-only inspection of saved artifacts; no new evaluation',
         'date': '2026-09-06', 'folds': {}, 'lineage': {}, 'results': {}, 'sha256': {}}
keys = ['model_type', 'hidden_size', 'num_hidden_layers', 'conv_bias',
        'feat_extract_norm', 'do_stable_layer_norm']
base_w = 'checkpoints/wav2vec2_large_960h_pretrained'
base_d = 'checkpoints/data2vec_audio_large_960h'
pairs = {
 'w2v2_f1': (base_w, 'artifacts/runs/l2_arctic_official_ut8/fold1/w2v2_large_960h_supcon_local_replica_full_gc'),
 'w2v2_f2': (base_w, 'artifacts/runs/l2_arctic_official_ut8/fold2/w2v2_large_960h_supcon_local_replica_full_gc'),
 'data2vec_f0': (base_d, 'artifacts/runs/l2_arctic_official_ut8/fold0/data2vec_audio_large_960h_ctc_formal_b4'),
 'excluded_public_w2v2_f0': (base_w, 'artifacts/oracles/wav2vec2-large-l2-arctic-supcon-repeated-8fold-0'),
}
for name, (base, ft) in pairs.items():
    a, b = read(base + '/config.json'), read(ft + '/config.json')
    audit['lineage'][name] = {
        'base': base, 'adapted': ft,
        'base_config': {k: a.get(k) for k in keys},
        'adapted_config': {k: b.get(k) for k in keys},
        'architecture_differences': {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)},
    }
    for path in [base + '/config.json', ft + '/config.json']:
        audit['sha256'][path] = digest(path)
    if name.startswith('excluded'):
        assert audit['lineage'][name]['architecture_differences']
    else:
        assert not audit['lineage'][name]['architecture_differences']
        summary = read(ft + '/training_summary.json')
        assert summary['pretrained_path'] == base
        audit['lineage'][name]['recorded_initialization_matches'] = True

for fold in range(3):
    prefix = f'manifests/l2_arctic_official_ut8/fold{fold}'
    data = {s: rows(f'{prefix}/{s}.jsonl') for s in ['train', 'dev', 'test', 'train_utility', 'train_teacher']}
    stats = {}
    for split, data_rows in data.items():
        headers = [audio_info(r['audio_path']) for r in data_rows]
        lengths = [frames / rate for frames, rate in headers]
        stats[split] = {
            'utterances': len(data_rows), 'hours_from_audio_headers': sum(lengths) / 3600,
            'stale_manifest_frame_counts': sum(int(r['frames']) != h[0] for r,h in zip(data_rows,headers)),
            'actual_sample_rates': sorted({h[1] for h in headers}),
            'speakers': sorted({r['speaker_id'] for r in data_rows}),
            'l1_groups': sorted({r['l1'] for r in data_rows}),
            'unique_prompts': len({r['prompt_id'] for r in data_rows}),
            'unique_normalized_transcripts': len({normalize_text(r['transcript']) for r in data_rows}),
            'reference_words': sum(len(normalize_text(r['transcript']).split()) for r in data_rows),
            'audio_longer_than_10s': sum(x > 10 for x in lengths),
            'normalized_text_longer_than_128_characters': sum(len(normalize_text(r['transcript'])) > 128 for r in data_rows),
        }
        audit['sha256'][f'{prefix}/{split}.jsonl'] = digest(f'{prefix}/{split}.jsonl')
    overlaps = {}
    for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
        overlaps[f'{a}_{b}'] = {
            field: len({r[field] for r in data[a]} & {r[field] for r in data[b]})
            for field in ['utt_id', 'speaker_id', 'prompt_id']
        }
        overlaps[f'{a}_{b}']['normalized_transcripts'] = len(
            {normalize_text(r['transcript']) for r in data[a]} &
            {normalize_text(r['transcript']) for r in data[b]})
        assert overlaps[f'{a}_{b}']['utt_id'] == 0
    csv_checks = {}
    for s, upstream in [('train', 'train'), ('dev', 'val'), ('test', 'test')]:
        path = f'artifacts/protocol_audit/official_l2_arctic_8fold_{fold}/{upstream}.csv'
        with (ROOT / path).open() as f:
            original = list(csv.DictReader(f))
        observed = {(r['speaker_id'], Path(r['audio_path']).name, ' '.join(r['transcript'].lower().split())) for r in data[s]}
        expected = {(r['speaker'], Path(r['audio_filename']).name, ' '.join(r['transcript'].lower().split())) for r in original}
        assert observed == expected and len(original) == len(data[s])
        csv_checks[s] = True
        audit['sha256'][path] = digest(path)
    assert {r['utt_id'] for r in data['train_utility']} <= {r['utt_id'] for r in data['train']}
    audit['folds'][str(fold)] = {'splits': stats, 'overlap': overlaps, 'csv_projection_matches': csv_checks,
                                'calibration_is_part_of_encoder_training': True}

metric_paths = {
 'w2v2_f1': 'artifacts/results/l2_arctic_official_ut8/fold1/w2v2_large_960h_oracle_shift_local_replica_core/core_metrics.json',
 'w2v2_f2': 'artifacts/results/l2_arctic_official_ut8/fold2/w2v2_large_960h_oracle_shift_local_replica_core/core_metrics.json',
 'data2vec_f0': 'artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_core/core_metrics.json',
}
for name, path in metric_paths.items():
    d = read(path)
    assert d['checkpoint'] == pairs[name][1]
    audit['results'][name] = {'source': path, 'splits': d['splits']}
    for split in ['dev', 'test']:
        x = d['splits'][split]
        assert x['conditions']['utility75']['wer'] == x['conditions']['drop_worst25']['wer']
        assert x['identity']['pass']
        extraction = read(f"{d['cache_root']}/{split}/extraction_report.json")
        assert extraction['pretrained_model'] == pairs[name][0]
        assert extraction['fine_tuned_model'] == pairs[name][1]
        assert extraction['layer'] == -1 and extraction['hidden_dim'] == 1024
    utility_path = str(Path(d['utility_ranking']).parent / 'utility_shift_taylor_stats.json')
    if not (ROOT / utility_path).exists():
        utility_path = 'artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_utility/utility_shift_taylor_stats.json'
    u = read(utility_path)
    audit['results'][name]['utility_metadata'] = {k:v for k,v in u.items() if not isinstance(v, (list, dict))}
    audit['sha256'][path] = digest(path)
    audit['sha256'][utility_path] = digest(utility_path)

package = 'artifacts/results/l2_arctic_official_ut8/fold0/data2vec_large_960h_shift_empirical_package/metrics.json'
p = read(package)
for split in ['dev', 'test']:
    for method in ['Random', 'Random+Rescale']:
        for fraction in ['25', '50', '75']:
            r = p['splits'][split]['retention']['methods'][method][fraction]
            w = [z['wer'] for z in r['seed_results']]
            assert abs(statistics.mean(w) - r['wer']) < 1e-12
            assert abs(statistics.stdev(w) - r['std_wer']) < 1e-12
audit['data2vec_retention'] = {s:p['splits'][s]['retention'] for s in ['dev', 'test']}
audit['sha256'][package] = digest(package)
calibration_path = 'artifacts/results/l2_arctic_official_ut8/fold0/data2vec_calibration_size/calibration_size_metrics.json'
c = read(calibration_path)
audit['data2vec_calibration_size'] = {}
for size in c['calibration_sizes']:
    group = [r for r in c['results'] if r['calibration_size_used'] == size]
    audit['data2vec_calibration_size'][str(size)] = {
        k: {'mean': statistics.mean(r[k] for r in group), 'sample_sd': statistics.stdev(r[k] for r in group)}
        for k in ['utility50_wer', 'utility75_wer', 'drop_best25_wer', 'top25_overlap_with_full']}
audit['sha256'][calibration_path] = digest(calibration_path)
(OUT / 'artifact_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print('Saved artifact_audit.json. All three retained checkpoint pairs and CSV projections verified.')
print('Excluded public Fold 0 pair: architecture mismatch confirmed.')
for f, d in audit['folds'].items():
    print('Fold', f, 'overlaps:', d['overlap'])
    print('Hours (actual audio headers):', {s: round(r['hours_from_audio_headers'], 3) for s,r in d['splits'].items()})
