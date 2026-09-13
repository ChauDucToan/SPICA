"""CPU-only thesis export from immutable saved results; no training or network.
Run from repo root: CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/thesis/build_assets.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = {
    'sketchy_104_21': ('percent20_official_execution_20260912T133000Z', 4446),
    'tuberlin_220_30': ('percent20_remaining_execution_20260912', 1189),
    'quickdraw_80_30': ('percent20_remaining_execution_20260912', 18229),
}
LABELS = ['Sketchy104/21', 'TU220/30', 'QuickDraw80/30']
METRICS = {'full_mAP': 'average_precision', 'P@200': 'P200',
           'mAP@200_min_relevant_k': 'AP200_min_relevant_k',
           'mAP@200_all_relevant': 'AP200_all_relevant',
           'mAP@200_prefix_positive': 'AP200_prefix_positive'}

def read(p):
    return json.loads(p.read_text())

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def csv_write(name, rows):
    with (OUT / 'tables' / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    for name in ['tables', 'figures']:
        (OUT / name).mkdir(exist_ok=True)
    final, conditions, curves, protocols, provenance, audits, configs = [], [], [], [], [], [], {}
    for dataset, (campaign, horizon) in RUNS.items():
        root = ROOT / 'outputs' / campaign; run = root / 'runs' / dataset
        cfg = read(run / 'resolved_config.json'); configs[dataset] = cfg
        checkpoint = read(run / 'checkpoint_latest.json')
        summary_path = run / 'test' / f'step_{horizon}' / 'summary.json'
        summary = read(summary_path)
        assert cfg['total_steps'] == horizon == checkpoint['step'] == summary['step']
        assert sha(run / 'checkpoint_latest.pt') == checkpoint['sha256'] == summary['checkpoint_selection']['sha256']
        assert cfg['source_snapshot_hash'] == summary['source_snapshot_hash']
        assert summary['model_state_before'] == summary['model_state_after'] == checkpoint['model_state_hash']
        assert summary['predictor_forwards'] == 0 and len(summary['conditions']) == 10
        manifest = read(root / 'execution_manifest.json')
        # Verify archived source bytes, not current post-training source.
        archive = run / 'source_snapshot'; index = read(archive / 'index.json'); aggregate = hashlib.sha256()
        for item in index['manifest']:
            data = (archive / 'files' / item['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            aggregate.update(item['path'].encode() + b'\0' + data + b'\0')
        assert aggregate.hexdigest() == index['sha256'] == cfg['source_snapshot_hash']
        rows = [json.loads(line) for line in (run / 'test_metrics.jsonl').read_text().splitlines()]
        assert [r['step_train'] for r in rows] == [(horizon * i + 4) // 5 for i in range(1, 6)]
        for row in rows:
            curves.append({'dataset': dataset, 'step': row['step_train'], 'progress_percent': row['step_train'] * 100 / horizon,
                           **{k: v for k, v in row.items() if k != 'step_train'}})
        for scope, metric in [('clean', summary['clean']), ('masked_macro', summary['masked_macro'])]:
            for key, value in metric.items():
                alias = {'full_mAP': 'mAP@all', 'P@200': 'P@200', 'mAP@200_min_relevant_k': 'mAP@200'}.get(key)
                if alias:
                    prefix = 'cleaned' if scope == 'clean' else 'masked'
                    assert rows[-1][f'test/{prefix}/{alias}'] == value
            final.append({'dataset': dataset, 'step': horizon, 'condition': scope, **metric})
        for fraction, metric in summary['masked_by_fraction'].items():
            final.append({'dataset': dataset, 'step': horizon, 'condition': f'mask_{fraction}', **metric})
        max_delta = 0.0
        for condition in summary['conditions']:
            cp = summary_path.parent / condition['summary_artifact']; cs = read(cp)
            assert sha(cp) == condition['summary_artifact_sha256']
            npz = summary_path.parent / condition['per_query']
            assert sha(npz) == cs['per_query_sha256']
            with np.load(npz, allow_pickle=False) as saved:
                for metric, array in METRICS.items():
                    assert saved[array].shape == (summary['query_count'],)
                    delta = abs(float(saved[array].astype(np.float64).mean()) - condition['metrics'][metric])
                    assert delta <= 2e-6, (dataset, condition['condition'], metric, delta)
                    max_delta = max(max_delta, delta)
            conditions.append({'dataset': dataset, 'condition': condition['condition'], **condition['metrics'],
                               'mask_status_counts': json.dumps(condition['mask_status_counts'], sort_keys=True)})
        for metric in METRICS:
            mean = np.mean([c['metrics'][metric] for c in summary['conditions'] if c['condition'] != 'clean'])
            assert abs(mean - summary['masked_macro'][metric]) < 1e-12
        identity = cfg['protocol_identity']
        protocols.append({'dataset': dataset, **{f'{split}_{k}': v for split in ['train', 'test'] for k, v in identity[split]['counts'].items()},
                          'updates': horizon, 'warmup': cfg['warmup_steps'], 'batch_size': cfg['batch_size'], 'seed': cfg['seed'], 'passes': cfg['observation_passes']})
        provenance.append({'dataset': dataset, 'run_path': str(run.relative_to(ROOT)), 'wandb_url': read(run / 'wandb_runtime.json')['run_url'],
                           'step': horizon, 'training_head': manifest['head_commit'], 'source_sha256': cfg['source_snapshot_hash'],
                           'config_sha256': sha(run / 'resolved_config.json'), 'checkpoint_sha256': checkpoint['sha256'],
                           'summary_sha256': sha(summary_path), 'raw_campaign_status': read(root / 'runtime.json')['status']})
        audits.append({'dataset': dataset, 'archive_files': len(index['manifest']), 'query_condition_rows': summary['query_count'] * 10,
                       'max_saved_mean_delta': max_delta, 'checkpoint_and_summary_and_npz_hashes_checked': True,
                       'scope': 'saved per-query metric means only; no encoder replay, no independent ranking or AP formula recomputation'})
    csv_write('final_metrics.csv', final); csv_write('conditions.csv', conditions); csv_write('progress.csv', curves)
    csv_write('datasets.csv', protocols); csv_write('provenance.csv', provenance)
    (OUT / 'audit.json').write_text(json.dumps({'status': 'SCOPED_SAVED_ARTIFACT_CHECKS_PASS', 'checks': audits,
        'builder_did_not_write_raw_status': True, 'encoder_replay_performed': False,
        'full_sort_performed': False, 'ap_formula_recomputed': False, 'online_verification': 'separate online_check.json if present'}, indent=2) + '\n')
    (OUT / 'training_configs.json').write_text(json.dumps({d: {k: c[k] for k in ['architecture', 'optimizer', 'loss_coefficient_identity', 'sampling_identity', 'runtime', 'test_steps', 'seed', 'batch_size', 'total_steps', 'warmup_steps', 'source_snapshot_hash']} for d, c in configs.items()}, indent=2) + '\n')
    # All plotted metrics are percentages; raw CSV retains 0–1 values.
    for name in ['full_mAP', 'P@200', 'mAP@200_min_relevant_k']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pos = np.arange(3); clean = [r[name] * 100 for r in final if r['condition'] == 'clean']; masked = [r[name] * 100 for r in final if r['condition'] == 'masked_macro']
        ax.bar(pos - .18, clean, .36, label='Clean'); ax.bar(pos + .18, masked, .36, label='Masked macro (9 conditions)')
        ax.set_xticks(pos, LABELS); ax.set_ylabel(name + ' (%)'); ax.set_ylim(0, 65); ax.legend(); ax.grid(axis='y', alpha=.2)
        fig.tight_layout(); stem = name.replace('@', '_at_');fig.savefig(OUT / 'figures' / f'{stem}.png', dpi=200); fig.savefig(OUT / 'figures' / f'{stem}.svg'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, dataset, label in zip(axes, RUNS, LABELS):
        rows = [r for r in curves if r['dataset'] == dataset]
        for scope in ['cleaned', 'masked']:
            ax.plot([r['progress_percent'] for r in rows], [100 * r[f'test/{scope}/mAP@all'] for r in rows], 'o-', label=scope)
        ax.set_title(label);ax.set_xlabel('Training progress (%)');ax.set_xticks([20,40,60,80,100]);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Full mAP (%)');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(OUT / 'figures' / f'progress.{ext}', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for dataset, label in zip(RUNS, LABELS):
        rows = {r['condition']: r for r in final if r['dataset'] == dataset}
        ax.plot([0, 25, 50, 75], [100 * rows[c]['full_mAP'] for c in ['clean', 'mask_0.25', 'mask_0.5', 'mask_0.75']], 'o-', label=label)
    ax.set_xlabel('Target raster ink deletion (%)'); ax.set_ylabel('Full mAP (%)')
    ax.set_xticks([0, 25, 50, 75]); ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    for ext in ['png', 'svg']: fig.savefig(OUT / 'figures' / f'severity.{ext}', dpi=200)
    plt.close(fig)
    text = ['# Kết quả cuối — official MP-Q', '', 'Giá trị trong bảng là %. CSV giữ độ chính xác số gốc 0–1.', '',
            '| Dataset | Clean full mAP | Masked full mAP | Clean P@200 | Masked P@200 | Clean AP200 minR | Masked AP200 minR |', '|---|---:|---:|---:|---:|---:|---:|']
    for dataset, label in zip(RUNS, LABELS):
        c = next(r for r in final if r['dataset'] == dataset and r['condition'] == 'clean');m = next(r for r in final if r['dataset'] == dataset and r['condition'] == 'masked_macro')
        text.append('| '+label+' | '+' | '.join(f'{r[k]*100:.4f}' for k in ['full_mAP','P@200','mAP@200_min_relevant_k'] for r in [c,m])+' |')
    text += ['', '![Full mAP](figures/full_mAP.png)', '', '![Tiến độ](figures/progress.png)', '', '![Mức xóa](figures/severity.png)', '',
             'Hình severity: mỗi điểm25/50/75% là trung bình3mask seeds ở checkpoint cuối;0% là clean. Đây là mức xóa mục tiêu, không khẳng định mọi mẫu đạt đúng mức đó.', '',
             '**Cảnh báo:** Sketchy đã đủ4446 và5test nhưng raw campaign FAILED_NO_RETRY do guard source cuối run. TU/Q exit0, raw UNVERIFIED. Kiểm tra ở audit.json chỉ xác minh artifacts/hash/trung bình các metric đã lưu; không chứng nhận encoder replay hay phép sắp xếp độc lập.', '',
             'Không so trực tiếp Sketchy104/21 với pseudo84/20. Không chọn điểm test tốt nhất thay cho checkpoint cuối. Ba seed mask không phải ba seed training.']
    (OUT / '03_results.md').write_text('\n'.join(text)+'\n')
    print('PASS: three archived sources/checkpoints; 30 condition NPZ means; 15 progress rows; CSV + PNG/SVG written')

if __name__ == '__main__':
    main()
