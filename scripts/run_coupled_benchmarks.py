"""Locked TU -> QuickDraw MP-Q campaign; inert without --launch, no retries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from run_coupled_campaign import _copy_source_archive, _json_atomic, _now, _start_child
from run_fusion_campaign import _read, _sha
from spica.provenance import capture_provenance
from spica.train_coupled_benchmark import ARM, DATASETS, METHOD_VERSION

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    'scripts/run_coupled_benchmarks.py', 'scripts/evaluate_coupled_benchmark.py',
    'scripts/check_coupled_benchmark_cpu.py', 'scripts/check_coupled_benchmark_data_cpu.py',
    'src/spica/train_coupled_benchmark.py', 'src/spica/data/coupled_benchmark.py',
    'src/spica/evaluation/coupled_benchmark.py', 'src/spica/train_coupled_predictive.py',
    'src/spica/models/coupled_predictive.py', 'src/spica/models/clip.py',
    'src/spica/coupled_predictive_losses.py', 'src/spica/data/coupled_training.py',
    'src/spica/data/datasets.py', 'src/spica/data/coupled_views.py', 'src/spica/data/masking.py',
    'src/spica/evaluation/coupled_predictive.py', 'src/spica/evaluation/metrics.py',
    'src/spica/evaluation/masked_view.py', 'src/spica/provenance.py',
    'src/spica/tracking/wandb.py', 'scripts/run_coupled_campaign.py',
    'scripts/run_fusion_campaign.py', 'uv.lock',
)


def gate_check(path: Path):
    gate = _read(path)
    assert gate['status'] == 'PASS' and gate['method'] == METHOD_VERSION
    assert gate['datasets'] == DATASETS
    assert set(gate['component_sha256']) == set(COMPONENTS)
    for name, digest in gate['component_sha256'].items():
        assert _sha(ROOT / name) == digest, name
    for row in gate['evidence']:
        p = (ROOT / row['path']).resolve()
        assert p.is_relative_to(ROOT) and _sha(p) == row['sha256'], row['path']
    assert set(gate['smoke_roots']) == set(DATASETS)
    return gate


def check_finished_train(run: Path, smoke: Path, dataset: str, source_hash: str):
    result = _read(run / 'run_result.json')
    cfg = _read(run / 'resolved_config.json')
    control = _read(smoke / 'resolved_config.json')
    assert result['status'] == 'COMPLETE' and result['step'] == DATASETS[dataset]['total_steps']
    assert result['source_snapshot_hash'] == source_hash and result['arm'] == ARM
    assert not cfg['smoke'] and cfg['selection_policy'] == 'none;final_only'
    assert cfg['protocol_identity'] == control['protocol_identity']
    assert cfg['initialization_hashes'] == control['initialization_hashes']
    assert cfg['loss_coefficient_identity'] == control['loss_coefficient_identity']
    for name, count in (('observation_trace.jsonl', 64), ('mask_metadata.jsonl', 64), ('lr_history_every_step.jsonl', 3)):
        with (run / name).open() as handle:
            prefix = [next(handle) for _ in range(count)]
        assert prefix == (smoke / name).read_text().splitlines(keepends=True), name
    assert result['frozen_original_state_hash_before'] == result['frozen_original_state_hash_after']
    assert list(result['selections']) == ['latest']
    final = result['selections']['latest']
    assert _sha(run / final['path']) == final['sha256'] and final['step'] == result['step']
    return result


def launch(args):
    output = Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'outputs') or output == ROOT / 'outputs':
        raise ValueError('fresh child of outputs/ required')
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError('at least 24GiB free required; no artifact cleanup permitted')
    gate = gate_check(Path(args.gate))
    config = {'method': METHOD_VERSION, 'arm': ARM, 'datasets': DATASETS,
              'order': list(DATASETS), 'evaluation': 'final_only_clean_plus_9_masks',
              'official_unseen_used_for_selection': False, 'no_retry_or_resume': True,
              'gate_sha256': _sha(Path(args.gate))}
    provenance = capture_provenance(ROOT, resolved_config=config)
    source_hash = provenance['source_snapshot']['sha256']
    manifest = {'head_commit': provenance['head_commit'], 'source_snapshot_hash': source_hash,
                'source_snapshot': provenance['source_snapshot'], 'resolved_config': config}
    output.mkdir()
    _copy_source_archive(output, manifest, provenance)
    _json_atomic(output / 'execution_manifest.json', manifest)
    shutil.copyfile(args.gate, output / 'launch_gate.json')
    runtime = {'status': 'LAUNCHING', 'source_snapshot_hash': source_hash, 'started_at': _now(),
               'child_pid': None, 'completed': [], 'dataset': None, 'phase': None}
    _json_atomic(output / 'runtime.json', runtime)
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'online', 'PYTHONPATH': str(ROOT / 'src')}
    results = {}
    try:
        for dataset in DATASETS:
            run = output / 'runs' / dataset
            evaluation = output / 'evaluation' / dataset
            commands = {
                'train': [sys.executable, '-m', 'spica.train_coupled_benchmark', '--dataset', dataset,
                          '--output-dir', str(run), '--campaign-root', str(output), '--device', 'cuda', '--wandb-mode', 'online'],
                'evaluate': [sys.executable, 'scripts/evaluate_coupled_benchmark.py', '--run-dir', str(run),
                             '--output-dir', str(evaluation), '--device', 'cuda'],
            }
            for phase, command in commands.items():
                assert capture_provenance(ROOT)['source_snapshot']['sha256'] == source_hash
                runtime.update(status='RUNNING', dataset=dataset, phase=phase, command=command)
                child, stdout, stderr = _start_child(command, cwd=ROOT, env=env,
                    stdout=output / 'logs' / f'{dataset}.{phase}.stdout.log',
                    stderr=output / 'logs' / f'{dataset}.{phase}.stderr.log')
                runtime['child_pid'] = child.pid
                _json_atomic(output / 'runtime.json', runtime)
                try:
                    code = child.wait()
                finally:
                    stdout.close()
                    stderr.close()
                runtime.update(child_pid=None, exit_code=code)
                if code:
                    raise RuntimeError(f'{dataset}/{phase} exited {code}; no retry')
                assert capture_provenance(ROOT)['source_snapshot']['sha256'] == source_hash
                if phase == 'train':
                    train = check_finished_train(run, ROOT / gate['smoke_roots'][dataset], dataset, source_hash)
                    _json_atomic(run / 'parent_final_gate.json', {'status': 'PASS', 'scope': 'final_checkpoint_init_inputs_before_official_evaluation', 'step': train['step'], 'checkpoint_sha256': train['selections']['latest']['sha256']})
                else:
                    stats = _read(evaluation / 'summary.json')
                    assert stats['status'] == 'COMPLETE' and stats['official_test_evaluated'] is True
                    assert stats['step'] == train['step'] and stats['source_snapshot_hash'] == source_hash
                    assert stats['predictor_forwards'] == 0 and len(stats['conditions']) == 10
                    assert stats['model_state_before'] == stats['model_state_after']
                    results[dataset] = {'step': train['step'], 'checkpoint_sha256': train['selections']['latest']['sha256'],
                        'wandb_url': train['wandb_url'], 'clean': stats['clean'], 'masked_macro': stats['masked_macro'],
                        'masked_by_fraction': stats['masked_by_fraction'], 'evaluation_summary_sha256': _sha(evaluation / 'summary.json')}
            runtime['completed'].append(dataset)
            _json_atomic(output / 'runtime.json', runtime)
        _json_atomic(output / 'results.json', {'status': 'LOCAL_RESULTS_PENDING_INDEPENDENT_REVIEW', 'source_snapshot_hash': source_hash, 'datasets': results})
        runtime.update(status='TRAIN_AND_OFFICIAL_EVAL_FINISHED_UNVERIFIED', finished_at=_now())
        _json_atomic(output / 'runtime.json', runtime)
        return 0
    except BaseException:
        runtime.update(status='FAILED_NO_RETRY', finished_at=_now(), traceback=traceback.format_exc())
        _json_atomic(output / 'runtime.json', runtime)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--gate')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps({'status': 'INERT', 'would_launch': False, 'datasets': DATASETS}))
        return 0
    if not args.output or not args.gate:
        parser.error('--launch requires --output and --gate')
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
