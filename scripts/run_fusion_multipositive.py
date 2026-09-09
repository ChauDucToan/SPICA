"""Launch exactly one authorized F2_MP arm, with immutable gates and no retries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from run_coupled_campaign import _copy_source_archive, _json_atomic, _now, _start_child
from run_fusion_campaign import _check_components, _components, _read, _sha
from spica.provenance import capture_provenance

ROOT = Path(__file__).resolve().parents[1]
ARM = 'F2_MP'
CAMPAIGN = 'coupled_predictive_fusion_mp_v1'
BASELINE = ROOT / 'outputs/fusion_execution_20260909T064000Z/runs/F2'
COMPONENTS = (
    'src/spica/coupled_predictive_losses.py', 'src/spica/data/coupled_training.py',
    'src/spica/train_coupled_predictive.py', 'src/spica/models/coupled_predictive.py',
    'src/spica/evaluation/coupled_predictive.py', 'src/spica/tracking/wandb.py',
    'scripts/check_fusion_multipositive_cpu.py', 'scripts/run_fusion_multipositive.py',
    'scripts/run_coupled_campaign.py', 'scripts/run_fusion_campaign.py', 'uv.lock',
)


def gate_check(path):
    receipt = _read(path)
    if receipt.get('status') != 'PASS' or receipt.get('arm') != ARM:
        raise ValueError('F2_MP PASS launch gate required')
    components = _components(receipt.get('component_sha256'), 'gate')
    if set(components) != set(COMPONENTS):
        raise ValueError('launch gate component coverage differs')
    for name, digest in components.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f'gate source changed: {name}')
    for row in receipt['evidence']:
        if _sha(Path(row['path'])) != row['sha256']:
            raise ValueError(f'gate evidence changed: {row["path"]}')
    return receipt, components


def launch(args):
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output == ROOT / 'outputs':
        raise ValueError('output must be a fresh child of outputs/')
    if output.exists():
        raise FileExistsError(output)
    # Single arm: historical checkpoint/probe sizes fit in ~12 GiB; retain headroom.
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError('single-arm launch requires at least24GiB free')
    gate_path = Path(args.gate).resolve()
    gate, components = gate_check(gate_path)
    baseline = _read(BASELINE / 'run_result.json')
    latest = baseline['selections']['latest']
    if baseline['status'] != 'COMPLETE' or latest['step'] != 3600:
        raise ValueError('historical F2 baseline is not COMPLETE3600')
    if _sha(BASELINE / latest['path']) != latest['sha256']:
        raise ValueError('baseline checkpoint hash mismatch')
    comparison = {'metric': 'clean/full_mAP', 'step': 3600, 'baseline_arm': 'F2',
                  'baseline_run_id': baseline['wandb_run_id'],
                  'baseline_checkpoint_sha256': latest['sha256'],
                  'baseline_value': latest['metrics']['clean']['full_mAP'],
                  'selection': 'fixed_step', 'secondary': ['P@200', 'mAP@200_prefix_positive', 'masked_macro/full_mAP']}
    resolved = {'arm': ARM, 'campaign_id': CAMPAIGN, 'max_steps': 3600,
                'wandb_mode': 'online', 'lambda_sig': 0, 'temperature': .07,
                'primary_comparison': comparison, 'gate_sha256': _sha(gate_path)}
    provenance = capture_provenance(ROOT, resolved_config=resolved)
    snapshot = provenance['source_snapshot']
    manifest = {'head_commit': provenance['head_commit'], 'source_snapshot_hash': snapshot['sha256'],
                'source_snapshot': snapshot, 'resolved_config': resolved}
    output.mkdir()
    runtime = {'status': 'LAUNCHING', 'arm': ARM, 'started_at': _now(),
               'source_snapshot_hash': snapshot['sha256'], 'child_pid': None}
    try:
        _json_atomic(output/'runtime.json', runtime)
        _json_atomic(output/'execution_manifest.json', manifest)
        _copy_source_archive(output, manifest, provenance)
        _check_components(output, snapshot, {'gate': components})
        shutil.copyfile(gate_path, output/'launch_gate.json')
        if _sha(output/'launch_gate.json') != resolved['gate_sha256']:
            raise RuntimeError('copied gate hash mismatch')
        run = output/'runs'/ARM
        command = [sys.executable, '-m', 'spica.train_coupled_predictive', '--arm', ARM,
                   '--campaign-id', CAMPAIGN, '--campaign-root', str(output), '--output-dir', str(run),
                   '--device', 'cuda', '--max-steps', '3600', '--wandb-mode', 'online']
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'online', 'PYTHONPATH': str(ROOT/'src')}
        child, out, err = _start_child(command, cwd=ROOT, env=env,
                                      stdout=output/'logs/train.stdout.log', stderr=output/'logs/train.stderr.log')
        runtime.update(child_pid=child.pid, command=command)
        _json_atomic(output/'runtime.json', runtime)
        try:
            code = child.wait()
        finally:
            out.close()
            err.close()
        runtime.update(exit_code=code, child_pid=None, finished_at=_now())
        result = _read(run/'run_result.json') if (run/'run_result.json').exists() else {}
        if code != 0 or any(result.get(k) != v for k,v in {
            'status':'COMPLETE', 'arm':ARM, 'campaign':CAMPAIGN,
            'step':3600, 'source_snapshot_hash':snapshot['sha256']}.items()):
            runtime['status'] = 'TRAINING_FAILED'
            _json_atomic(output/'runtime.json', runtime)
            return code or 1
        runtime.update(status='ARM_FINISHED_UNVERIFIED', wandb_url=result['wandb_url'])
        _json_atomic(output/'runtime.json', runtime)
        try:
            current = result['selections']['latest']['metrics']
            _json_atomic(output/'comparison.json', {
                'status':'LOCAL_COMPARISON_NOT_ONLINE_VERIFICATION', 'comparison':comparison,
                'candidate_metrics':current, 'baseline_metrics':latest['metrics'],
                'clean_full_mAP_delta':current['clean']['full_mAP']-comparison['baseline_value'],
                'improved_clean_full_mAP':current['clean']['full_mAP']>comparison['baseline_value'],
                'candidate_checkpoint_sha256':result['selections']['latest']['sha256'],
                'wandb_url':result['wandb_url'],
            })
        except Exception:
            _json_atomic(output/'postprocess_failure.json', {'status':'FAIL','traceback':traceback.format_exc()})
            return 1
        return 0
    except BaseException:
        runtime.update(status='LAUNCH_OR_RUNNER_FAILED', traceback=traceback.format_exc())
        _json_atomic(output/'runtime.json', runtime)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--gate')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps({'status':'INERT','arm':ARM,'would_launch':False}))
        return 0
    if not args.output or not args.gate:
        parser.error('--launch requires --output and --gate')
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
