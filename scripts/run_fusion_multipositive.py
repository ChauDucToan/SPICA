"""Launch one explicit MP-family arm; immutable gates, no retries."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping

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
PCE_CAMPAIGN = 'coupled_predictive_fusion_mp_photo_ce_v1'
PCE_DIAGNOSTIC = 'fusion_mp_photo_ce_init_v1'
PCE_SCOPE = 'PHOTO_CE_INITIALIZATION_CALIBRATION_ONLY'
PCE_ARTIFACTS = {
    'provenance.json', 'gradient_layout.json', 'batch_records.json',
    'lambda_selection.json', 'measured_rows.json',
    *(f'raw_gradients_batch{i:02d}.pt' for i in range(4)),
}
PCE_CALIBRATION_COMPONENTS = {
    'scripts/diagnose_fusion_photo_ce.py', 'scripts/diagnose_coupled_sigreg.py',
    'src/spica/coupled_predictive_losses.py', 'src/spica/data/coupled_training.py',
    'src/spica/models/clip.py', 'src/spica/models/coupled_predictive.py',
    'src/spica/provenance.py', 'src/spica/train_coupled_predictive.py', 'uv.lock',
}
_HASH_SIZE = 64


def _repo_file(value: object, label: str, *, relative_only: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f'{label} must be a non-empty path')
    candidate = Path(value)
    if relative_only and (candidate.is_absolute() or '..' in candidate.parts):
        raise ValueError(f'{label} must be repository-relative')
    path = (ROOT / candidate if not candidate.is_absolute() else candidate).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'{label} must be an existing repository file')
    return path


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _HASH_SIZE:
        raise ValueError(f'{label} must be a SHA256 hex digest')
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f'{label} must be a SHA256 hex digest') from error
    return value


def _validate_hash_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f'{label} must be a non-empty object')
    result = {}
    for name, digest in value.items():
        path = _repo_file(name, f'{label} path', relative_only=True)
        result[name] = _digest(digest, f'{label}[{name!r}]')
        if _sha(path) != result[name]:
            raise ValueError(f'{label} hash mismatch: {name}')
    return result


def _validate_pce_artifacts(raw_path: Path, value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError('PCE raw diagnostic artifacts must be a non-empty list')
    seen = set()
    result = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError('PCE artifact record must be an object')
        name = row.get('path')
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or name in seen:
            raise ValueError(f'invalid PCE artifact path: {name!r}')
        artifact = (raw_path.parent / name).resolve()
        if not artifact.is_relative_to(ROOT) or not artifact.is_file():
            raise ValueError(f'missing PCE artifact: {name}')
        digest = _digest(row.get('sha256'), f'PCE artifact {name}')
        if _sha(artifact) != digest or row.get('bytes') != artifact.stat().st_size:
            raise ValueError(f'PCE artifact hash/size mismatch: {name}')
        seen.add(name)
        result.append({'path': name, 'bytes': artifact.stat().st_size, 'sha256': digest})
    if not PCE_ARTIFACTS <= seen:
        raise ValueError('PCE calibration artifact set is incomplete')
    return result


def _validate_pce_calibration(path: str) -> dict[str, Any]:
    verified_path = _repo_file(path, 'PCE reviewed calibration receipt')
    receipt = _read(verified_path)
    if not isinstance(receipt, Mapping) or receipt.get('status') != 'PASS' or receipt.get('verified') is not True or receipt.get('scope') != PCE_SCOPE:
        raise ValueError('PCE calibration receipt is not a verified initialization-only review')
    review = receipt.get('independent_review')
    if not isinstance(review, Mapping) or review.get('status') != 'PASS' or review.get('verified') is not True or review.get('independent') is not True:
        raise ValueError('PCE calibration requires an independent PASS review')
    evidence = review.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('PCE independent review evidence is missing')
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError('PCE review evidence record must be an object')
        evidence_path = _repo_file(row.get('path'), 'PCE review evidence', relative_only=True)
        if _sha(evidence_path) != _digest(row.get('sha256'), 'PCE review evidence hash'):
            raise ValueError(f'PCE review evidence hash mismatch: {row.get("path")!r}')
    value = receipt.get('lambda_photo_ce')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('PCE calibration lambda_photo_ce must be positive and finite')
    raw_path = _repo_file(receipt.get('raw_diagnostic_path'), 'PCE raw diagnostic', relative_only=True)
    raw_sha = _digest(receipt.get('raw_diagnostic_sha256'), 'PCE raw diagnostic hash')
    if _sha(raw_path) != raw_sha:
        raise ValueError('PCE raw diagnostic hash mismatch')
    raw = _read(raw_path)
    if not isinstance(raw, Mapping) or raw.get('status') != 'MEASURED_PENDING_REVIEW' or raw.get('verified') is not False:
        raise ValueError('PCE raw diagnostic must remain pending and unverified')
    config = raw.get('config')
    expected = {'diagnostic': PCE_DIAGNOSTIC, 'seed': 42, 'batches': 4, 'batch_size': 32,
                'rho': 0.1, 'tau': 0.07, 'positive_pool': 'full', 'optimizer_updates': 0,
                'wandb': 'disabled'}
    if not isinstance(config, Mapping) or any(config.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError('PCE raw diagnostic config binding mismatch')
    selection = raw.get('selection')
    if not isinstance(selection, Mapping) or selection.get('lambda_photo_ce') != value or receipt.get('selection') != selection:
        raise ValueError('PCE selection does not match raw diagnostic')
    candidates = selection.get('lambda_candidates')
    if not isinstance(candidates, Mapping) or not candidates or selection.get('binding_scope') not in candidates:
        raise ValueError('PCE lambda candidates/binding are missing')
    if any(isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or not math.isfinite(candidate) or candidate <= 0 for candidate in candidates.values()):
        raise ValueError('PCE lambda candidates must be positive and finite')
    if not isinstance(raw.get('initialization_hashes'), Mapping) or not isinstance(raw.get('data_identity'), Mapping):
        raise ValueError('PCE raw diagnostic identity bindings are missing')
    artifacts = _validate_pce_artifacts(raw_path, raw.get('artifacts'))
    inputs = _validate_hash_map(raw.get('input_sha256'), 'PCE diagnostic input')
    if review.get('raw_diagnostic_sha256') != raw_sha or review.get('lambda_photo_ce') != value:
        raise ValueError('PCE independent review is not bound to this raw calibration')
    historical = _validate_hash_map(raw.get('historical_bindings'), 'PCE historical binding')
    if not historical or not isinstance(raw.get('source_snapshot_hash'), str):
        raise ValueError('PCE source/historical binding is missing')
    _digest(raw['source_snapshot_hash'], 'PCE source snapshot hash')
    if any(raw.get(key) != raw.get(other) for key, other in (
        ('model_hash_before', 'model_hash_after'),
        ('teacher_hash_before', 'teacher_hash_after'),
    )) or raw.get('model_unchanged') is not True or raw.get('teacher_unchanged') is not True:
        raise ValueError('PCE no-update state guards failed')
    if raw.get('parameter_grad_all_none') is not True or raw.get('optimizer_updates') != 0 or raw.get('no_map_claim') is not True:
        raise ValueError('PCE no-update or no-evaluation guard failed')
    for key in ('config', 'selection', 'initialization_hashes', 'data_identity', 'input_sha256', 'historical_bindings', 'artifacts', 'source_snapshot_hash'):
        if receipt.get(key) != raw.get(key):
            raise ValueError(f'PCE reviewed receipt does not bind raw {key}')
    components = _components(receipt.get('component_sha256'), 'PCE calibration')
    if not PCE_CALIBRATION_COMPONENTS <= set(components):
        raise ValueError('PCE calibration component coverage is incomplete')
    for name, digest in components.items():
        if _sha(_repo_file(name, 'PCE calibration component', relative_only=True)) != digest:
            raise ValueError(f'PCE component changed: {name}')
    return {'path': str(verified_path), 'sha256': _sha(verified_path), 'receipt': dict(receipt),
            'raw_path': str(raw_path), 'raw_sha256': raw_sha, 'raw': dict(raw),
            'lambda_photo_ce': float(value), 'selection': dict(selection), 'config': dict(config),
            'initialization_hashes': dict(raw['initialization_hashes']), 'data_identity': dict(raw['data_identity']),
            'input_sha256': inputs, 'historical_bindings': historical, 'artifacts': artifacts,
            'source_snapshot_hash': raw['source_snapshot_hash'],
            'component_sha256': components}


def gate_check(path, arm=ARM, calibration=None):
    receipt = _read(path)
    if receipt.get('status') != 'PASS' or receipt.get('arm') != arm:
        raise ValueError(f'{arm} PASS launch gate required')
    components = _components(receipt.get('component_sha256'), 'gate')
    required = set(COMPONENTS)
    if arm in {'F2_QMP', 'F2_TQMP', 'F2_MP_PCE'}:
        required.add('scripts/check_fusion_qmp_cpu.py')
    if arm == 'F2_TQMP':
        required.update(('scripts/check_fusion_tqmp_cpu.py', 'scripts/diagnose_fusion_mp_tq.py'))
    if arm == 'F2_MP_PCE':
        required.update(('scripts/check_fusion_photo_ce_cpu.py', 'scripts/diagnose_fusion_photo_ce.py', 'scripts/diagnose_coupled_sigreg.py'))
        if not isinstance(calibration, Mapping) or receipt.get('calibration_identity_checked') is not True:
            raise ValueError('PCE gate must verify calibration identity')
        if receipt.get('lambda_photo_ce') != calibration['lambda_photo_ce'] or receipt.get('calibration_receipt_sha256', receipt.get('diagnostic_sha256')) != calibration['sha256']:
            raise ValueError('PCE gate calibration does not match receipt')
    if set(components) != required:
        raise ValueError('launch gate component coverage differs')
    for name, digest in components.items():
        if _sha(ROOT / name) != digest:
            raise ValueError(f'gate source changed: {name}')
    for row in receipt['evidence']:
        if arm == 'F2_MP_PCE':
            if not isinstance(row, Mapping):
                raise ValueError('PCE gate evidence record must be an object')
            evidence_path = _repo_file(row.get('path'), 'PCE gate evidence', relative_only=True)
            if _sha(evidence_path) != _digest(row.get('sha256'), 'PCE gate evidence hash'):
                raise ValueError(f'gate evidence changed: {row["path"]}')
        elif _sha(Path(row['path'])) != row['sha256']:
            raise ValueError(f'gate evidence changed: {row["path"]}')
    return receipt, components


def _pce_training_command(output: Path, lambda_photo_ce: float) -> list[str]:
    """PCE has a reviewed calibration receipt, not a trainer diagnostic input."""
    return [sys.executable, '-m', 'spica.train_coupled_predictive', '--arm', 'F2_MP_PCE',
            '--campaign-id', PCE_CAMPAIGN, '--campaign-root', str(output), '--output-dir',
            str(output / 'runs' / 'F2_MP_PCE'), '--device', 'cuda', '--max-steps', '3600',
            '--wandb-mode', 'online', '--lambda-photo-ce', str(lambda_photo_ce)]


def launch(args):
    arm = args.arm
    campaign = {'F2_MP': CAMPAIGN, 'F2_QMP': 'coupled_predictive_fusion_qmp_v1', 'F2_TQMP': 'coupled_predictive_fusion_tqmp_v1', 'F2_MP_PCE': PCE_CAMPAIGN}[arm]
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output == ROOT / 'outputs':
        raise ValueError('output must be a fresh child of outputs/')
    if output.exists():
        raise FileExistsError(output)
    # Single arm: historical checkpoint/probe sizes fit in ~12 GiB; retain headroom.
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise RuntimeError('single-arm launch requires at least24GiB free')
    gate_path = Path(args.gate).resolve()
    calibration = None
    if arm == 'F2_MP_PCE':
        if not args.diagnostic:
            raise ValueError('F2_MP_PCE requires --diagnostic calibration receipt')
        calibration = _validate_pce_calibration(args.diagnostic)
    gate, components = gate_check(gate_path, arm, calibration)
    diagnostic = None
    if arm == 'F2_TQMP':
        from spica.train_coupled_predictive import _validate_f2_tqmp_diagnostic
        if not args.diagnostic:
            raise ValueError('F2_TQMP requires --diagnostic')
        diagnostic = _validate_f2_tqmp_diagnostic(args.diagnostic)
    elif arm != 'F2_MP_PCE' and args.diagnostic:
        raise ValueError('only F2_TQMP and F2_MP_PCE accept --diagnostic in this runner')
    baseline_root = ROOT/'outputs/fusion_mp_execution_20260909T115500Z/runs/F2_MP' if arm in {'F2_QMP', 'F2_TQMP', 'F2_MP_PCE'} else BASELINE
    baseline = _read(baseline_root / 'run_result.json')
    latest = dict(baseline['selections']['latest'])
    readout = None
    if arm in {'F2_QMP', 'F2_TQMP', 'F2_MP_PCE'}:
        from spica.train_coupled_predictive import _validate_f2_qmp_readout
        readout = _validate_f2_qmp_readout()
        if latest['sha256'] != readout['checkpoint_sha256']:
            raise ValueError('baseline q readout checkpoint mismatch')
        latest['metrics'] = {'clean': readout['comparisons']['clean']['q'], 'masked_macro': readout['masked_macro']['q']}
    if baseline['status'] != 'COMPLETE' or latest['step'] != 3600:
        raise ValueError('historical F2 baseline is not COMPLETE3600')
    if _sha(baseline_root / latest['path']) != latest['sha256']:
        raise ValueError('baseline checkpoint hash mismatch')
    comparison = {'metric': 'clean/full_mAP', 'step': 3600, 'baseline_arm': 'F2_MP-Q' if readout else 'F2',
                  'baseline_readout': 'q' if readout else 'mu_i',
                  'baseline_run_id': baseline['wandb_run_id'],
                  'baseline_checkpoint_sha256': latest['sha256'],
                  'baseline_value': latest['metrics']['clean']['full_mAP'],
                  'selection': 'fixed_step', 'secondary': ['P@200', 'mAP@200_prefix_positive', 'masked_macro/full_mAP']}
    if readout:
        from spica.train_coupled_predictive import F2_QMP_READOUT, F2_QMP_READOUT_SHA256
        comparison.update(baseline_evaluation=str(F2_QMP_READOUT), baseline_evaluation_sha256=F2_QMP_READOUT_SHA256)
    resolved = {'arm': arm, 'campaign_id': campaign, 'max_steps': 3600,
                'wandb_mode': 'online', 'lambda_sig': 0, 'temperature': .07,
                'primary_comparison': comparison, 'gate_sha256': _sha(gate_path)}
    if diagnostic:
        resolved.update({k: diagnostic[k] for k in ('lambda_mp_t', 'lambda_mp_q', 'diagnostic_sha256')})
        resolved.update(main_query='q', main_photo_objective='multi_positive_three_head_contrastive', diagnostic_source_snapshot_hash=diagnostic['source_snapshot_hash'])
    if calibration:
        resolved.update({
            'lambda_photo_ce': calibration['lambda_photo_ce'],
            'calibration_identity': {
                'scope': PCE_SCOPE,
                'verified_receipt_sha256': calibration['sha256'],
                'raw_diagnostic_path': calibration['raw_path'],
                'raw_diagnostic_sha256': calibration['raw_sha256'],
                'initialization_hashes': calibration['initialization_hashes'],
                'data_identity': calibration['data_identity'],
                'input_sha256': calibration['input_sha256'],
                'selection': calibration['selection'],
            },
            'training_main_query': 'mu_i', 'evaluation_query': 'q',
            'objective_identity': 'F2_MP_total+lambda_photo_ce*CE(actual_unique_live_photos,learned_train_text_bank)',
        })
    provenance = capture_provenance(ROOT, resolved_config=resolved)
    snapshot = provenance['source_snapshot']
    manifest = {'head_commit': provenance['head_commit'], 'source_snapshot_hash': snapshot['sha256'],
                'source_snapshot': snapshot, 'resolved_config': resolved}
    output.mkdir()
    runtime = {'status': 'LAUNCHING', 'arm': arm, 'started_at': _now(),
               'source_snapshot_hash': snapshot['sha256'], 'child_pid': None}
    try:
        _json_atomic(output/'runtime.json', runtime)
        _json_atomic(output/'execution_manifest.json', manifest)
        _copy_source_archive(output, manifest, provenance)
        _check_components(output, snapshot, {'gate': components})
        shutil.copyfile(gate_path, output/'launch_gate.json')
        if _sha(output/'launch_gate.json') != resolved['gate_sha256']:
            raise RuntimeError('copied gate hash mismatch')
        if readout:
            shutil.copyfile(F2_QMP_READOUT, output/'baseline_q_readout.json')
            if _sha(output/'baseline_q_readout.json') != F2_QMP_READOUT_SHA256:
                raise ValueError('archived baseline readout mismatch')
        if diagnostic:
            shutil.copyfile(args.diagnostic, output/'diagnostic_verified.json')
            if _sha(output/'diagnostic_verified.json') != diagnostic['diagnostic_sha256']:
                raise ValueError('archived calibration receipt mismatch')
            shutil.copyfile(ROOT/diagnostic['raw_diagnostic_path'], output/'diagnostic_measurement.json')
            if _sha(output/'diagnostic_measurement.json') != diagnostic['raw_diagnostic_path_sha256']:
                raise ValueError('archived raw measurement mismatch')
        if calibration:
            for source, name, digest in ((calibration['path'], 'calibration_receipt.json', calibration['sha256']), (calibration['raw_path'], 'calibration_raw.json', calibration['raw_sha256'])):
                shutil.copyfile(source, output/name)
                if _sha(output/name) != digest:
                    raise RuntimeError(f'archived {name} hash mismatch')
        run = output/'runs'/arm
        command = [sys.executable, '-m', 'spica.train_coupled_predictive', '--arm', arm,
                   '--campaign-id', campaign, '--campaign-root', str(output), '--output-dir', str(run),
                   '--device', 'cuda', '--max-steps', '3600', '--wandb-mode', 'online']
        if diagnostic:
            command += ['--diagnostic', str(output/'diagnostic_verified.json')]
        if calibration:
            command = _pce_training_command(output, calibration['lambda_photo_ce'])
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
            'status':'COMPLETE', 'arm':arm, 'campaign':campaign,
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
    parser.add_argument('--arm', choices=('F2_MP','F2_QMP','F2_TQMP','F2_MP_PCE'), default=ARM)
    parser.add_argument('--diagnostic')
    parser.add_argument('--output')
    parser.add_argument('--gate')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps({'status':'INERT','arm':args.arm,'would_launch':False}))
        return 0
    if not args.output or not args.gate:
        parser.error('--launch requires --output and --gate')
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
