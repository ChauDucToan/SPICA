"""Rebuild this pilot's handoff directly from its raw evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import torch

from spica.alignment_artifacts import canonical_sha256
from spica.train_alignment import _rng_matches

OUT = Path(__file__).resolve().parent
RUNS = OUT / 'fixed_attempt/pilot'
STEPS = (0, 100, 500, 1000, 1800)
matching = json.loads((OUT / 'matching_validation.json').read_text())
assert matching['matching_mode'] == 'corrected_v2'
assert all(p['status'] == 'MATCHED' for ps in matching['pairs'].values() for p in ps)
assert matching['MD_MS'][0]['status'] == 'MATCHED'
calpath = OUT / 'fixed_attempt/calibration/calibration.json'
cal = json.loads(calpath.read_text())
calhash = hashlib.sha256(calpath.read_bytes()).hexdigest()
assert cal['status'] == 'VALID'
assert cal['calibration']['lambda_alignment_mean'] == 0.1 * statistics.median(b/d for b,d in zip(cal['base']['sketch_gradient_norms'], cal['detached']['sketch_gradient_norms'], strict=True))
shutil.copy2(calpath, OUT / 'calibration_diagnostic_corrected.json')
source = json.loads((OUT / 'source_snapshot_fixed.json').read_text())
assert source['sha256'] == cal['source_snapshot_hash']
initial = json.loads((OUT / 'artifact_inventory_initial.json').read_text())
inventory = initial['artifacts'] + json.loads((OUT / 'artifact_inventory_historical_references.json').read_text())
raws = {}
summary = []
checkpoints = {}
for arm in ('R', 'MD', 'MS'):
    path = RUNS / arm / 'run_result.json'
    raw = json.loads(path.read_text())
    raws[arm] = raw
    history = json.loads((path.parent / 'training_history.json').read_text())
    assert history == raw['training_history']
    assert [h['training_global_step'] for h in history] == list(range(1, 1801))
    assert [h['training_global_step'] for h in raw['history']] == list(STEPS)
    assert raw['source_snapshot_hash'] == source['sha256']
    assert raw['initial_model_state_hash'] == cal['initial_model_state_hash']
    assert raw['initial_text_bank_state_hash'] == cal['initial_text_bank_state_hash']
    assert raw['clip_freeze_policy']['all_clip_owned_parameters_byte_identical']
    if arm != 'R':
        assert raw['gradient_calibration']['artifact_sha256'] == calhash
        assert Path(raw['gradient_calibration']['artifact']).resolve() == calpath.resolve()
        assert raw['gradient_calibration']['calibration_batch_replay_verified']
    shutil.copy2(path, OUT / 'raw_metrics' / f'{arm}_run_result.json')
    shutil.copy2(path.parent / 'training_history.json', OUT / 'raw_metrics' / f'{arm}_training_history.json')
    config = {'run_result': str(path), 'run_result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'resolved_config': raw['resolved_config'], 'config_hash': canonical_sha256(raw['resolved_config']), 'run_id': raw['manifest_entry_identity'], 'source_snapshot_hash': source['sha256'], 'source_archive': 'training_source_fixed.tar.gz'}
    (OUT / 'resolved_configs' / f'{arm}.json').write_text(json.dumps(config, indent=2)+'\n')
    cp_records = []
    for h in raw['history']:
        step = h['training_global_step']
        cp = Path(h['checkpoint'])
        digest = hashlib.sha256(cp.read_bytes()).hexdigest()
        assert digest == h['checkpoint_sha256']
        loaded = torch.load(cp, map_location='cpu', weights_only=True)
        assert loaded['step'] == loaded['training_global_step'] == step
        assert loaded['full_pseudo_unseen_mAP'] == h['full_pseudo_unseen_mAP']
        assert loaded['source_snapshot_hash'] == source['sha256']
        assert loaded['clip_freeze_policy']['all_clip_owned_parameters_byte_identical']
        assert json.loads((path.parent / f'probe_step{step}.json').read_text()) == h
        ap = h['val']['average_precision_per_query']
        assert len(ap) == h['val']['query_identity']['count']
        assert all(0 <= value <= 1 for value in ap)
        assert abs(statistics.mean(ap)-h['full_pseudo_unseen_mAP']) < 1e-12
        checkpoints[arm, step] = loaded
        cp_records.append({'step': step, 'path': str(cp.resolve()), 'sha256': digest, 'metric': h['full_pseudo_unseen_mAP'], 'AP_mean_recomputed_float64': statistics.mean(ap), 'status': 'COMPLETED'})
    final = raw['history'][-1]
    peak = max(raw['history'], key=lambda h: (h['full_pseudo_unseen_mAP'], -h['training_global_step']))
    trajectory = []
    for h in raw['history']:
        step = h['training_global_step']
        geometry_path = OUT / 'raw_metrics' / f'geometry_{arm}_{step}.json'
        gap = None
        if geometry_path.exists():
            geo = json.loads(geometry_path.read_text())
            assert geo['checkpoint_sha256'] == h['checkpoint_sha256']
            gap = geo['variants']['fully_corrected_geometry']['pseudo_unseen']['mean_distance']
        trajectory.append({'step': step, 'mAP': h['full_pseudo_unseen_mAP'], 'semantic_margin': h['semantic_margin'], 'sketch_effective_rank': h['geometry']['sketch']['effective_rank'], 'sketch_mean_feature_variance': h['geometry']['sketch']['mean_feature_variance'], 'photo_effective_rank': h['geometry']['photo']['effective_rank'], 'corrected_mean_gap': gap, 'corrected_mean_gap_reason': None if gap is not None else 'offline_geometry_not_scheduled_at_this_step', 'geometry_artifact': str(geometry_path) if gap is not None else None, 'checkpoint': h['checkpoint'], 'checkpoint_sha256': h['checkpoint_sha256']})
    summary.append({'arm': arm, 'seed': 42, 'mAP@1800': final['full_pseudo_unseen_mAP'], 'matching_status': 'MATCHED' if arm != 'R' else 'VALID_CONTROL', 'peak_mAP': peak['full_pseudo_unseen_mAP'], 'peak_step': peak['training_global_step'], 'absolute_decay': peak['full_pseudo_unseen_mAP']-final['full_pseudo_unseen_mAP'], 'runtime': raw['runtime'], 'trajectory': trajectory, 'lineage': config, 'checkpoints': cp_records})
    inventory.append({'role': arm, 'seed': 42, 'split': raw['pseudo_split_identity']['sha256'], 'horizon': 1800, 'path': str(path.parent), 'status': 'COMPLETED', 'reuse_eligibility': 'Eligible: corrected_v2 VALID, all pairings MATCHED, raw histories and checkpoint bytes verified.', 'checkpoints': cp_records})
rng_checks = []
for step in STEPS:
    for arm in ('MD', 'MS'):
        left, right = checkpoints['R', step], checkpoints[arm, step]
        assert _rng_matches(left['rng_state'], right['rng_state']), (arm, step)
        if step == 0:
            for key in ('model_state_dict', 'soft_prompt_state_dict'):
                assert left[key].keys() == right[key].keys()
                assert all(torch.equal(left[key][name], right[key][name]) for name in left[key])
        rng_checks.append({'arm_vs_R': arm, 'step': step, 'rng_equal': True, 'initial_prompts_and_soft_text_equal': True if step == 0 else None})
(OUT / 'checkpoint_state_validation.json').write_text(json.dumps(rng_checks, indent=2)+'\n')
for entry in summary:
    entry['delta_vs_R'] = entry['mAP@1800']-summary[0]['mAP@1800']
bootstrap = {pair: json.loads((OUT / 'raw_metrics' / f'bootstrap_{pair}.json').read_text()) for pair in ('MD_R', 'MS_R', 'MS_MD')}
md_delta, ms_delta = summary[1]['delta_vs_R'], summary[2]['delta_vs_R']
conclusion = ('Negative result for this seed42/pseudo-split pilot: MD and MS both lose to R at step1800. No mainline promotion.' if md_delta < 0 and ms_delta < 0 else 'At least one arm is a candidate for future confirmation, not a robust improvement or mainline result.')
for directory in sorted((OUT / 'fixed_attempt/smoke').iterdir()):
    if (directory / 'run_result.json').exists():
        inventory.append({'role': directory.name, 'seed': 42, 'split': 3407, 'horizon': 50, 'path': str(directory), 'status': 'COMPLETED', 'reuse_eligibility': 'Smoke-only; excluded from scientific tables.'})
inventory.extend([
    {'role': 'calibration_corrected', 'seed': 42, 'split': cal['split_identity_hash'], 'horizon': 1800, 'path': str(calpath), 'sha256': calhash, 'status': 'COMPLETED', 'reuse_eligibility': 'Used by both MD and MS with exact source/init/config/fixed-batch identity.'},
    {'role': 'calibration_before_execution_fix', 'seed': 42, 'split': 3407, 'horizon': 1800, 'path': str(OUT/'calibration/calibration.json'), 'status': 'COMPLETED', 'reuse_eligibility': 'Superseded source snapshot; preserved, not attached to fixed runs.'},
    {'role': 'R_failed_smoke_at_HEAD', 'seed': 42, 'split': 3407, 'horizon': 50, 'path': str(OUT/'smoke/R'), 'status': 'INCOMPLETE', 'reuse_eligibility': 'Reproducer: TypeError at probe(0), zero updates; retained unchanged.'},
])
(OUT / 'artifact_inventory.json').write_text(json.dumps({'search_roots': initial['search_roots']+[str(OUT)], 'artifacts': inventory}, indent=2)+'\n')
result = {'status': 'COMPLETED', 'primary_horizon': 1800, 'training_seed': 42, 'pseudo_split_seed': 3407, 'split_sha256': cal['split_identity_hash'], 'lambda_mean_MD_MS': cal['calibration']['lambda_alignment_mean'], 'calibration_artifact_sha256': calhash, 'arms': summary, 'MS_minus_MD': summary[2]['mAP@1800']-summary[1]['mAP@1800'], 'paired_bootstrap': bootstrap, 'uncertainty_scope': 'Paired queries on ONE training seed and ONE pseudo split; not multi-seed evidence.', 'scientific_conclusion': conclusion, 'source_snapshot_hash': source['sha256'], 'source_patch': 'execution_fix.patch', 'official_unseen_run': False, 'published': False}
(OUT / 'corrected_pilot_summary.json').write_text(json.dumps(result, indent=2)+'\n')
text = '# Corrected pilot: verified raw-metric replay\n\nSeed42; pseudo split3407; 84 train / 20 pseudo-validation classes; covariance=0. Fixed horizon1800.\n\n| Arm | Seed | mAP@1800 | Delta vs R | Matching status |\n|---|---:|---:|---:|---|\n'
for e in summary:
    text += f"| {e['arm']} | 42 | {e['mAP@1800']:.9f} | {e['delta_vs_R']:+.9f} | {e['matching_status']} |\n"
text += '\n| Arm | Peak mAP | Peak step | mAP@1800 | Absolute decay |\n|---|---:|---:|---:|---:|\n'
for e in summary:
    text += f"| {e['arm']} | {e['peak_mAP']:.9f} | {e['peak_step']} | {e['mAP@1800']:.9f} | {e['absolute_decay']:.9f} |\n"
text += '\nPeak is only the maximum over the five scheduled evaluations, not a continuous-training maximum. Candidate peaks are not compared to a fixed-step control.\n\n## Query uncertainty\n\nExact query IDs/order and gallery identity match before bootstrap. 10,000 paired resamples; bootstrap seed3407.\n\n| Difference | Mean AP delta | 95% query-bootstrap CI |\n|---|---:|---|\n'
for pair, value in bootstrap.items():
    b = value['bootstrap']
    text += f"| {pair.replace('_',' − ')} | {b['mean_delta']:+.9f} | [{b['ci_95'][0]:+.9f}, {b['ci_95'][1]:+.9f}] |\n"
text += '\nThese intervals quantify queries of one seed/split, NOT robustness across seeds/classes/splits. Bootstrap and checkpoint mAP both use float64 means of the stored AP values (independent recomputation agrees within 1e-12).\n\n## Geometry and efficiency\n\n| Arm | Step | mAP | Semantic margin | Corrected mean gap | Sketch effective rank | Sketch mean feature variance |\n|---|---:|---:|---:|---:|---:|---:|\n'
for e in summary:
    for h in e['trajectory']:
        if h['corrected_mean_gap'] is not None:
            text += f"| {e['arm']} | {h['step']} | {h['mAP']:.6f} | {h['semantic_margin']:.6f} | {h['corrected_mean_gap']:.6f} | {h['sketch_effective_rank']:.3f} | {h['sketch_mean_feature_variance']:.8f} |\n"
text += '\nMean gaps use fixed first16 images/class, correctly routed photo prompts and valid log-map samples; no fitting on validation. Spread/rank and semantic margin use the existing training-probe subset/protocol; these are different summaries, not identical subsets.\n\n| Arm | Training seconds | Amortized seconds/update | Peak CUDA allocation (bytes) |\n|---|---:|---:|---:|\n'
for e in summary:
    rt = e['runtime']
    text += f"| {e['arm']} | {rt['training_seconds']:.3f} | {rt['seconds_per_update']:.6f} | {rt['peak_gpu_memory_bytes']} |\n"
text += '\nTiming includes scheduled probes after step0; it is not isolated optimizer/kernel timing. Peak allocation excludes the initial probe reset. Raw and weighted losses and per-objective gradient norms/ratios/cosines at0/500/1800 are in `raw_metrics/offline_gradients_*.json`; full per-update raw losses are in each raw history. Offline diagnostics never perturb training.\n\n## Conclusion\n\n'+conclusion+'\nNo novelty claim follows from these numbers. Mean agreement alone does not establish retrieval utility. No extra lambda, covariance, horizon, seed or official-unseen campaign was tried.\n\n## Traceability\n\n`corrected_pilot_summary.json` links every row to run_result/config/source, exact checkpoint step/path/SHA256 and raw history. `resolved_configs/`, `raw_metrics/`, `matching_validation.json`, `checkpoint_state_validation.json` and `training_source_fixed.tar.gz` retain the evidence. Historical results are inventory-only, not numbers substituted into these tables. This bundle is local, not published.\n'
(OUT / 'corrected_pilot_summary.md').write_text(text)
print(json.dumps({'arms': [{k:e[k] for k in ('arm','mAP@1800','delta_vs_R','peak_mAP','peak_step')} for e in summary], 'MS_minus_MD': result['MS_minus_MD'], 'conclusion': conclusion}, indent=2))
