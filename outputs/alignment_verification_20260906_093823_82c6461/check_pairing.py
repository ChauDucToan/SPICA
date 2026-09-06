"""One-shot negative checks for the concrete gaps in existing pairing tests."""
import copy
import json
from pathlib import Path
import runpy
import tempfile

from scripts.summarize_alignment import _direct_corrected_pair, _pair, _validate, write_report

OUT = Path(__file__).resolve().parent
fixture = runpy.run_path('tests/test_alignment_corrected_reporting.py')
records = []
with tempfile.TemporaryDirectory(prefix='spica-pairing-') as directory:
    root = Path(directory)
    fixture['_write_runs'](root)
    runs = []
    for role in fixture['ROLES']:
        path = root / role / 'run_result.json'
        raw = json.loads(path.read_text())
        raw['_artifact_path'] = str(path)
        runs.append(_validate(raw, 2))
    r, md, ms = runs
    assert _pair(md, [r])['status'] == 'MATCHED'
    assert _pair(ms, [r])['status'] == 'MATCHED'
    assert _direct_corrected_pair(md, ms)['status'] == 'MATCHED'
    for key, value in [('batch_size', 64), ('num_positive_photos', 2),
                       ('lambda_rank', 2.0), ('lambda_cls', 2.0),
                       ('visual_prompt_learning_rate', 0.01),
                       ('visual_prompt_weight_decay', 0.01)]:
        changed = copy.deepcopy(md)
        changed['resolved_config'][key] = value
        result = _pair(changed, [r])
        assert result['status'] == 'UNMATCHED_CONFIG', (key, result)
        assert result['paired_delta'] is None
        records.append({'mutation': key, 'result': result})
    for groups in ([], [{'name': 'different_optimizer'}]):
        changed = copy.deepcopy(md)
        changed['optimizer_identity'] = groups
        result = _pair(changed, [r])
        assert result['status'] != 'MATCHED' and result['paired_delta'] is None
        records.append({'mutation': 'optimizer_groups', 'value': groups, 'result': result})
    changed = copy.deepcopy(ms)
    changed['resolved_config']['lambda_alignment_mean'] = 0.3
    result = _direct_corrected_pair(md, changed)
    assert result['status'] == 'UNMATCHED_CONFIG'
    records.append({'mutation': 'MD-MS unequal lambda', 'result': result})
    for field, value in [('artifact_path', str(root / 'other_calibration.json')),
                         ('artifact_sha256', 'bad-hash')]:
        changed = copy.deepcopy(ms)
        changed['calibration_validation'][field] = value
        result = _direct_corrected_pair(md, changed)
        assert result['status'] == 'UNMATCHED_CALIBRATION'
        records.append({'mutation': 'MD-MS '+field, 'result': result})
    changed = copy.deepcopy(r)
    changed['training_seed'] = 123
    result = _pair(md, [changed])
    assert result['status'] == 'UNMATCHED' and result['paired_delta'] is None
    records.append({'mutation': 'only control has different seed', 'result': result})
    original = root / fixture['ROLES'][1] / 'run_result.json'
    duplicate = root / 'higher_map_duplicate' / 'run_result.json'
    duplicate.parent.mkdir()
    raw = json.loads(original.read_text())
    raw['history'][-1]['full_pseudo_unseen_mAP'] = 0.99
    duplicate.write_text(json.dumps(raw))
    report = write_report(root, root / 'duplicate.md', horizon=2)
    pairs = report['campaigns'][0]['pairs'][fixture['ROLES'][1]]
    assert len(pairs) == 2
    assert all(p['status'] == 'DUPLICATE_ARM' and p['paired_delta'] is None for p in pairs)
    records.append({'mutation': 'duplicate with higher mAP', 'result': pairs})
(OUT / 'pairing_negative_checks.json').write_text(json.dumps(records, indent=2)+'\n')
print(f'{len(records)} pairing mutation checks passed')
