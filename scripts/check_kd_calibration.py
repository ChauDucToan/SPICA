"""CPU-only synthetic receipt boundary check, not real calibration evidence."""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from spica.train_student import KD_CALIBRATION_RULE, read_kd_calibration, sha256, train


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # Rejected before data, model/device access, or output creation.
        for value, joint, calibration in [(0., True, None), (-1., True, None),
                (float('nan'), True, None), (float('inf'), True, None),
                (1.2, False, None), (1.2, True, root / 'reviewed.json')]:
            args = SimpleNamespace(output=root / 'unused', lora_teacher=joint,
                                   visual_prompt=joint, kd_calibration=calibration, lambda_kd=value)
            try:
                train(args)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid fixed lambda accepted')
            assert not args.output.exists()
        artifacts = {}
        for i in range(4):
            file = root / f'batch_{i}.pt'
            file.write_bytes(f'synthetic boundary fixture {i}'.encode())
            artifacts[file.name] = sha256(file)
        valid = {'status': 'SCOPED_GRADIENT_CALIBRATION_PASS', 'rule': KD_CALIBRATION_RULE,
                 'optimizer_updates': 0, 'test_images_opened': 0,
                 'state_before': {'synthetic': 'same'}, 'state_after': {'synthetic': 'same'},
                 'batches': [{'batch': i, 'retrieval_grad_norm': 2., 'kd_grad_norm': .5,
                              'ratio_kd_to_retrieval': .25} for i in range(4)],
                 'lambda_kd': .4, 'weighted_ratios': [.1]*4, 'artifacts_sha256': artifacts}
        file = root / 'reviewed.json'
        file.write_text(json.dumps(valid))
        assert read_kd_calibration(file) == valid
        bad_fields = [('status', 'MEASURED_PENDING_REVIEW'), ('rule', 'another_rule'),
                      ('optimizer_updates', 1), ('test_images_opened', 1),
                      ('state_after', {'synthetic': 'changed'}), ('lambda_kd', 1.),
                      ('lambda_kd', float('nan')), ('lambda_kd', -1.),
                      ('batches', valid['batches'][:3]), ('weighted_ratios', [.2]*4),
                      ('artifacts_sha256', {'../outside.pt': 'x'})]
        for key, value in bad_fields:
            bad = copy.deepcopy(valid)
            bad[key] = value
            file.write_text(json.dumps(bad))
            try:
                read_kd_calibration(file)
            except ValueError:
                pass
            else:
                raise AssertionError(key)
        file.write_text(json.dumps(valid))
        (root / 'batch_0.pt').write_bytes(b'tampered')
        try:
            read_kd_calibration(file)
        except ValueError:
            pass
        else:
            raise AssertionError('artifact tampering accepted')
    print('PASS: synthetic calibration contract, lambda formula, artifact tampering, fixed-lambda boundary')


if __name__ == '__main__':
    main()
