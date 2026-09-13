"""Optional read-only W&B check; writes only this pack's online_check.json.
Uses existing W&B credentials; no remote logging, resume, state edit or artifact download.
"""
import csv
import datetime
import json
from pathlib import Path

import wandb


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    api = wandb.Api(timeout=30)
    checks = []
    with (out / 'tables/provenance.csv').open() as f:
        records = list(csv.DictReader(f))
    for row in records:
        path = row['wandb_url'].removeprefix('https://wandb.ai/').replace('/runs/', '/')
        run = api.run(path)
        local = [json.loads(s) for s in (root / row['run_path'] / 'test_metrics.jsonl').read_text().splitlines()]
        online = list(run.scan_history(keys=list(local[0])))
        assert len(online) == len(local) == 5
        assert all(all(abs(a[k] - b[k]) <= 1e-12 for k in a) for a, b in zip(local, online))
        checks.append({'dataset': row['dataset'], 'wandb_url': row['wandb_url'], 'state': run.state,
                       'history_rows': len(online), 'steps': [r['step_train'] for r in online],
                       'scalar_comparisons': 30, 'exact_within_1e_12': True})
    result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'READ_ONLY_FINAL_ONLINE_HISTORY; no artifact download, no remote writes', 'checks': checks}
    (out / 'online_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: 15 rows, 90 retrieval scalars; no remote writes')


if __name__ == '__main__':
    main()
