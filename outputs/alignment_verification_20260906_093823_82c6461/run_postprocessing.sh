#!/usr/bin/env bash
set -euo pipefail
cd /home/oslamelon/Desktop/Projects/spica
OUT=outputs/alignment_verification_20260906_093823_82c6461
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run() {
  local name=$1; shift
  set +e
  (set -x; direnv exec . env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" uv run --frozen --no-sync "$@") > "$OUT/logs/$name.log" 2>&1
  local status=$?
  set -e
  printf 'EXIT_CODE=%s\n' "$status" >> "$OUT/logs/$name.log"
  test "$status" = 0
}
run corrected_report python scripts/summarize_alignment.py "$OUT/fixed_attempt/pilot" --output "$OUT/corrected_pilot_report.md" --horizon 1800
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/alignment_verification_20260906_093823_82c6461')
r=json.loads((p/'corrected_pilot_report.json').read_text())
c=r['campaigns'][0]
assert c['matching_mode']=='corrected_v2', c['matching_mode']
assert all(run['status']=='VALID' for run in c['runs']), [(run['experiment_role'],run['status'],run['errors']) for run in c['runs']]
assert all(pair['status']=='MATCHED' for values in c['pairs'].values() for pair in values)
assert c['pairwise_comparisons'][0]['status']=='MATCHED'
raw=[json.loads((p/'fixed_attempt/pilot'/arm/'run_result.json').read_text()) for arm in ('R','MD','MS')]
vals=[next(h['val'] for h in r['history'] if h['training_global_step']==1800) for r in raw]
assert all(v['query_identity']==vals[0]['query_identity'] and v['gallery_identity']==vals[0]['gallery_identity'] for v in vals)
(p/'matching_validation.json').write_text(json.dumps({'matching_mode':c['matching_mode'],'runs':c['runs'],'pairs':c['pairs'],'MD_MS':c['pairwise_comparisons'],'query_and_gallery_identity_exact_match':True},indent=2)+'\n')
PY
for pair in MD_R MS_R MS_MD; do
  candidate=${pair%_*}; control=${pair#*_}
  run "bootstrap_$pair" python scripts/bootstrap_alignment.py --candidate "$OUT/fixed_attempt/pilot/$candidate/run_result.json" --control "$OUT/fixed_attempt/pilot/$control/run_result.json" --horizon 1800 --output "$OUT/raw_metrics/bootstrap_$pair.json" --repetitions 10000 --seed 3407
done
run offline_gradients env PYTHONPATH=. python "$OUT/offline_gradients.py"
for arm in R MD MS; do
  for step in 0 500 1800; do
    run "geometry_${arm}_$step" python scripts/diagnose_alignment_geometry.py --checkpoint "$OUT/fixed_attempt/pilot/$arm/checkpoints/alignment_step$step.pt" --data-config configs/data/sketchy_104_21.yaml --device cuda --max-per-class 16 --batch-size 256 --output "$OUT/raw_metrics/geometry_${arm}_$step.json"
  done
done
printf 'POSTPROCESSING_COMPLETED\n'
