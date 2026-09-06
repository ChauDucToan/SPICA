#!/usr/bin/env bash
# Reproducer/recheck for the probe(0) CategoryRetrievalEvaluation access failure.
set -euo pipefail
cd /home/oslamelon/Desktop/Projects/spica
OUT=outputs/alignment_verification_20260906_093823_82c6461/fixed_attempt
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$OUT/logs"
run() {
  local label=$1; shift
  test ! -e "$OUT/$label"
  set +e
  (set -x; direnv exec . env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" uv run --frozen --no-sync spica-train-alignment seed=42 device=cuda lambda_alignment_covariance=0.0 log_every=100 "$@" hydra.run.dir="$OUT/$label") > "$OUT/logs/${label//\//_}.log" 2>&1
  local status=$?
  set -e
  printf 'EXIT_CODE=%s\n' "$status" >> "$OUT/logs/${label//\//_}.log"
  test "$status" = 0
}
PILOT=(experiment_manifest_path="$OUT/corrected_pilot_manifest_v2.json" run_kind=pilot max_steps=1800 'probe_steps=[0,100,500,1000,1800]')
SMOKE=(experiment_manifest_path="$OUT/corrected_smoke_manifest_v2.json" run_kind=smoke max_steps=50 'probe_steps=[0,50]')
run calibration "${PILOT[@]}" experiments=alignment_mean_text_log lambda_alignment_mean=1.0 calibration_only=true
run smoke/calibration "${SMOKE[@]}" experiments=alignment_mean_text_log lambda_alignment_mean=1.0 calibration_only=true
LAMBDA=$(.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/alignment_verification_20260906_093823_82c6461/fixed_attempt')
a=json.loads((p/'calibration/calibration.json').read_text())
b=json.loads((p/'smoke/calibration/calibration.json').read_text())
assert a['status']==b['status']=='VALID'
assert a['calibration']['lambda_alignment_mean']==b['calibration']['lambda_alignment_mean']
print(a['calibration']['lambda_alignment_mean'])
PY
)
run smoke/R "${SMOKE[@]}" experiments=alignment_control lambda_alignment_mean=0.0
run smoke/MD "${SMOKE[@]}" experiments=alignment_mean_text_log lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/smoke/calibration/calibration.json"
run smoke/MS "${SMOKE[@]}" experiments=alignment_mean_text_log_symmetric lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/smoke/calibration/calibration.json"
printf 'SMOKE_COMPLETED\n'
run pilot/R "${PILOT[@]}" experiments=alignment_control lambda_alignment_mean=0.0
run pilot/MD "${PILOT[@]}" experiments=alignment_mean_text_log lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/calibration/calibration.json"
run pilot/MS "${PILOT[@]}" experiments=alignment_mean_text_log_symmetric lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/calibration/calibration.json"
printf 'PILOT_COMPLETED\n'
