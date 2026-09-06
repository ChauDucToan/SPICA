#!/usr/bin/env bash
set -euo pipefail
cd /home/oslamelon/Desktop/Projects/spica
OUT=outputs/alignment_verification_20260906_093823_82c6461
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COMMON=(seed=42 device=cuda experiment_manifest_path="$OUT/corrected_smoke_manifest_v2.json" run_kind=smoke max_steps=50 'probe_steps=[0,50]' log_every=100 lambda_alignment_covariance=0.0)
run() {
  local label=$1; shift
  test ! -e "$OUT/smoke/$label"
  (set -x; direnv exec . env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" uv run --frozen --no-sync spica-train-alignment "${COMMON[@]}" "$@" hydra.run.dir="$OUT/smoke/$label") > "$OUT/logs/smoke_$label.log" 2>&1
  printf 'EXIT_CODE=0\n' >> "$OUT/logs/smoke_$label.log"
}
run calibration experiments=alignment_mean_text_log lambda_alignment_mean=1.0 calibration_only=true
LAMBDA=$(.venv/bin/python -c "import json; c=json.load(open('$OUT/smoke/calibration/calibration.json')); assert c['status']=='VALID'; print(c['calibration']['lambda_alignment_mean'])")
run R experiments=alignment_control lambda_alignment_mean=0.0
run MD experiments=alignment_mean_text_log lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/smoke/calibration/calibration.json"
run MS experiments=alignment_mean_text_log_symmetric lambda_alignment_mean="$LAMBDA" alignment_calibration_artifact="$OUT/smoke/calibration/calibration.json"
printf 'SMOKE_COMPLETED\n'
