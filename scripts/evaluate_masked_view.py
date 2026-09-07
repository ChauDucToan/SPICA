"""Run the final fixed-step query-only masked-view evaluation."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from spica.evaluation.masked_view import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument(
        "--selection", choices=("latest", "best_clean", "best_masked"), default=None,
        help="named 3600-campaign checkpoint selection (mutually exclusive with --checkpoint-step)",
    )
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        report = evaluate_run(
            args.run_result.resolve(), output_dir,
            device=args.device, checkpoint_step=args.checkpoint_step,
            selection=args.selection, allow_smoke=args.allow_smoke,
        )
    except Exception as error:
        # Preserve a machine-readable failure without ever labelling it COMPLETE.
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
            (output_dir / "masked_view_evaluation_failed.json").write_text(
                json.dumps({"status": "FAILED", "error": str(error), "traceback": traceback.format_exc()}, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"FAILED: {error}")
        raise SystemExit(1)
    print(
        f"COMPLETE: {report['primary_mask_score_macro_full_mAP_9_conditions']:.9f} "
        f"({report['query_count']} queries, {report['gallery_count']} gallery items)"
    )


if __name__ == "__main__":
    main()
