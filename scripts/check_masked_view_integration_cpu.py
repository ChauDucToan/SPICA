"""Fail-closed CPU integration gate for the masked-view C/M campaign.

This gate runs the production Hydra-composed trainer twice (C and M), using
only the tiny CLIP and complete fixture builders from the pairing gate.  It
never calls the pairing gate itself.  The production masked evaluator is then
run against both step-2 checkpoints with its CLIP loader replaced by the same
explicit fixture seam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiments"
ARMS = ("masked_view_C", "masked_view_M")
CAMPAIGN = "frozen_prompt_masked_view_pilot_2026-09-07"
ROLES = {
    "masked_view_C": "frozen_prompt_masked_view_C",
    "masked_view_M": "frozen_prompt_masked_view_M",
}

# The existing gate owns the tiny model and complete 220-class fixture.  Import
# the builders only; importing its run_gate would make this gate meaningless.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pairing_pilot_integration_cpu import (  # noqa: E402
    _fixture,
    _sha256,
    _tiny_clip,
)
from spica.frozen_prompt_artifacts import MASK_POLICY  # noqa: E402
import spica.train_frozen_prompt as trainer  # noqa: E402
import spica.evaluation.masked_view as masked_evaluator  # noqa: E402


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _compose_arm(arm: str, fixture_root: Path, output: Path) -> object:
    data_config = fixture_root / "toy_data.yaml"
    pairing_manifest = fixture_root / "pairing_manifest.json"
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[
                f"+experiments={arm}",
                f"data_config={data_config}",
                f"experiment_manifest_path={fixture_root / (arm + '_manifest.json')}",
                f"pairing_manifest_path={pairing_manifest}",
                "allow_smoke_fixture=true",
                "synthetic_fixture=true",
                "run_kind=smoke",
                "device=cpu",
                # The fixture is explicit and records that no pretrained
                # weights were loaded; the production trainer/evaluator paths
                # remain unchanged.
                "model_name=ViT-B-32-quickgelu",
                "pretrained=null",
                "num_workers=0",
                "pin_memory=false",
                "batch_size=2",
                "eval_batch_size=32",
                "diagnostic_num_seen=3",
                "max_steps=2",
                "probe_steps=[0,1,2]",
                "allow_short_run=false",
                "pseudo_val_num_classes=20",
                f"hydra.run.dir={output}",
            ],
        )


def _runtime(path: Path) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(runtime=SimpleNamespace(output_dir=str(path)))


def _assert_config(args: object, arm: str) -> None:
    assert str(args.experiment_role) == ROLES[arm]
    assert str(args.experiment_campaign) == CAMPAIGN
    assert str(args.sketch_view_mode) == ("full_full" if arm.endswith("C") else "full_masked")
    assert str(args.text_mode) == "hard"
    assert str(args.model_name) == "ViT-B-32-quickgelu"
    assert args.pretrained is None
    assert str(args.encoder_mode) == "frozen"
    assert int(args.encoder_unfreeze_depth) == 0
    assert int(args.max_steps) == 2
    assert tuple(int(v) for v in args.probe_steps) == (0, 1, 2)
    assert dict(args.mask_policy) == MASK_POLICY


def _run_arm(arm: str, fixture: object, output: Path) -> dict[str, object]:
    args = _compose_arm(arm, fixture.root, output)
    _assert_config(args, arm)
    _validate = trainer._validate
    # This must be the production validator, not a gate-local substitute.
    _validate(args)

    tiny = _tiny_clip()
    query_calls: list[Tensor] = []
    photo_calls: list[Tensor] = []
    original_encode = trainer.FrozenPromptModel._encode

    def traced_encode(model: object, values: Tensor, prompt: Tensor) -> Tensor:
        if torch.is_grad_enabled() and prompt is model.sketch_prompt:
            query_calls.append(values.detach().cpu().clone())
        return original_encode(model, values, prompt)  # type: ignore[arg-type]

    def traced_photo(model: object, values: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            photo_calls.append(values.detach().cpu().clone())
        return original_encode(model, values, model.photo_prompt)  # type: ignore[arg-type]

    clip_before = {
        name: value.detach().cpu().clone()
        for name, value in tiny.encoder.model.state_dict().items()
    }
    with (
        patch.object(trainer, "load_frozen_clip", return_value=tiny),
        patch.object(trainer.FrozenPromptModel, "_encode", traced_encode),
        patch.object(trainer.FrozenPromptModel, "encode_photo", traced_photo),
        patch.object(trainer.HydraConfig, "get", return_value=_runtime(output)),
    ):
        trainer.run(args)

    report = _json(output / "run_result.json")
    report["_gate_query_calls"] = query_calls
    report["_gate_photo_calls"] = photo_calls
    report["_gate_clip_before"] = clip_before
    report["_gate_clip_after"] = {
        name: value.detach().cpu().clone()
        for name, value in tiny.encoder.model.state_dict().items()
    }
    if not (output / "train_observations.jsonl").is_file():
        raise AssertionError(f"missing production observation trace: {output}")
    return report


def _trace_rows(report: dict[str, object]) -> list[dict[str, object]]:
    trace = report.get("observation_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("path"), str):
        raise AssertionError("production report has no observation trace")
    rows = [
        json.loads(line)
        for line in Path(str(trace["path"])).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) == 0:
        raise AssertionError("production observation trace is empty")
    if not all(isinstance(row, dict) for row in rows):
        raise AssertionError("observation trace contains a non-object")
    return rows


def _trace_key(row: dict[str, object]) -> dict[str, object]:
    # Deliberately exclude only role/view-mode/model-output fields.  Everything
    # describing a sampled training example and its masks must match C and M.
    return {
        key: row[key]
        for key in (
            "path", "global_step", "label", "positive_photo_path",
            "negative_photo_path", "views",
        )
    }


def _assert_training(reports: dict[str, dict[str, object]], fixture: object) -> None:
    c, m = reports["masked_view_C"], reports["masked_view_M"]
    for arm, report in reports.items():
        assert report["campaign"] == CAMPAIGN
        assert report["run_kind"] == "smoke"
        assert report["artifact_identity"]["status"] == "CPU_SMOKE"
        assert report["artifact_identity"]["actual_final_step"] == 2
        assert report["seed"] == 42 and report["pseudo_validation_seed"] == 3407
        assert report["resolved_config"]["pretrained"] is None
        assert report["resolved_config"]["data_config"] == str(fixture.data_config)
        assert report["manifest_identity"]["pairing_manifest_sha256"] == fixture.pairing_sha256
        assert report["manifest_identity"]["pairing_manifest_path"] == str(fixture.pairing_manifest)
        assert report["pairing_identity"]["sha256"] == fixture.pairing_sha256
        assert report["mask_policy"] == MASK_POLICY
        assert report["two_view_budget"] is True
        assert report["resolved_config"]["synthetic_fixture"] is True
        assert report["sketch_view_mode"] == ("full_full" if arm.endswith("C") else "full_masked")
        assert report["selection"]["training_global_step"] == 2
        assert report["selection"]["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert report["optimizer_groups"]
        active = {
            name
            for group in report["optimizer_groups"]
            if group["active"]
            for name in group["parameter_names"]
        }
        assert active == {"sketch_prompt", "photo_prompt"}
        assert set(report["trainable_parameter_names"]) == active
        assert report["clip_freeze_policy"]["all_clip_owned_parameters_byte_identical"] is True
        assert report["clip_freeze_policy"]["visual_projection_frozen"] is True
        assert report["clip_freeze_policy"]["text_tower_frozen"] is True

        history = report["history"]
        assert [row["training_global_step"] for row in history] == [0, 1, 2]
        train_history = report["training_history"]
        assert [row["training_global_step"] for row in train_history] == [1, 2]
        for row in train_history:
            for name in ("last_train_batch_rank_loss", "last_train_batch_classification_loss"):
                assert torch.isfinite(torch.tensor(float(row[name])))
            for name in ("sketch_prompt", "photo_prompt"):
                value = row["gradient_norms_by_parameter"][name]
                assert value is not None and float(value) > 0 and torch.isfinite(torch.tensor(float(value)))
            assert abs(float(row["averaged_rank_loss"]) - (float(row["first_view_rank_loss"]) + float(row["second_view_rank_loss"])) / 2) < 1e-7
            assert abs(float(row["averaged_classification_loss"]) - (float(row["first_view_classification_loss"]) + float(row["second_view_classification_loss"])) / 2) < 1e-7
        if arm.endswith("C"):
            for row in train_history:
                assert abs(float(row["first_view_rank_loss"]) - float(row["second_view_rank_loss"])) < 1e-7
                assert abs(float(row["first_view_classification_loss"]) - float(row["second_view_classification_loss"])) < 1e-7

        rows = _trace_rows(report)
        assert {int(row["step"]) for row in rows} == {1, 2}
        for row in rows:
            for field in ("path", "positive_photo_path", "negative_photo_path"):
                assert not Path(str(row[field])).is_absolute(), f"{field} is not root-relative"
            for view in row["views"]:
                assert view["policy_version"] == MASK_POLICY["version"]
                assert view["input_sha256"] != view["output_sha256"]
                assert int(view["ink_pixels_erased"]) > 0

        calls = report["_gate_query_calls"]
        photo_calls = report["_gate_photo_calls"]
        assert len(calls) == 2, f"{arm}: expected one concatenated 2B query forward per step"
        assert len(photo_calls) == 2, f"{arm}: expected one 2B photo encode per step"
        assert all(call.shape[0] == 4 for call in calls)
        assert all(call.shape[0] == 4 for call in photo_calls)
        before = report["_gate_clip_before"]
        after = report["_gate_clip_after"]
        assert before.keys() == after.keys()
        assert all(torch.equal(before[name], after[name]) for name in before)

        for history_row in history:
            checkpoint = Path(str(history_row["checkpoint"]))
            assert checkpoint.is_file()
            assert _sha256(checkpoint) == history_row["checkpoint_sha256"]
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert payload["model_type"] == "frozen_prompt_v2"
            assert payload["step"] == history_row["training_global_step"]
            assert payload["resolved_config"] == report["resolved_config"]
            assert payload["data_manifest_identity"] == report["manifest_identity"]
            assert payload["source_snapshot_hash"] == report["source_snapshot_hash"]
            assert payload["experiment_code_commit"] == report["experiment_code_commit"]
        assert report["checkpoints"]["2"]["checkpoint_sha256"] == report["history"][2]["checkpoint_sha256"]

    c_rows, m_rows = _trace_rows(c), _trace_rows(m)
    assert [_trace_key(row) for row in c_rows] == [_trace_key(row) for row in m_rows]
    def state_hash(report: dict[str, object], step: int) -> str:
        row = next(row for row in report["history"] if row["training_global_step"] == step)
        payload = torch.load(Path(str(row["checkpoint"])), map_location="cpu", weights_only=True)
        digest = hashlib.sha256()
        for name in sorted(payload["model_state_dict"]):
            digest.update(name.encode())
            digest.update(payload["model_state_dict"][name].contiguous().numpy().tobytes())
        return digest.hexdigest()

    c_initial, m_initial = state_hash(c, 0), state_hash(m, 0)
    c_final, m_final = state_hash(c, 2), state_hash(m, 2)
    assert c_initial == m_initial
    assert c_final != c_initial
    assert m_final != m_initial
    assert c["_gate_query_calls"][0].shape == m["_gate_query_calls"][0].shape
    for c_call, m_call in zip(c["_gate_query_calls"], m["_gate_query_calls"], strict=True):
        assert torch.equal(c_call[:2], m_call[:2]), "C/M first full view differs"
        assert torch.equal(c_call[:2], c_call[2:]), "C did not forward two identical full copies"
        assert not torch.equal(m_call[:2], m_call[2:]), "M masked view is a no-op"
        assert not torch.equal(c_call[2:], m_call[2:]), "M mask did not reach encoder input"


def _run_evaluator(arm: str, report: dict[str, object], output: Path) -> dict[str, object]:
    # The report path is unambiguous from its selected checkpoint.
    checkpoint = Path(str(report["selection"]["checkpoint"]))
    run_result = checkpoint.parents[1] / "run_result.json"
    tiny = _tiny_clip()
    original_query_count = masked_evaluator.PSEUDO_QUERY_COUNT
    original_gallery_count = masked_evaluator.PSEUDO_GALLERY_COUNT
    # The production evaluator's fixed-size guards are patched only to the
    # complete fixture cardinalities; arithmetic and lineage stay production.
    data = report["resolved_config"]["data_config"]
    from spica.config.data import load_data_config
    from spica.data.manifest import read_class_map, read_manifest
    from spica.data.splits import make_classwise_retrieval_split
    loaded = load_data_config(Path(str(data)))
    split = make_classwise_retrieval_split(
        read_manifest(loaded.train.sketch_manifest, loaded.root),
        read_manifest(loaded.train.photo_manifest, loaded.root),
        read_class_map(loaded.train.class_map),
        num_validation_classes=20,
        seed=3407,
    )
    masked_evaluator.PSEUDO_QUERY_COUNT = len(split.validation_sketch_entries)
    masked_evaluator.PSEUDO_GALLERY_COUNT = len(split.validation_photo_entries)
    try:
        with patch.object(masked_evaluator, "load_frozen_clip", return_value=tiny):
            result = masked_evaluator.evaluate_run(
            run_result, output, device="cpu", checkpoint_step=2, allow_smoke=True
        )
    finally:
        masked_evaluator.PSEUDO_QUERY_COUNT = original_query_count
        masked_evaluator.PSEUDO_GALLERY_COUNT = original_gallery_count
    assert result["status"] == "COMPLETE"
    assert len(result["conditions"]) == 9
    assert result["fractions"] == MASK_POLICY["eval_fractions"]
    assert result["seeds"] == MASK_POLICY["eval_seeds"]
    assert result["checkpoint_sha256"] == _sha256(Path(str(result["checkpoint"])))
    assert result["query_mask_manifest_sha256"] == _sha256(Path(str(result["query_mask_manifest"])))
    assert abs(float(result["clean_mAP_forward"]) - float(result["clean_mAP_raw_final_metric"])) < 1e-7
    assert all(torch.isfinite(torch.tensor(row["average_precision_per_query"])).all() for row in result["conditions"])
    macro = sum(float(row["full_mAP"]) for row in result["conditions"]) / 9
    assert abs(float(result["primary_mask_score_macro_full_mAP_9_conditions"]) - macro) < 1e-12
    return result


def _assert_evaluations(results: dict[str, dict[str, object]]) -> None:
    c, m = results["masked_view_C"], results["masked_view_M"]
    assert c["data_manifest_identity"] == m["data_manifest_identity"]
    assert c["gallery_paths_unchanged"] and m["gallery_paths_unchanged"]
    c_rows = [json.loads(line) for line in Path(str(c["query_mask_manifest"])).read_text().splitlines()]
    m_rows = [json.loads(line) for line in Path(str(m["query_mask_manifest"])).read_text().splitlines()]
    def identity(row: dict[str, object]) -> dict[str, object]:
        return {key: row[key] for key in ("path", "label", "fraction", "seed", "input_sha256", "output_sha256", "bbox")}
    assert [identity(row) for row in c_rows] == [identity(row) for row in m_rows]
    assert len({row["gallery_paths_unchanged"] for row in results.values()}) == 1


def run_gate(output_root: Path, *, allow_smoke: bool = False) -> int:
    if not allow_smoke:
        print("REFUSED: pass --allow-smoke to run the explicit CPU fixture gate")
        return 2
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    fixture = _fixture(output_root / "fixture")
    reports: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        reports[arm] = _run_arm(arm, fixture, output_root / arm)
    _assert_training(reports, fixture)
    evaluations = {
        arm: _run_evaluator(arm, reports[arm], output_root / f"evaluation_{arm}")
        for arm in ARMS
    }
    _assert_evaluations(evaluations)
    summary = {
        "status": "CPU_MASKED_VIEW_SMOKE_PASS",
        "fixture_only": True,
        "pretrained_config_recorded": None,
        "synthetic_fixture": True,
        "pretrained_downloaded": False,
        "official_unseen_used": False,
        "campaign": CAMPAIGN,
        "arms": [ROLES[arm] for arm in ARMS],
        "updates_per_arm": 2,
        "mask_conditions": 9,
        "pairing_manifest_sha256": fixture.pairing_sha256,
        "training_reports": {arm: str(output_root / arm / "run_result.json") for arm in ARMS},
        "evaluation_reports": {arm: str(output_root / f"evaluation_{arm}" / "masked_view_evaluation.json") for arm in ARMS},
    }
    (output_root / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / f"masked_view_cpu_gate_{uuid.uuid4().hex[:12]}")
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(run_gate(args.output_root, allow_smoke=args.allow_smoke))


if __name__ == "__main__":
    main()
