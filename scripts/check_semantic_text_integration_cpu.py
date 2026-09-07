"""Fail-closed CPU production smoke gate for semantic S0/S1/S2.

The fixture is small, but the exercised paths are not: Hydra composition, the
production ``_run_impl`` loop, the real OpenCLIP text transformer/tokenizer,
the production checkpoint writer, the production W&B wrapper, and the
standalone nine-condition evaluator all remain in the path.  Only the CLIP
weights and dataset are replaced by an explicit CPU fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from hydra import compose, initialize_config_dir
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
ARMS = ("semantic_text_S0", "semantic_text_S1", "semantic_text_S2")
CAMPAIGN = "frozen_prompt_semantic_text_step1_2026-09-07"
SELECTIONS = ("latest", "best_clean", "best_masked")
MASK_FRACTIONS = (0.25, 0.5, 0.75)
MASK_SEEDS = (101, 202, 303)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_pairing_pilot_integration_cpu import (  # noqa: E402
    _fixture,
    _sha256,
    _tiny_clip,
)
from spica.evaluation import masked_view as evaluator  # noqa: E402
from spica.evaluation.text_bank import SoftPromptTextBank  # noqa: E402
import spica.tracking.wandb as wandb_tracking  # noqa: E402
import spica.train_frozen_prompt as trainer  # noqa: E402


class _FakeArtifact:
    def __init__(self, name: str, type: str, **_: object) -> None:
        self.name, self.type = name, type
        self.paths: list[str] = []

    def add_dir(self, path: str) -> None:
        self.paths.append(path)

    def add_file(self, path: str) -> None:
        self.paths.append(path)


class _FakeRun:
    id = "semantic-cpu-smoke"
    url = "https://wandb.invalid/semantic-cpu-smoke"

    def __init__(self) -> None:
        self.mode: object = None
        self.metrics: list[dict[str, object]] = []
        self.defined: list[str] = []
        self.artifacts: list[dict[str, object]] = []
        self.summary: dict[str, object] = {}
        self.finished: list[int] = []
        self._last_step_train: int | None = None

    def define_metric(self, name: str, **_: object) -> None:
        self.defined.append(name)

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        if "step_train" not in values:
            raise AssertionError("fake W&B log omitted step_train")
        step_train = int(values["step_train"])
        if step is not None and int(step) != step_train:
            raise AssertionError("fake W&B step disagrees with step_train")
        if self._last_step_train is not None and step_train < self._last_step_train:
            raise AssertionError("fake W&B step_train regressed")
        self._last_step_train = step_train
        self.metrics.append({"step": step, **values})

    def log_artifact(self, artifact: _FakeArtifact, *, aliases: list[str]) -> None:
        self.artifacts.append({"artifact": artifact, "aliases": aliases})

    def finish(self, *, exit_code: int) -> None:
        self.finished.append(exit_code)


class _FakeWandb:
    Table = object
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.run = _FakeRun()
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_kwargs = kwargs
        self.run.mode = kwargs.get("mode")
        return self.run


def _compose_arm(
    arm: str, fixture_root: Path, output: Path, *, experiment: str | None = None
) -> object:
    if arm not in ARMS:
        raise ValueError(f"unknown semantic arm: {arm}")
    config_name = experiment or arm
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="train_frozen_prompt",
            overrides=[
                f"+experiments={config_name}",
                f"data_config={fixture_root / 'toy_data.yaml'}",
                f"experiment_manifest_path={fixture_root / (config_name + '_manifest.json')}",
                f"pairing_manifest_path={fixture_root / 'pairing_manifest.json'}",
                "allow_smoke_fixture=true",
                "synthetic_fixture=true",
                "run_kind=smoke",
                "device=cpu",
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
                # Smoke is allowed to replace the protected primary paths.
                f"hydra.run.dir={output}",
            ],
        )


def _assert_config(args: object, arm: str) -> None:
    expected_lambda = 1.0 if arm == "semantic_text_S2" else 0.0
    expected_mode = "hard" if arm == "semantic_text_S0" else "soft"
    assert str(args.experiment_role) == arm
    assert str(args.experiment_campaign) == CAMPAIGN
    assert str(args.run_kind) == "smoke"
    assert str(args.device) == "cpu"
    assert str(args.model_name) == "ViT-B-32-quickgelu"
    assert args.pretrained is None
    assert str(args.sketch_view_mode) == "full_full"
    assert str(args.text_mode) == expected_mode
    assert int(args.soft_prompt_length) == 4
    assert float(args.lambda_anchor) == expected_lambda
    assert int(args.max_steps) == 2
    assert tuple(int(value) for value in args.probe_steps) == (0, 1, 2)
    assert dict(args.mask_policy)["train_fractions"] == []
    assert (
        tuple(float(value) for value in args.mask_policy["eval_fractions"])
        == MASK_FRACTIONS
    )
    assert tuple(int(value) for value in args.mask_policy["eval_seeds"]) == MASK_SEEDS


def _state_dict(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise AssertionError(f"checkpoint is not a mapping: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise AssertionError(f"checkpoint has no model_state_dict: {path}")
    return state


def _checkpoint_payload(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise AssertionError(f"checkpoint is not a mapping: {path}")
    return payload


def _trace_rows(report: dict[str, object]) -> list[dict[str, object]]:
    trace = report.get("observation_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("path"), str):
        raise AssertionError("semantic smoke must write a production observation trace")
    path = Path(str(trace["path"]))
    if not path.is_file():
        raise AssertionError(f"missing production observation trace: {path}")
    if _sha256(path) != trace.get("sha256"):
        raise AssertionError("production observation trace SHA256 is stale")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise AssertionError("production observation trace is empty or malformed")
    return rows


def _observation_key(row: dict[str, object]) -> dict[str, object]:
    """Compare sampling identity, deliberately ignoring historical mask fields."""
    ignored = {
        "views",
        "mask_metadata_sha256",
        "mask_policy",
        "masked",
        "mask_meta",
        "global_step",
    }
    return {key: value for key, value in row.items() if key not in ignored}


def _history(report: dict[str, object]) -> list[dict[str, object]]:
    value = report.get("history")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AssertionError("run-result history is missing")
    return value


def _assert_probe_rows(report: dict[str, object], arm: str) -> None:
    history = _history(report)
    assert [int(row["training_global_step"]) for row in history] == [0, 1, 2]
    assert report["campaign"] == CAMPAIGN
    assert report["experiment_role"] == arm
    assert report["artifact_identity"]["status"] == "CPU_SMOKE"
    assert report["artifact_identity"]["actual_final_step"] == 2
    assert report["official_unseen_used_for_selection"] is False
    assert report["resolved_config"]["synthetic_fixture"] is True
    assert report["resolved_config"]["pretrained"] is None
    assert report["sketch_view_mode"] == "full_full"
    assert report["two_view_budget"] is True
    assert report["mask_policy"]["train_fractions"] == []
    assert report["mask_policy"]["train_seed"] is None

    for step, row in zip((0, 1, 2), history, strict=True):
        masked = row.get("masked_view")
        if not isinstance(masked, dict):
            raise AssertionError(f"step {step} has no masked-view probe")
        probe_path = (
            Path(str(row["checkpoint"])).parent.parent
            / f"masked_view_probe_step{step}.json"
        )
        if not probe_path.is_file():
            raise AssertionError(f"step {step} has no raw masked probe: {probe_path}")
        conditions = json.loads(probe_path.read_text(encoding="utf-8")).get(
            "conditions"
        )
        if not isinstance(conditions, list) or len(conditions) != 9:
            raise AssertionError(
                f"step {step} does not contain exactly 9 mask conditions"
            )
        observed = {(float(item["fraction"]), int(item["seed"])) for item in conditions}
        assert observed == {
            (fraction, seed) for fraction in MASK_FRACTIONS for seed in MASK_SEEDS
        }
        for item in conditions:
            for key in ("full_mAP", "P@200", "mAP@200_prefix_positive"):
                assert torch.isfinite(torch.tensor(float(item[key])))

    training = report.get("training_history")
    if not isinstance(training, list) or len(training) != 2:
        raise AssertionError("semantic smoke must record both optimizer updates")
    assert all(int(row["training_global_step"]) in (1, 2) for row in training)
    for row in training:
        for key in (
            "last_train_batch_rank_loss",
            "last_train_batch_classification_loss",
            "text_anchor_loss",
            "weighted_text_anchor_loss",
        ):
            assert torch.isfinite(torch.tensor(float(row[key])))


def _assert_text_checkpoint(payload: dict[str, object], arm: str) -> None:
    validator = getattr(evaluator, "validate_semantic_text_checkpoint", None)
    if validator is None:
        raise RuntimeError(
            "semantic evaluator worker is missing validate_semantic_text_checkpoint"
        )
    validator(payload, role=arm)
    if payload.get("model_type") != "frozen_prompt_v2":
        raise AssertionError("semantic checkpoint format is not the production format")
    if payload.get("campaign") != CAMPAIGN or payload.get("experiment_role") != arm:
        raise AssertionError("serialized checkpoint role/campaign identity is wrong")
    identity = payload.get("semantic_text_identity")
    fixed = payload.get("semantic_text_fixed_bank")
    if not isinstance(identity, dict) or not isinstance(fixed, dict):
        raise AssertionError("semantic checkpoint omitted text identity or fixed T0")
    embeddings = fixed.get("embeddings")
    if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
        raise AssertionError("serialized T0 is not a real 2-D tensor")
    if fixed.get("sha256") != identity.get("fixed_bank_sha256"):
        raise AssertionError("serialized fixed-bank identity/hash disagree")
    assert fixed["class_ids"] == identity["class_ids"]
    if arm == "semantic_text_S0":
        assert payload.get("soft_prompt_state_dict") is None
    else:
        state = payload.get("soft_prompt_state_dict")
        if not isinstance(state, dict) or not isinstance(state.get("context"), Tensor):
            raise AssertionError("soft semantic checkpoint omitted serialized context")
        assert int(state["context"].shape[0]) == 4
        assert identity["initial_context_sha256"] is not None
        assert payload.get("lambda_anchor") == (
            1.0 if arm == "semantic_text_S2" else 0.0
        )


def _assert_context_parity(report: dict[str, object], arm: str) -> None:
    history = _history(report)
    zero = _checkpoint_payload(Path(str(history[0]["checkpoint"])))
    _assert_text_checkpoint(zero, arm)
    if arm == "semantic_text_S0":
        return
    fixed = zero["semantic_text_fixed_bank"]["embeddings"]  # type: ignore[index]
    context = zero["soft_prompt_state_dict"]["context"]  # type: ignore[index]
    identity = zero["semantic_text_identity"]
    assert identity["class_order_sha256"]
    # The production report records the actual parity result; the serialized
    # context and T0 are additionally checked by executing the real text tower.
    tiny = _tiny_clip()
    names = {
        int(class_id): str(name)
        for class_id, name in zip(
            identity["class_ids"], identity["class_names"], strict=True
        )
    }
    bank = SoftPromptTextBank(
        tiny.encoder,
        tiny.tokenizer,
        names,
        prompt_length=4,
        strict_prefix=True,
    )
    with torch.no_grad():
        bank.context.copy_(context)
    learned = bank()
    assert learned.requires_grad is True
    assert learned.shape == fixed.shape
    max_error = float((learned.detach() - fixed).abs().max().item())
    assert max_error <= 1e-6, f"initial T(C) != T0: {max_error}"
    assert identity["initial_learned_bank_sha256"]


def _assert_initial_visual_parity(reports: dict[str, dict[str, object]]) -> None:
    states = {
        arm: _state_dict(Path(str(_history(report)[0]["checkpoint"])))
        for arm, report in reports.items()
    }
    for arm in ARMS[1:]:
        for name in ("sketch_prompt", "photo_prompt"):
            assert name in states["semantic_text_S0"] and name in states[arm]
            assert torch.equal(states["semantic_text_S0"][name], states[arm][name]), (
                f"{name} initialization differs for {arm}"
            )
    soft_states = {
        arm: _checkpoint_payload(
            Path(str(_history(reports[arm])[0]["checkpoint"]))
        ).get("soft_prompt_state_dict")
        for arm in ARMS[1:]
    }
    assert isinstance(soft_states["semantic_text_S1"], dict)
    assert isinstance(soft_states["semantic_text_S2"], dict)
    assert torch.equal(
        soft_states["semantic_text_S1"]["context"],
        soft_states["semantic_text_S2"]["context"],
    )


def _assert_frozen_text(
    reports: dict[str, dict[str, object]],
    before: dict[str, dict[str, Tensor]],
    after: dict[str, dict[str, Tensor]],
) -> None:
    for arm in ARMS:
        assert before[arm].keys() == after[arm].keys()
        assert all(
            torch.equal(before[arm][name], after[arm][name]) for name in before[arm]
        )
        assert reports[arm]["clip_freeze_policy"]["text_tower_frozen"] is True
        for payload in (
            _checkpoint_payload(Path(str(_history(reports[arm])[0]["checkpoint"]))),
        ):
            assert payload["clip_freeze_policy"]["text_tower_frozen"] is True
    for arm in ARMS[1:]:
        gradients = reports[arm]["gradient_validation"][
            "last_update_gradient_norms_by_parameter"
        ]
        assert float(gradients["soft_prompt.context"]) > 0.0


def _run_arm(
    arm: str, fixture: object, output: Path
) -> tuple[
    dict[str, object],
    _FakeRun,
    dict[str, Tensor],
    dict[str, Tensor],
    list[dict[str, object]],
]:
    args = _compose_arm(arm, fixture.root, output)
    _assert_config(args, arm)
    tiny = _tiny_clip()
    before = {
        name: value.detach().cpu().clone()
        for name, value in tiny.encoder.model.state_dict().items()
    }
    fake = _FakeWandb()
    sample_trace: list[dict[str, object]] = []
    original_getitem = trainer.MultiPositiveRetrievalTrainDataset.__getitem__

    def traced_getitem(dataset: object, index: int) -> object:
        item = original_getitem(dataset, index)
        sample_trace.append(
            {
                "sketch_path": item.get("sketch_path"),
                "label": item.get("label"),
                "negative_label": item.get("negative_label"),
                "negative_photo_path": item.get("negative_photo_path"),
                "positive_photo_paths": item.get("positive_photo_paths"),
            }
        )
        return item

    output.mkdir(parents=True)
    success = False
    with (
        patch.object(trainer, "load_frozen_clip", return_value=tiny),
        patch.object(wandb_tracking.wandb, "init", side_effect=fake.init),
        patch.object(wandb_tracking.wandb, "Artifact", _FakeArtifact),
        patch.object(
            trainer.HydraConfig,
            "get",
            return_value=SimpleNamespace(
                runtime=SimpleNamespace(output_dir=str(output))
            ),
        ),
        patch.object(
            trainer,
            "_masked_training_views",
            side_effect=AssertionError("semantic smoke called the training masker"),
        ),
        patch.object(
            trainer.MultiPositiveRetrievalTrainDataset,
            "__getitem__",
            traced_getitem,
        ),
    ):
        experiment = trainer._new_wandb_experiment(args, output)
        if experiment is None:
            raise AssertionError(
                "semantic smoke did not construct the real W&B wrapper"
            )
        try:
            trainer._run_impl(args, experiment)
            success = True
        finally:
            experiment.finish(exit_code=0 if success else 1)
    report = json.loads((output / "run_result.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise AssertionError("run-result is not a JSON object")
    _assert_probe_rows(report, arm)
    (output / "gate_observations.json").write_text(
        json.dumps(sample_trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    after = {
        name: value.detach().cpu().clone()
        for name, value in tiny.encoder.model.state_dict().items()
    }
    assert fake.init_kwargs is not None and fake.init_kwargs["mode"] == "disabled"
    assert fake.run.mode == "disabled"
    assert fake.run.finished == [0]
    assert len(fake.run.metrics) >= 3
    logged_steps = [int(row["step_train"]) for row in fake.run.metrics]
    assert logged_steps == sorted(logged_steps)
    assert set(logged_steps) == {0, 1, 2}
    if len(sample_trace) != 4:
        raise AssertionError(
            f"expected four traced tiny training samples, got {len(sample_trace)}"
        )
    return report, fake.run, before, after, sample_trace


def _run_historical_c(
    fixture: object, output: Path
) -> tuple[dict[str, object], dict[str, Tensor], dict[str, Tensor], int]:
    """Run historical C separately to guard full/full update parity and routing."""
    args = _compose_arm("semantic_text_S0", fixture.root, output, experiment="masked_view_3600_C")
    tiny = _tiny_clip()
    before = {name: value.detach().cpu().clone() for name, value in tiny.encoder.model.state_dict().items()}
    fake = _FakeWandb()
    mask_calls = 0
    original_mask = trainer._masked_training_views

    def traced_mask(*values: object, **kwargs: object) -> object:
        nonlocal mask_calls
        mask_calls += 1
        return original_mask(*values, **kwargs)

    output.mkdir(parents=True)
    success = False
    with (
        patch.object(trainer, "load_frozen_clip", return_value=tiny),
        patch.object(wandb_tracking.wandb, "init", side_effect=fake.init),
        patch.object(wandb_tracking.wandb, "Artifact", _FakeArtifact),
        patch.object(trainer.HydraConfig, "get", return_value=SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output)))),
        patch.object(trainer, "_masked_training_views", side_effect=traced_mask),
    ):
        experiment = trainer._new_wandb_experiment(args, output)
        if experiment is None:
            raise AssertionError("historical C did not construct the production W&B wrapper")
        try:
            trainer._run_impl(args, experiment)
            success = True
        finally:
            experiment.finish(exit_code=0 if success else 1)
    report = json.loads((output / "run_result.json").read_text(encoding="utf-8"))
    after = {name: value.detach().cpu().clone() for name, value in tiny.encoder.model.state_dict().items()}
    assert mask_calls == 2, f"historical C masker call count changed: {mask_calls}"
    assert fake.run.finished == [0]
    return report, before, after, mask_calls


def _assert_historical_c_s0_parity(
    historical: dict[str, object], s0: dict[str, object],
    historical_before: dict[str, Tensor], historical_after: dict[str, Tensor],
    s0_before: dict[str, Tensor], s0_after: dict[str, Tensor],
    historical_rows: list[dict[str, object]], s0_rows: list[dict[str, object]],
) -> None:
    c_history = _history(historical)
    s0_history = _history(s0)
    assert [int(row["training_global_step"]) for row in c_history] == [0, 1, 2]
    assert [int(row["training_global_step"]) for row in s0_history] == [0, 1, 2]
    for row in historical.get("training_history", []):
        assert float(row["first_view_rank_loss"]) == float(row["second_view_rank_loss"])
        assert float(row["first_view_classification_loss"]) == float(row["second_view_classification_loss"])
    for index in (0, 2):
        c_state = _state_dict(Path(str(c_history[index // 2]["checkpoint"])))
        s0_state = _state_dict(Path(str(s0_history[index // 2]["checkpoint"])))
        for name in ("sketch_prompt", "photo_prompt"):
            assert torch.equal(c_state[name], s0_state[name]), f"{name} diverged at step {index}"
    for report in (historical, s0):
        assert report["clip_freeze_policy"]["text_tower_frozen"] is True
        assert report["clip_freeze_policy"]["frozen_clip_parameter_byte_identical"] is True
    assert historical_before.keys() == historical_after.keys()
    assert s0_before.keys() == s0_after.keys()
    assert all(torch.equal(historical_before[name], historical_after[name]) for name in historical_before)
    assert all(torch.equal(s0_before[name], s0_after[name]) for name in s0_before)
    assert [_observation_key(row) for row in historical_rows] == [
        _observation_key(row) for row in s0_rows
    ]
    assert not any(
        set(row) & {"views", "mask_metadata_sha256", "mask_policy", "masked", "mask_meta"}
        for row in s0_rows
    )


def _replay_selected(
    report: dict[str, object], output: Path
) -> dict[str, dict[str, object]]:
    selections = report.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(SELECTIONS):
        raise AssertionError(
            "semantic run-result has no complete latest/best selection records"
        )
    run_result = output / "run_result.json"
    replayed: dict[str, dict[str, object]] = {}
    old_query_count = evaluator.PSEUDO_QUERY_COUNT
    old_gallery_count = evaluator.PSEUDO_GALLERY_COUNT
    split = report["pseudo_split_identity"]
    evaluator.PSEUDO_QUERY_COUNT = int(split["validation_sketches"])
    evaluator.PSEUDO_GALLERY_COUNT = int(split["validation_photos"])
    try:
        for selection in SELECTIONS:
            destination = output.parent / f"{output.name}_replay_{selection}"
            tiny = _tiny_clip()
            with patch.object(evaluator, "load_frozen_clip", return_value=tiny):
                value = evaluator.evaluate_run(
                    run_result,
                    destination,
                    device="cpu",
                    selection=selection,
                    allow_smoke=True,
                )
            if value.get("status") != "COMPLETE":
                raise AssertionError(f"semantic replay did not complete: {selection}")
            if len(value.get("conditions", ())) != 9:
                raise AssertionError(
                    f"semantic replay has not got 9 conditions: {selection}"
                )
            selected = selections[selection]
            assert value["checkpoint_sha256"] == selected["checkpoint_sha256"]
            replayed[selection] = value
    finally:
        evaluator.PSEUDO_QUERY_COUNT = old_query_count
        evaluator.PSEUDO_GALLERY_COUNT = old_gallery_count
    return replayed


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
    failures: dict[str, dict[str, str]] = {}
    run_records: dict[str, _FakeRun] = {}
    clip_before: dict[str, dict[str, Tensor]] = {}
    clip_after: dict[str, dict[str, Tensor]] = {}
    sample_traces: dict[str, list[dict[str, object]]] = {}
    replays: dict[str, dict[str, dict[str, object]]] = {}
    for arm in ARMS:
        try:
            report, fake, before, after, sample_trace = _run_arm(
                arm, fixture, output_root / arm
            )
            reports[arm] = report
            run_records[arm] = fake
            clip_before[arm] = before
            clip_after[arm] = after
            sample_traces[arm] = sample_trace
            _assert_context_parity(report, arm)
            # This requires the evaluator worker's semantic serialized-state gate;
            # falling back to a fake replay would hide the production gap.
            replays[arm] = _replay_selected(report, output_root / arm)
        except Exception as error:  # noqa: BLE001 - gate must persist failure evidence
            failures[arm] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
    historical_report: dict[str, object] | None = None
    if not failures:
        try:
            _assert_initial_visual_parity(reports)
            _assert_frozen_text(reports, clip_before, clip_after)
            historical_report, historical_before, historical_after, _ = _run_historical_c(
                fixture, output_root / "historical_C"
            )
            _assert_historical_c_s0_parity(
                historical_report,
                reports["semantic_text_S0"],
                historical_before,
                historical_after,
                clip_before["semantic_text_S0"],
                clip_after["semantic_text_S0"],
                _trace_rows(historical_report),
                _trace_rows(reports["semantic_text_S0"]),
            )
            traces = {
                arm: [_observation_key(row) for row in sample_traces[arm]]
                for arm in reports
            }
            for arm in ARMS:
                assert (output_root / arm / "gate_observations.json").is_file()
            for arm in ARMS[1:]:
                assert traces[arm] == traces["semantic_text_S0"], (
                    f"observation trace differs: {arm}"
                )
            for arm in ARMS:
                for checkpoint in _history(reports[arm]):
                    _assert_text_checkpoint(
                        _checkpoint_payload(Path(str(checkpoint["checkpoint"]))), arm
                    )
        except Exception as error:  # noqa: BLE001 - preserve cross-arm failure evidence
            failures["cross_arm"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
    summary: dict[str, object] = {
        "status": "CPU_SEMANTIC_TEXT_SMOKE_PASS"
        if not failures
        else "CPU_SEMANTIC_TEXT_SMOKE_FAIL",
        "fixture_only": True,
        "campaign": CAMPAIGN,
        "arms": list(ARMS),
        "updates_per_arm": 2,
        "probe_steps": [0, 1, 2],
        "mask_conditions_per_probe": 9,
        "pretrained_downloaded": False,
        "official_unseen_used": False,
        "training_reports": {
            arm: str(output_root / arm / "run_result.json") for arm in reports
        },
        "historical_c_report": None
        if historical_report is None
        else str(output_root / "historical_C" / "run_result.json"),
        "observation_traces": {
            arm: str(output_root / arm / "gate_observations.json")
            for arm in sample_traces
        },
        "selected_replays": {
            arm: {
                name: str(
                    output_root / f"{arm}_replay_{name}" / "masked_view_evaluation.json"
                )
                for name in values
            }
            for arm, values in replays.items()
        },
        "wandb_modes": {arm: run_records[arm].mode for arm in run_records},
        "failures": failures,
    }
    (output_root / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / f"semantic_text_cpu_gate_{uuid.uuid4().hex[:12]}",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(run_gate(args.output_root, allow_smoke=args.allow_smoke))


if __name__ == "__main__":
    main()
