"""CPU integration gate for the pairing pilot A/B trainer paths.

This deliberately calls the production ``train_frozen_prompt.run`` loop.  The
only test seams are a tiny in-memory OpenCLIP-compatible model and a generated
fixture dataset; loss, optimizer, checkpoint, evaluation, and invariant code
remain production code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import open_clip
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg
from PIL import Image
from torch.utils.data import DataLoader

from spica.data.manifest import ManifestEntry
from spica.models.clip import FrozenClipBundle, FrozenClipEncoder
import spica.train_frozen_prompt as trainer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
ARM_CONFIGS = ("pairing_pilot_A", "pairing_pilot_B")
EXPERIMENT_DIR = CONFIG_DIR / "experiments"


@dataclass
class Fixture:
    root: Path
    data_config: Path
    pairing_manifest: Path
    pairing_sha256: str
    entries: tuple[ManifestEntry, ...]
    pairing_by_sketch: dict[str, str]
    mapped_photo_paths_by_label: dict[int, tuple[str, ...]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_image(image: Image.Image) -> torch.Tensor:
    image = image.resize((8, 8), Image.Resampling.NEAREST).convert("RGB")
    values = torch.from_numpy(np.asarray(image, dtype=np.float32))
    return values.permute(2, 0, 1).div(255.0)


def _fixture(root: Path) -> Fixture:
    dataset = root / "dataset"
    split = dataset / "zeroshot0"
    split.mkdir(parents=True)
    sketches: list[str] = []
    photos: list[str] = []
    entries: list[ManifestEntry] = []
    records: list[dict[str, object]] = []
    pairing_by_sketch: dict[str, str] = {}
    mapped_photo_paths_by_label: dict[int, tuple[str, ...]] = {}
    # 220 classes leave 200 pseudo-train classes when the production seed-3407
    # split reserves twenty classes.  Each class has three distinct canonical
    # sketch/photo originals, while the full eleven-photo pool keeps validation
    # at 220 gallery items (> P@200).
    for label in range(220):
        sketch_dir = dataset / f"256x256/sketch/tx_000000000000_ready/{label:02d}"
        photo_dir = dataset / f"256x256/photo/tx_000000000000_ready/{label:02d}"
        sketch_dir.mkdir(parents=True, exist_ok=True)
        photo_dir.mkdir(parents=True, exist_ok=True)
        mapped_photos: list[str] = []
        for original in range(3):
            stem = f"{label:02d}" if original == 0 else f"{label:02d}_orig_{original}"
            sketch_rel = Path(
                f"256x256/sketch/tx_000000000000_ready/{label:02d}/{stem}.png"
            )
            photo_rel = Path(
                f"256x256/photo/tx_000000000000_ready/{label:02d}/{stem}.jpg"
            )
            Image.new(
                "RGB", (8, 8), (label * 9 % 256, 40 + original * 20, 90)
            ).save(dataset / sketch_rel)
            Image.new(
                "RGB", (8, 8), (label * 9 % 256, 180, original * 10)
            ).save(dataset / photo_rel)
            sketches.append(f"{sketch_rel.as_posix()} {label}")
            photos.append(f"{photo_rel.as_posix()} {label}")
            sketch = dataset / sketch_rel
            photo = dataset / photo_rel
            records.append(
                {
                    "sketch_path": sketch_rel.as_posix(),
                    "photo_path": photo_rel.as_posix(),
                    "label": label,
                    "sketch_sha256": _sha256(sketch),
                    "photo_sha256": _sha256(photo),
                }
            )
            pairing_by_sketch[str(sketch.resolve())] = str(photo.resolve())
            mapped_photos.append(str(photo.resolve()))
            entries.append(ManifestEntry(sketch, label))
        mapped_photo_paths_by_label[label] = tuple(mapped_photos)
        for photo_index in range(3, 11):
            photo_rel = Path(
                f"256x256/photo/tx_000000000000_ready/{label:02d}/{label:02d}-{photo_index}.jpg"
            )
            Image.new("RGB", (8, 8), (label * 9 % 256, 180, photo_index * 10)).save(
                dataset / photo_rel
            )
            photos.append(f"{photo_rel.as_posix()} {label}")
    for name, lines in (
        ("sketch.txt", sketches),
        ("photo.txt", photos),
        ("class.txt", [f"class_{i} {i}" for i in range(220)]),
    ):
        (split / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    data_config = root / "toy_data.yaml"
    data_config.write_text(
        "\n".join(
            [
                "version: 1",
                "name: pairing_cpu_toy",
                f"root: {dataset.as_posix()}",
                "train:",
                "  sketch_manifest: zeroshot0/sketch.txt",
                "  photo_manifest: zeroshot0/photo.txt",
                "  class_map: zeroshot0/class.txt",
                "test:",
                "  sketch_manifest: zeroshot0/sketch.txt",
                "  photo_manifest: zeroshot0/photo.txt",
                "  class_map: zeroshot0/class.txt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pairing_manifest = root / "pairing_manifest.json"
    validation = set(random.Random(3407).sample(range(220), 20))
    pairing_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pairing_kind": "filename_convention_canonical",
                "records": [record for record in records if record["label"] not in validation],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return Fixture(
        root=root,
        data_config=data_config,
        pairing_manifest=pairing_manifest,
        pairing_sha256=_sha256(pairing_manifest),
        entries=tuple(entries),
        pairing_by_sketch=pairing_by_sketch,
        mapped_photo_paths_by_label=mapped_photo_paths_by_label,
    )


def _tiny_clip() -> FrozenClipBundle:
    torch.manual_seed(42)
    model = CLIP(
        embed_dim=6,
        vision_cfg=CLIPVisionCfg(
            layers=1, width=8, head_width=4, patch_size=4, image_size=8
        ),
        text_cfg=CLIPTextCfg(
            context_length=77, vocab_size=49408, width=8, heads=2, layers=1
        ),
    ).float().cpu()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return FrozenClipBundle(
        encoder=FrozenClipEncoder(model),
        transform=_tensor_image,
        tokenizer=tokenizer,
        model_name="tiny_cpu_fixture",
        pretrained=None,
    )


class _TraceDataset:
    def __init__(self, dataset: object, trace: list[dict[str, object]]) -> None:
        self.dataset = dataset
        self.trace = trace

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> object:
        item = self.dataset[index]  # type: ignore[index]
        self.trace.append(
            {
                "sketch_path": item.get("sketch_path"),
                "label": item.get("label"),
                "negative_label": item.get("negative_label"),
                "negative_photo_path": item.get("negative_photo_path"),
                "positive_photo_paths": item.get("positive_photo_paths"),
            }
        )
        return item


def _assert_intended_configs() -> None:
    configs = []
    for arm in ARM_CONFIGS:
        value = OmegaConf.to_container(
            OmegaConf.load(EXPERIMENT_DIR / f"{arm}.yaml"), resolve=False
        )
        assert isinstance(value, dict)
        configs.append(value)
    left, right = configs
    ignored = {"experiment_name", "experiment_role", "positive_sampling", "experiment_manifest_path", "hydra"}
    normalized = [
        {key: value for key, value in config.items() if key not in ignored}
        for config in configs
    ]
    if normalized[0] != normalized[1]:
        raise AssertionError("pairing A/B configs differ outside role, positive sampling, and output identity")
    for config in configs:
        if config.get("text_mode") != "hard" or config.get("classification_location") != "query":
            raise AssertionError("pairing fixture config lost the hard-text query contract")
        if config.get("encoder_mode") != "frozen" or config.get("encoder_unfreeze_depth") != 0:
            raise AssertionError("pairing fixture config lost frozen encoder fields")
        if config.get("train_class_scope") not in (None, "pseudo_train"):
            raise AssertionError("pairing config does not use pseudo-train scope")


def _compose(arm: str, fixture: Fixture) -> object:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        overrides = [
            f"+experiments={arm}",
            f"data_config={fixture.data_config}",
            "run_kind=smoke",
            "device=cpu",
            "pretrained=null",
            "num_workers=0",
            "pin_memory=false",
            "batch_size=2",
            "eval_batch_size=32",
            "diagnostic_num_seen=4",
            "max_steps=2",
            "probe_steps=[0,1,2]",
            "allow_short_run=false",
            "pseudo_val_num_classes=20",
            f"experiment_manifest_path={fixture.root / (arm + '_manifest.json')}",
        ]
        overrides.append(f"pairing_manifest_path={fixture.pairing_manifest}")
        return compose(config_name="train_frozen_prompt", overrides=overrides)


def _runtime(path: Path) -> object:
    return SimpleNamespace(runtime=SimpleNamespace(output_dir=str(path)))



def _assert_pairing_binding(payload: dict[str, object], fixture: Fixture) -> None:
    manifest_identity = payload.get("manifest_identity", payload.get("data_manifest_identity"))
    if not isinstance(manifest_identity, dict):
        raise AssertionError("artifact has no resolved data manifest identity")
    if manifest_identity.get("pairing_manifest_sha256") != fixture.pairing_sha256:
        raise AssertionError("artifact pairing SHA256 does not match fixture")
    if manifest_identity.get("pairing_manifest_path") != str(fixture.pairing_manifest):
        raise AssertionError("artifact pairing path is not the resolved fixture path")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, dict):
        raise AssertionError("artifact has no resolved config")
    if resolved.get("pairing_manifest_path") != str(fixture.pairing_manifest):
        raise AssertionError("resolved config pairing path is not the fixture path")
    if resolved.get("run_kind") != "smoke" or int(resolved.get("max_steps", -1)) != 2:
        raise AssertionError("artifact is not an explicit two-step smoke run")


def _run_arm(arm: str, fixture: Fixture, output: Path) -> dict[str, object]:
    args = _compose(arm, fixture)
    trace: list[dict[str, object]] = []
    text_outputs: list[torch.Tensor] = []
    tiny_clip = _tiny_clip()
    clip_before = {
        name: value.detach().cpu().clone()
        for name, value in tiny_clip.encoder.model.state_dict().items()
    }
    original_loader = trainer._loader
    original_text_bank = trainer.encode_class_text_bank

    def traced_loader(*values: object, **kwargs: object) -> DataLoader:
        loader = original_loader(*values, **kwargs)
        if not kwargs.get("train", False):
            return loader
        # DataLoader.dataset is immutable after construction; rebuild only this
        # tiny train loader with its untouched generator and production settings.
        return DataLoader(
            _TraceDataset(loader.dataset, trace),
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=loader.pin_memory,
            drop_last=loader.drop_last,
            generator=loader.generator,
            collate_fn=loader.collate_fn,
        )

    def traced_text_bank(*values: object, **kwargs: object) -> object:
        bank = original_text_bank(*values, **kwargs)
        text_outputs.append(bank.embeddings.detach().cpu().clone())
        return bank

    with (
        patch.object(trainer, "load_frozen_clip", return_value=tiny_clip),
        patch.object(trainer, "_loader", side_effect=traced_loader),
        patch.object(trainer, "encode_class_text_bank", side_effect=traced_text_bank),
        patch.object(trainer.HydraConfig, "get", return_value=_runtime(output)),
    ):
        trainer.run(args)
    report_path = output / "run_result.json"
    if not report_path.is_file():
        raise AssertionError(f"trainer did not write {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_gate_trace"] = trace
    report["_gate_text_sha256"] = [
        hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for value in text_outputs
    ]
    clip_after = {
        name: value.detach().cpu().clone()
        for name, value in tiny_clip.encoder.model.state_dict().items()
    }
    report["_gate_clip_state_before"] = {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in clip_before.items()
    }
    report["_gate_clip_state_after"] = {
        name: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
        for name, value in clip_after.items()
    }
    if clip_before.keys() != clip_after.keys() or any(
        not torch.equal(clip_before[name], clip_after[name]) for name in clip_before
    ):
        raise AssertionError("tiny CLIP state changed")
    _assert_pairing_binding(report, fixture)
    for row in report.get("history", []):
        checkpoint = Path(str(row["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        actual_checkpoint_hash = _sha256(checkpoint)
        if actual_checkpoint_hash != row.get("checkpoint_sha256"):
            raise AssertionError("history checkpoint SHA256 does not match the file")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if int(payload.get("step", -1)) != int(row["training_global_step"]):
            raise AssertionError("checkpoint step and report step disagree")
        _assert_pairing_binding(payload, fixture)
        if payload.get("campaign") != report.get("campaign"):
            raise AssertionError("checkpoint campaign is not bound to report campaign")
        if payload.get("run_kind") != report.get("run_kind"):
            raise AssertionError("checkpoint run_kind is not bound to report run_kind")
        if payload.get("resolved_config") != report.get("resolved_config"):
            raise AssertionError("checkpoint config binding differs from run config")
        checkpoint_identity = payload.get("artifact_identity", {})
        report_identity = report.get("artifact_identity", {})
        if (
            checkpoint_identity.get("campaign") != report_identity.get("campaign")
            or checkpoint_identity.get("run_kind") != report_identity.get("run_kind")
            or checkpoint_identity.get("selection_target_step")
            != report_identity.get("selection_target_step")
            or checkpoint_identity.get("actual_step")
            != int(row["training_global_step"])
        ):
            raise AssertionError("checkpoint artifact identity is inconsistent")
        if payload.get("source_snapshot_hash") != report.get("source_snapshot_hash"):
            raise AssertionError("checkpoint source snapshot differs from report")
        if payload.get("experiment_code_commit") != report.get("experiment_code_commit"):
            raise AssertionError("checkpoint code commit differs from report")
        if payload.get("provenance") != report.get("provenance"):
            raise AssertionError("checkpoint provenance differs from report")
        if payload.get("resolved_config") != report.get("resolved_config"):
            raise AssertionError("checkpoint resolved config differs from report")
        if payload.get("data_manifest_identity") != report.get("manifest_identity"):
            raise AssertionError("checkpoint data identity differs from report")
        if payload.get("source_snapshot_hash") in (None, ""):
            raise AssertionError("checkpoint has no source snapshot binding")
        report.setdefault("_gate_initial_hash", payload.get("initial_model_state_hash"))
        if int(payload.get("step", -1)) == 2:
            report["_gate_final_hash"] = payload.get("model_state_hash")
    step_payloads = {
        int(row["training_global_step"]): torch.load(
            ROOT / row["checkpoint"]
            if not Path(str(row["checkpoint"])).is_absolute()
            else Path(str(row["checkpoint"])),
            map_location="cpu",
            weights_only=True,
        )
        for row in report.get("history", [])
    }
    initial_state = step_payloads.get(0, {}).get("model_state_dict", {})
    final_state = step_payloads.get(2, {}).get("model_state_dict", {})
    prompt_changes = {}
    for name in ("sketch_prompt", "photo_prompt"):
        if name not in initial_state or name not in final_state:
            raise AssertionError(f"checkpoint is missing {name}")
        prompt_changes[name] = not torch.equal(initial_state[name], final_state[name])
        if not prompt_changes[name]:
            raise AssertionError(f"{name} did not change from step 0 to step 2")
    report["_gate_prompt_changes"] = prompt_changes
    report["_gate_initial_prompt_hashes"] = {
        name: hashlib.sha256(initial_state[name].numpy().tobytes()).hexdigest()
        for name in ("sketch_prompt", "photo_prompt")
    }
    checkpoints = report.get("checkpoints", {})
    for row in report.get("history", []):
        checkpoint_record = checkpoints.get(str(row["training_global_step"]))
        if not isinstance(checkpoint_record, dict):
            raise AssertionError("report checkpoint index is incomplete")
        if checkpoint_record.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            raise AssertionError("checkpoint index hash differs from history")
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise AssertionError("missing fixed-step selection")
    selected_path = Path(str(selection["checkpoint"]))
    if not selected_path.is_absolute():
        selected_path = ROOT / selected_path
    if _sha256(selected_path) != selection.get("checkpoint_sha256"):
        raise AssertionError("selection checkpoint SHA256 does not match the file")
    if selection.get("checkpoint_sha256") != report.get("checkpoint_sha256"):
        raise AssertionError("top-level selection hash differs from selection")
    if report.get("_gate_initial_hash") == report.get("_gate_final_hash"):
        raise AssertionError("prompt parameters did not change")
    return report


def _assert_reports(reports: list[dict[str, object]], fixture: Fixture) -> None:
    if len(reports) != 2:
        raise AssertionError("expected exactly two pairing arms")
    a, b = reports
    for report in reports:
        if report.get("artifact_identity", {}).get("status") != "CPU_SMOKE":
            raise AssertionError("CPU gate artifact is not explicitly marked CPU smoke")
        campaign = str(report.get("campaign", "")).lower()
        if "pairing" not in campaign or campaign == "frozen_prompt_probe_v2_2026-09-04":
            raise AssertionError(f"not a new pairing campaign: {report.get('campaign')}")
        names = {str(name) for name in report.get("trainable_parameter_names", [])}
        if not {"sketch_prompt", "photo_prompt"} <= names:
            raise AssertionError("both visual prompts are not trainable")
        policy = report.get("clip_freeze_policy", {})
        if not policy.get("all_clip_owned_parameters_byte_identical", False):
            raise AssertionError("CLIP-owned weights changed")
        groups = report.get("optimizer_groups", [])
        group_names = {
            str(name) for group in groups for name in group.get("parameter_names", [])
        }
        if group_names != {"sketch_prompt", "photo_prompt"}:
            raise AssertionError(
                "optimizer must contain exactly the two visual prompt parameters"
            )
        gradients = report.get("gradient_validation", {}).get("last_update_gradient_norms_by_parameter", {})
        for name in ("sketch_prompt", "photo_prompt"):
            value = gradients.get(name)
            if value is None or not torch.isfinite(torch.tensor(float(value))) or float(value) <= 0:
                raise AssertionError(f"{name} did not receive a finite nonzero gradient")
        if report.get("inference_contract", {}).get("text_required") is not False:
            raise AssertionError("query inference incorrectly requires text")
        if report.get("inference_contract", {}).get("oracle_class_required"):
            raise AssertionError("query inference incorrectly requires a true label")
        if report.get("official_unseen_used_for_selection", True):
            raise AssertionError("official unseen data entered the gate")
        selection = report.get("selection")
        if not isinstance(selection, dict) or selection.get("training_global_step") != 2:
            raise AssertionError("smoke selection did not target actual step 2")
        if report.get("_gate_initial_hash") == report.get("_gate_final_hash"):
            raise AssertionError("prompt parameters did not change")
    trace_a = a.get("_gate_trace", [])
    trace_b = b.get("_gate_trace", [])
    if [
        (row.get("sketch_path"), row.get("label"), row.get("negative_label"), row.get("negative_photo_path"))
        for row in trace_a
    ] != [
        (row.get("sketch_path"), row.get("label"), row.get("negative_label"), row.get("negative_photo_path"))
        for row in trace_b
    ]:
        raise AssertionError("A/B query, label, or negative-photo sampling is not matched")
    if not trace_a or not trace_b:
        raise AssertionError("training trace is empty")
    paired_by_sketch = fixture.pairing_by_sketch
    mapped_by_label = fixture.mapped_photo_paths_by_label
    different_a_positive = False
    for left, right in zip(trace_a, trace_b, strict=True):
        sketch = str(left["sketch_path"])
        label = int(left["label"])
        left_positive = tuple(str(value) for value in left["positive_photo_paths"])
        right_positive = tuple(str(value) for value in right["positive_photo_paths"])
        if not left_positive or not right_positive:
            raise AssertionError("A/B trace has no positive photo")
        if any(value not in mapped_by_label[label] for value in left_positive):
            raise AssertionError("A positive is outside the mapped canonical pool")
        if right_positive != (paired_by_sketch[sketch],):
            raise AssertionError("B positive is not exactly the assigned canonical pair")
        if left_positive != right_positive:
            different_a_positive = True
    if not different_a_positive:
        raise AssertionError("A never selected a mapped positive different from B")
    if a.get("_gate_text_sha256") != b.get("_gate_text_sha256"):
        raise AssertionError("hard text output is not stable across arms")
    if a.get("_gate_initial_hash") != b.get("_gate_initial_hash"):
        raise AssertionError("A/B initial prompts/backbone are not identical")
    if a.get("_gate_initial_prompt_hashes") != b.get("_gate_initial_prompt_hashes"):
        raise AssertionError("A/B initial sketch/photo prompts are not identical")
    if a.get("_gate_prompt_changes") != {"sketch_prompt": True, "photo_prompt": True}:
        raise AssertionError("A did not change both prompts separately")
    if b.get("_gate_prompt_changes") != {"sketch_prompt": True, "photo_prompt": True}:
        raise AssertionError("B did not change both prompts separately")
    if a.get("_gate_clip_state_before") != a.get("_gate_clip_state_after"):
        raise AssertionError("A tiny CLIP snapshot changed")
    if b.get("_gate_clip_state_before") != b.get("_gate_clip_state_after"):
        raise AssertionError("B tiny CLIP snapshot changed")
    for left, right in zip(a.get("history", []), b.get("history", []), strict=True):
        if left["training_global_step"] != right["training_global_step"]:
            raise AssertionError("A/B checkpoint horizons differ")
        if left["val"]["query_identity"] != right["val"]["query_identity"]:
            raise AssertionError("A/B query evaluation order differs")
        if left["val"]["gallery_identity"] != right["val"]["gallery_identity"]:
            raise AssertionError("A/B gallery evaluation order differs")


def run_gate(output_root: Path) -> int:
    missing = [
        name for name in ARM_CONFIGS if not (EXPERIMENT_DIR / f"{name}.yaml").is_file()
    ]
    if missing:
        print("PENDING: worker-2 pairing configs are absent: " + ", ".join(missing))
        return 2
    _assert_intended_configs()
    output_root.mkdir(parents=True, exist_ok=False)
    fixture = _fixture(output_root / "fixture")
    reports = []
    for arm in ARM_CONFIGS:
        reports.append(_run_arm(arm, fixture, output_root / arm))
    _assert_reports(reports, fixture)
    summary = {
        "status": "CPU_SYNTHETIC_SMOKE_PASS",
        "fixture_only": True,
        "pretrained_backbone": False,
        "official_unseen_used": False,
        "pairing_manifest_sha256": fixture.pairing_sha256,
        "arms": ARM_CONFIGS,
        "updates_per_arm": 2,
    }
    (output_root / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / f"pairing_cpu_gate_{uuid.uuid4().hex[:12]}",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    raise SystemExit(run_gate(args.output_root.resolve()))


if __name__ == "__main__":
    main()
