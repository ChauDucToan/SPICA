"""No-update clean retrieval/head-cosine measurement for finished F2 runs.

Fixed3600 comparison with R0 cached features and S0 replay. S0@best_clean1800
is additionally measured and explicitly labelled, not mixed into fixed-step rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coupled_diagnostic_metrics import feature_geometry, self_check
from diagnose_coupled_retrieval import evaluate, sha, write_json, archived_verifier
from spica.data.coupled_training import load_protocol_data, verify_clip_cache
from spica.data.datasets import RetrievalEvalDataset
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.frozen_prompt import encode_prompted_loader
from spica.models.checkpoint import load_prompt_checkpoint
from spica.models.clip import load_frozen_clip
from spica.models.coupled_predictive import CoupledPredictiveModel
from spica.models.frozen_prompt import FrozenPromptModel
from spica.provenance import source_snapshot
from spica.train_coupled_predictive import _state_hash

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/fusion_execution_20260909T064000Z"
R0_CACHE = ROOT / "outputs/coupled_retrieval_diagnostics_20260908T191840Z/R0"
S0_ROOT = (
    ROOT / "outputs/semantic_text_execution_20260908T011006Z/runs/semantic_text_S0"
)


def read(path):
    return json.loads(path.read_text())


def load_checkpoint(path):
    with torch.serialization.safe_globals(
        [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    ):
        return torch.load(path, map_location="cpu", weights_only=True)


def encode(model, entries, transform, *, semantic=False):
    arrays = defaultdict(list)
    paths, labels = [], []
    loader = DataLoader(
        RetrievalEvalDataset(entries, transform),
        batch_size=256,
        shuffle=False,
        num_workers=4,
    )
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to("cuda"))
            values = (
                {"embedding": output}
                if semantic
                else {k: getattr(output, k) for k in ("g", "q", "mu_i", "mu_t")}
            )
            for key, value in values.items():
                arrays[key].append(value.cpu().numpy())
            paths.extend(batch["path"])
            labels.extend(batch["label"].tolist())
    return (
        {k: np.concatenate(v) for k, v in arrays.items()},
        tuple(paths),
        np.asarray(labels),
    )


def measure(name, step, features, paths, labels, gallery, output):
    output.mkdir()
    canonical_root = ROOT / "datasets/Sketchy/256x256/photo/tx_000000000000_ready"
    canonical = np.asarray(
        [Path(p).parent.parent == canonical_root for p in gallery.paths]
    )
    assert (
        len(labels) == 10963 and len(gallery.paths) == 13999 and canonical.sum() == 2000
    )
    np.savez(
        output / "features.npz",
        **features,
        labels=labels,
        gallery=gallery.embeddings.numpy(),
        gallery_labels=gallery.labels.numpy(),
        canonical=canonical,
    )
    write_json(
        output / "identities.json",
        {"query_paths": paths, "gallery_paths": gallery.paths},
    )
    geometry = {key: feature_geometry(value, labels) for key, value in features.items()}
    geometry["gallery"] = feature_geometry(
        gallery.embeddings.numpy(), gallery.labels.numpy()
    )
    retrieval, cross_head, alignment = {}, {}, {}
    heads = [key for key in features if key != "g"]
    for index, head in enumerate(heads):
        retrieval[head], _ = evaluate(
            features[head],
            labels,
            paths,
            gallery,
            output / f"{head}_retrieval.npz",
            canonical,
        )
        x = F.normalize(torch.from_numpy(features[head]).double(), dim=1)
        for other in heads[index + 1 :]:
            y = F.normalize(torch.from_numpy(features[other]).double(), dim=1)
            cosine = (x * y).sum(1).numpy()
            cross_head[f"{head}__{other}"] = {
                "mean": float(cosine.mean()),
                "std": float(cosine.std()),
                "min": float(cosine.min()),
                "max": float(cosine.max()),
            }
            np.save(output / f"cosine_{head}_{other}.npy", cosine)
        # Descriptive class alignment: labels used ONLY after inference for analysis.
        p = F.normalize(gallery.embeddings.double(), dim=1)
        gl = gallery.labels.numpy()
        class_sum = {int(c): p[gl == c].sum(0) for c in np.unique(gl)}
        total = p.sum(0)
        pos, neg = [], []
        for c in np.unique(labels):
            queries = x[labels == c]
            n = int((gl == c).sum())
            pos.extend((queries @ (class_sum[int(c)] / n)).tolist())
            neg.extend(
                (queries @ ((total - class_sum[int(c)]) / (len(gl) - n))).tolist()
            )
        alignment[head] = {
            "mean_same_class_photo_cosine": float(np.mean(pos)),
            "mean_different_class_photo_cosine": float(np.mean(neg)),
            "mean_class_margin": float(np.mean(pos) - np.mean(neg)),
            "definition": "all-gallery pair mean per query then query macro; class mean NOT renormalized; labels analysis-only",
        }
    result = {
        "name": name,
        "step": step,
        "retrieval": retrieval,
        "geometry": geometry,
        "cross_head_cosine": cross_head,
        "query_gallery_cosine": alignment,
    }
    write_json(output / "measurements.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-self-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_check:
        print(json.dumps(self_check()))
        return
    if args.output is None:
        parser.error("--output required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "RUNNING",
        "scope": "clean pseudo-validation no-update head measurements",
        "optimizer_updates": 0,
        "backward_calls": 0,
        "arms": {},
    }
    start = time.monotonic()
    try:
        assert torch.cuda.is_available(), "CUDA unavailable; no fallback"
        runtime = read(CAMPAIGN / "runtime.json")
        assert (
            runtime["completed_arms"] == ["F2", "F2_SIG"]
            and runtime["current_arm"] is None
        )
        training_index = read(CAMPAIGN / "source_snapshot/index.json")
        for row in training_index["manifest"]:
            if row["path"].startswith(("src/", "configs/")) or row["path"] in (
                "uv.lock",
                "pyproject.toml",
            ):
                assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
        snapshot = source_snapshot(ROOT)
        for row in snapshot["manifest"]:
            dest = output / "source_snapshot/files" / row["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / row["path"], dest)
            assert sha(dest) == row["sha256"]
        write_json(output / "source_snapshot/index.json", snapshot)
        report["source_snapshot_sha256"] = snapshot["sha256"]
        report["training_source_snapshot_sha256"] = training_index["sha256"]
        clip = verify_clip_cache()
        report["clip_identity"] = clip
        protocol = load_protocol_data()
        split = protocol["split"]
        expected_paths = tuple(str(e.path) for e in split.validation_sketch_entries)
        expected_gallery = tuple(str(e.path) for e in split.validation_photo_entries)
        expected_labels = np.asarray([e.label for e in split.validation_sketch_entries])
        verifier = archived_verifier(CAMPAIGN)
        for arm in ("F2", "F2_SIG"):
            print(arm, "encoding all heads at3600", flush=True)
            run = CAMPAIGN / "runs" / arm
            config = read(run / "resolved_config.json")
            res = read(run / "run_result.json")
            checkpoint = run / "checkpoint_step3600.pt"
            expected_sha = next(
                x["sha256"] for x in res["checkpoints"] if x["step"] == 3600
            )
            assert sha(checkpoint) == expected_sha
            bundle = load_frozen_clip(
                model_name="ViT-B-32-quickgelu",
                pretrained=clip["path"],
                device=torch.device("cuda"),
            )
            model = (
                CoupledPredictiveModel(
                    bundle.encoder,
                    bundle.tokenizer,
                    {int(k): v for k, v in config["class_names"].items()},
                    architecture=config["architecture"],
                )
                .to("cuda")
                .eval()
            )
            verifier.load_replay_state(
                model, checkpoint, arm, 3600, res["source_snapshot_hash"]
            )
            before = _state_hash(model)
            features, paths, labels = encode(
                model, split.validation_sketch_entries, bundle.transform
            )
            gallery = encode_prompted_loader(
                model,
                DataLoader(
                    RetrievalEvalDataset(
                        split.validation_photo_entries, bundle.transform
                    ),
                    batch_size=256,
                    shuffle=False,
                    num_workers=4,
                ),
                photo=True,
            )
            assert (
                paths == expected_paths
                and gallery.paths == expected_gallery
                and np.array_equal(labels, expected_labels)
            )
            data = measure(arm, 3600, features, paths, labels, gallery, output / arm)
            old = read(run / "probe_step3600.json")["clean"]
            delta = {
                key: abs(value - old[key])
                for key, value in data["retrieval"]["mu_i"]["metrics"].items()
                if key in old
            }
            assert max(delta.values()) < 1e-7, delta
            with np.load(output / arm / "mu_i_retrieval.npz") as saved:
                assert np.array_equal(
                    saved["top_indices"], np.asarray(old["top_indices"])
                )
            assert before == _state_hash(model) and all(
                p.grad is None for p in model.parameters()
            )
            data.update(
                checkpoint_sha256=expected_sha,
                model_state_before=before,
                model_state_after=_state_hash(model),
                main_head_replay_deltas=delta,
                main_head_top_indices_exact=True,
            )
            report["arms"][arm] = data
            write_json(output / "runtime.json", report)
            del model, bundle, gallery, features, old
            gc.collect()
            torch.cuda.empty_cache()
        # Reuse already measured and independently verified R0 features, no R0 inference.
        cache_receipt = read(R0_CACHE.parent / "independent_cpu/receipt.json")
        assert cache_receipt["status"] == "PASS"
        assert (
            sha(R0_CACHE / "validation_features.npz")
            == cache_receipt["artifact_sha256"]["R0/validation_features.npz"]
        )
        with np.load(R0_CACHE / "validation_features.npz") as raw:
            features = {k: raw[k] for k in ("g", "q")}
            ids = read(R0_CACHE / "identities.json")
            paths = tuple(ids["query_paths"])
            labels = raw["labels"]
            gallery = EncodedRetrievalSet(
                torch.from_numpy(raw["gallery"]),
                torch.from_numpy(raw["gallery_labels"]),
                tuple(ids["gallery_paths"]),
            )
        assert (
            paths == expected_paths
            and gallery.paths == expected_gallery
            and np.array_equal(labels, expected_labels)
        )
        report["arms"]["R0"] = measure(
            "R0", 3600, features, paths, labels, gallery, output / "R0"
        )
        report["arms"]["R0"]["cache_source"] = str(R0_CACHE / "validation_features.npz")
        report["arms"]["R0"]["cache_sha256"] = sha(R0_CACHE / "validation_features.npz")
        del features, gallery
        semantic_result = read(S0_ROOT / "run_result.json")
        config = semantic_result["resolved_config"]
        # Frozen-prompt runtime source is unchanged relative to semantic training.
        semantic_index = read(S0_ROOT / "source_snapshot/index.json")
        recorded = {r["path"]: r["sha256"] for r in semantic_index["manifest"]}
        for path in (
            "src/spica/models/frozen_prompt.py",
            "src/spica/models/clip.py",
            "src/spica/data/transforms.py",
        ):
            assert sha(ROOT / path) == recorded[path], path
        for step in (3600, 1800):
            print(
                "S0", step, "encoding native prompted retrieval embedding", flush=True
            )
            row = semantic_result["checkpoints"][str(step)]
            checkpoint = Path(row["checkpoint"])
            assert sha(checkpoint) == row["checkpoint_sha256"]
            payload = load_checkpoint(checkpoint)
            assert (
                payload["source_snapshot_hash"]
                == semantic_result["source_snapshot_hash"]
            )
            bundle = load_frozen_clip(
                model_name=config["model_name"],
                pretrained=clip["path"],
                device=torch.device("cuda"),
            )
            model = (
                FrozenPromptModel(
                    bundle.encoder.model.visual,
                    prompt_length=config["visual_prompt_length"],
                    train_visual_layernorm=False,
                )
                .to("cuda")
                .eval()
            )
            load_prompt_checkpoint(model, payload, expected_config=config)
            before = _state_hash(model)
            features, paths, labels = encode(
                model, split.validation_sketch_entries, bundle.transform, semantic=True
            )
            gallery = encode_prompted_loader(
                model,
                DataLoader(
                    RetrievalEvalDataset(
                        split.validation_photo_entries, bundle.transform
                    ),
                    batch_size=256,
                    shuffle=False,
                    num_workers=4,
                ),
                photo=True,
            )
            assert (
                paths == expected_paths
                and gallery.paths == expected_gallery
                and np.array_equal(labels, expected_labels)
            )
            name = "S0" if step == 3600 else "S0_best_clean"
            data = measure(name, step, features, paths, labels, gallery, output / name)
            old = read(S0_ROOT / f"masked_view_probe_step{step}.json")["clean"]
            delta = {
                k: abs(v - old[k])
                for k, v in data["retrieval"]["embedding"]["metrics"].items()
                if k in old
            }
            assert max(delta.values()) < 1e-7, delta
            assert before == _state_hash(model)
            data.update(
                checkpoint_sha256=row["checkpoint_sha256"],
                model_state_before=before,
                model_state_after=_state_hash(model),
                main_head_replay_deltas=delta,
            )
            report["arms"][name] = data
            write_json(output / "runtime.json", report)
            del features, gallery, model, bundle, payload, old
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="COMPLETE", elapsed_seconds=time.monotonic() - start)
        write_json(output / "summary.json", report)
        write_json(output / "runtime.json", report)
        print(json.dumps({"status": "COMPLETE", "output": str(output)}), flush=True)
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        write_json(output / "failure.json", report)
        raise


if __name__ == "__main__":
    main()
