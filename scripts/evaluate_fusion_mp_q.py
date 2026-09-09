"""Fixed3600 F2_MP-Q clean+9mask check; no training or predictor at inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnose_coupled_retrieval import archived_verifier, sha, write_json
from measure_fusion_heads import read
from spica.data.coupled_training import load_protocol_data, verify_clip_cache
from spica.data.datasets import RetrievalEvalDataset
from spica.evaluation.coupled_predictive import (
    CoupledPredictiveAdapter,
    evaluate_views,
    write_probe,
)
from spica.evaluation.masked_view import _masked_loader, _transform_stats
from spica.models.clip import load_frozen_clip
from spica.models.coupled_predictive import CoupledPredictiveModel
from spica.provenance import source_snapshot
from spica.train_coupled_predictive import _state_hash

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/fusion_mp_execution_20260909T115500Z"
PREVIOUS = ROOT / "outputs/fusion_mp_head_measurements_20260909T135400Z"
METRICS = (
    "full_mAP",
    "P@200",
    "mAP@200_prefix_positive",
    "mAP@200_all_relevant",
    "mAP@200_min_relevant_k",
)


class QOnlyAdapter(CoupledPredictiveAdapter):
    """Same pooled q calculation, skipping the entire predictor branch."""

    def __call__(self, images):
        return F.normalize(
            self.model.pooled_head(self.model._context_tokens(images).mean(dim=1)),
            dim=-1,
        )


def comparison(candidate, control, labels, names):
    q = np.asarray(candidate["average_precision_per_query"], dtype=np.float64)
    mu = np.asarray(control["average_precision_per_query"], dtype=np.float64)
    delta = q - mu
    classes = {
        str(c): {
            "name": names[int(c)],
            "count": int((labels == c).sum()),
            "q_full_mAP": float(q[labels == c].mean()),
            "mu_i_full_mAP": float(mu[labels == c].mean()),
            "full_mAP_delta": float(delta[labels == c].mean()),
        }
        for c in np.unique(labels)
    }
    return {
        "q": {k: candidate[k] for k in METRICS},
        "mu_i": {k: control[k] for k in METRICS},
        "delta": {k: candidate[k] - control[k] for k in METRICS},
        "per_class": classes,
        "per_query_AP_win_loss_tie": {
            "win": int((delta > 1e-7).sum()),
            "loss": int((delta < -1e-7).sum()),
            "tie": int((np.abs(delta) <= 1e-7).sum()),
            "tolerance": 1e-7,
        },
        "improved_classes": sum(x["full_mAP_delta"] > 1e-7 for x in classes.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-self-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_check:
        from check_fusion_multipositive_cpu import make_model

        model = make_model().eval()
        x = torch.randn(3, 3, 8, 8)
        before = _state_hash(model)
        calls = []
        with torch.no_grad():
            old = model(x).q
            hook = model.predictor.register_forward_hook(lambda *_: calls.append(1))
            actual = QOnlyAdapter(model)(x)
            hook.remove()
        assert torch.equal(old, actual) and not calls and before == _state_hash(model)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "q_full_vs_predictor_bypass_exact": True,
                    "predictor_calls": 0,
                }
            )
        )
        return
    if args.output is None:
        parser.error("--output required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "status": "RUNNING",
        "step": 3600,
        "head": "q",
        "optimizer_updates": 0,
        "backward_calls": 0,
        "selection": "fixed3600 before new masked evaluation; clean q already observed",
        "main_training_head_unchanged": "mu_i",
        "official_unseen": False,
    }
    try:
        assert torch.cuda.is_available(), "CUDA unavailable; no fallback"
        runtime = read(CAMPAIGN / "runtime.json")
        assert runtime["child_pid"] is None and runtime["exit_code"] == 0
        run = CAMPAIGN / "runs/F2_MP"
        result = read(run / "run_result.json")
        config = read(run / "resolved_config.json")
        assert result["status"] == "COMPLETE" and result["step"] == 3600
        expected = next(r["sha256"] for r in result["checkpoints"] if r["step"] == 3600)
        checkpoint = run / "checkpoint_step3600.pt"
        assert sha(checkpoint) == expected
        train_index = read(CAMPAIGN / "source_snapshot/index.json")
        for row in train_index["manifest"]:
            if row["path"].startswith(("src/", "configs/")) or row["path"] in (
                "uv.lock",
                "pyproject.toml",
            ):
                assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
        source = source_snapshot(ROOT)
        for row in source["manifest"]:
            dest = output / "source_snapshot/files" / row["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / row["path"], dest)
            assert sha(dest) == row["sha256"]
        write_json(output / "source_snapshot/index.json", source)
        report.update(
            checkpoint_sha256=expected,
            training_source_sha256=train_index["sha256"],
            source_snapshot_sha256=source["sha256"],
        )
        write_json(output / "protocol_lock.json", report)
        clip = verify_clip_cache()
        protocol = load_protocol_data()
        split = protocol["split"]
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
        archived_verifier(CAMPAIGN).load_replay_state(
            model, checkpoint, "F2_MP", 3600, train_index["sha256"]
        )
        before = _state_hash(model)
        adapter = QOnlyAdapter(model, query="q")
        transform = bundle.transform
        sketches = tuple(split.validation_sketch_entries)
        photos = tuple(split.validation_photo_entries)
        root = Path(protocol["data"].root)
        mean, std = _transform_stats(transform)
        clean = DataLoader(
            RetrievalEvalDataset(sketches, transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
        )
        gallery = DataLoader(
            RetrievalEvalDataset(photos, transform),
            batch_size=256,
            shuffle=False,
            num_workers=4,
        )

        def masked(fraction, seed):
            return _masked_loader(
                sketches,
                transform,
                root=root,
                fraction=fraction,
                seed=seed,
                batch_size=256,
                num_workers=4,
                mean=mean,
                std=std,
                ink_threshold=0.9,
            )

        # Exact real-data bypass parity on one clean and one severe batch.
        parity = []
        with torch.no_grad():
            for loader in (clean, masked(0.75, 101)):
                x = next(iter(loader))["image"][:32].to("cuda")
                old = model(x).q
                new = adapter(x)
                parity.append(float((old - new).abs().max()))
                assert torch.equal(old, new)
        report["bypass_parity_max_abs"] = parity
        predictor_calls = []
        features = []
        gallery_features = []
        hook = model.predictor.register_forward_hook(
            lambda *_: predictor_calls.append(1)
        )
        q_hook = model.pooled_head.register_forward_hook(
            lambda _, inputs, output: features.append(
                F.normalize(output, dim=-1).detach().cpu()
            )
        )
        original_encode = adapter.encode_photo

        def capture_photo(images):
            z = original_encode(images)
            gallery_features.append(z.detach().cpu())
            return z

        adapter.encode_photo = capture_photo
        print("Evaluating q-only clean + nine masks at fixed3600", flush=True)
        try:
            with torch.no_grad():
                probe = evaluate_views(
                    adapter,
                    clean,
                    gallery,
                    masked,
                    query_entries=sketches,
                    device=torch.device("cuda"),
                )
        finally:
            hook.remove()
            q_hook.remove()
        assert not predictor_calls and before == _state_hash(model)
        assert all(p.grad is None for p in model.parameters())
        write_probe(output / "q_probe_step3600.json", probe)
        q_features = torch.cat(features).numpy()
        g_features = torch.cat(gallery_features).numpy()
        n = probe["query_count"]
        assert (
            q_features.shape == (10 * n, 512)
            and g_features.shape == (13999, 512)
            and n == 10963
        )
        feature_arrays = {
            "clean": q_features[:n],
            "gallery": g_features,
            "query_labels": np.asarray(probe["identities"]["query_labels"]),
            "gallery_labels": np.asarray(probe["identities"]["gallery_labels"]),
        }
        for index, row in enumerate(probe["conditions"], 1):
            feature_arrays[f"mask_{int(row['fraction'] * 100)}_{row['seed']}"] = (
                q_features[index * n : (index + 1) * n]
            )
        np.savez(output / "evaluation_features.npz", **feature_arrays)
        control_path = run / "probe_step3600.json"
        old = read(control_path)
        assert sha(control_path) == next(
            p["sha256"] for p in result["probes"] if p["step"] == 3600
        )
        assert probe["identities"] == old["identities"]
        previous = read(PREVIOUS / "summary.json")
        for k in METRICS:
            assert (
                probe["clean"][k]
                == previous["arms"]["F2_MP"]["retrieval"]["q"]["metrics"][k]
            )
        with np.load(PREVIOUS / "F2_MP/q_retrieval.npz") as raw:
            assert np.array_equal(probe["clean"]["top_indices"], raw["top_indices"])
        with np.load(PREVIOUS / "F2_MP/features.npz") as raw:
            assert np.array_equal(feature_arrays["clean"], raw["q"])
            assert np.array_equal(g_features, raw["gallery"])
        labels = np.asarray(probe["identities"]["query_labels"])
        compared = {
            "clean": comparison(probe["clean"], old["clean"], labels, protocol["names"])
        }
        for row in probe["conditions"]:
            control = next(
                r
                for r in old["conditions"]
                if r["fraction"] == row["fraction"] and r["seed"] == row["seed"]
            )
            assert (
                row["mask_metadata"] == control["mask_metadata"]
                and row["status_counts"] == control["status_counts"]
            )
            compared[f"mask_{int(row['fraction'] * 100)}_{row['seed']}"] = comparison(
                row, control, labels, protocol["names"]
            )
        report.update(
            status="COMPLETE",
            model_state_before=before,
            model_state_after=_state_hash(model),
            predictor_calls_during_full_eval=len(predictor_calls),
            clean_q_replay_all5scalars_exact=True,
            clean_q_top200_and_embeddings_exact=True,
            query_gallery_identities_equal=True,
            mask_records_equal=True,
            mask_records_count=9 * n,
            comparisons=compared,
            masked_macro={
                "q": probe["masked_macro"],
                "mu_i": old["masked_macro"],
                "delta": {
                    k: probe["masked_macro"][k] - old["masked_macro"][k]
                    for k in METRICS
                },
            },
            masked_by_fraction={
                f: {
                    "q": x,
                    "mu_i": old["masked_by_fraction"][f],
                    "delta": {
                        k: x[k] - old["masked_by_fraction"][f][k] for k in METRICS
                    },
                }
                for f, x in probe["masked_by_fraction"].items()
            },
            control_probe=str(control_path),
            control_probe_sha256=sha(control_path),
            probe_sha256=sha(output / "q_probe_step3600.json"),
            features_sha256=sha(output / "evaluation_features.npz"),
            elapsed_seconds=time.monotonic() - start,
        )
        write_json(output / "summary.json", report)
        print(
            json.dumps({"status": "COMPLETE", "masked_macro": report["masked_macro"]}),
            flush=True,
        )
    except Exception:
        report.update(status="FAIL", traceback=traceback.format_exc())
        write_json(output / "failure.json", report)
        raise


if __name__ == "__main__":
    main()
