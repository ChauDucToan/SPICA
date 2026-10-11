"""Frozen seen-class CLIP semantic loss and prototype preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config.data import load_data_config
from .data.datasets import RetrievalEvalDataset
from .data.manifest import ManifestEntry, read_class_map, read_manifest
from .models.clip import FrozenClipBundle, load_frozen_clip

ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE = 0.07
WEIGHTS = {"retrieval": 1.0, "text": 1.0, "photo": 1.0}
SELECTION_RULE = (
    "sorted distinct train photo paths; random.Random(42).sample; up to 20/class"
)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def photo_centroids(features: Tensor, labels: Tensor, class_ids: Tensor) -> Tensor:
    features = F.normalize(features, dim=-1)
    labels, class_ids = labels.to(features.device), class_ids.to(features.device)
    return torch.stack(
        [F.normalize(features[labels == label].mean(0), dim=0) for label in class_ids]
    )


def prepare_cache(
    bundle: FrozenClipBundle,
    data_config: Path,
    output: Path,
    clip_sha256: str | None = None,
) -> Path:
    """Encode seen TRAIN text and selected photo prototypes once; never test data."""
    output = output.expanduser().absolute()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be under outputs/")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    data = load_data_config(data_config)
    names = read_class_map(data.train.class_map)
    unseen = read_class_map(data.test.class_map)
    if (
        data.name != "tuberlin_220_30"
        or len(names) != 220
        or len(unseen) != 30
        or set(names.values()) & set(unseen.values())
    ):
        raise ValueError("Expected disjoint TU220/30")
    output.mkdir(parents=True)
    entries = read_manifest(data.train.photo_manifest, data.root)
    if {entry.label for entry in entries} != set(names):
        raise ValueError("TRAIN photo labels differ from seen class map")
    by_label: dict[int, list[Path]] = defaultdict(list)
    for entry in entries:
        by_label[entry.label].append(entry.path)
    rng = random.Random(42)
    selections = {
        label: sorted(rng.sample(sorted(set(paths)), min(20, len(set(paths)))))
        for label, paths in sorted(by_label.items())
    }
    if any(not paths for paths in selections.values()):
        raise ValueError("Every seen class needs a TRAIN photo")

    labels = sorted(names)
    text_tokens = bundle.tokenizer(
        [f"a photo of a {names[label].replace('_', ' ')}" for label in labels]
    ).to(bundle.encoder.device)
    with torch.inference_mode():
        text = F.normalize(bundle.encoder.encode_text(text_tokens).float(), dim=-1)
        paths = [path for label in labels for path in selections[label]]
        sampled_labels = [label for label in labels for _ in selections[label]]
        dataset = RetrievalEvalDataset(
            [
                ManifestEntry(path, label)
                for path, label in zip(paths, sampled_labels, strict=True)
            ],
            bundle.transform,
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        photos = F.normalize(
            torch.cat(
                [
                    bundle.encoder.encode_image(
                        batch["image"].to(bundle.encoder.device)
                    ).float()
                    for batch in loader
                ]
            ),
            dim=-1,
        )
        photo_prototypes = photo_centroids(
            photos, torch.tensor(sampled_labels), torch.tensor(labels)
        )
    for tensor, name in (
        (text, "text prototypes"),
        (photos, "photo features"),
        (photo_prototypes, "photo prototypes"),
    ):
        if not torch.isfinite(tensor).all() or not torch.allclose(
            tensor.norm(dim=-1),
            torch.ones(tensor.shape[0], device=tensor.device),
            atol=1e-4,
        ):
            raise ValueError(f"Invalid {name}: expected finite unit vectors")
    cache = output / "semantic_prototypes.pt"
    torch.save(
        {
            "labels": torch.tensor(labels),
            "text": text.cpu(),
            "photo": photo_prototypes.cpu(),
            "photo_features": photos.cpu(),
            "photo_labels": torch.tensor(sampled_labels),
        },
        cache,
    )
    metadata = {
        "cache_sha256": _sha(cache),
        "data_config_sha256": _sha(data_config),
        "train_photo_manifest_sha256": _sha(data.train.photo_manifest),
        "train_class_map_sha256": _sha(data.train.class_map),
        "clip_model": bundle.model_name,
        "clip_pretrained": bundle.pretrained,
        "clip_weights_sha256": clip_sha256,
        "source_sha256": _sha(Path(__file__)),
        "text_prompt": "a photo of a {class_name_with_underscores_replaced}",
        "selection_rule": SELECTION_RULE,
        "seed": 42,
        "class_count": len(labels),
        "class_names": {str(label): names[label] for label in labels},
        "photo_counts": {str(label): len(selections[label]) for label in labels},
        "photo_paths": {
            str(label): [str(path) for path in selections[label]] for label in labels
        },
        "temperature": TEMPERATURE,
        "weights": WEIGHTS,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return cache


class PrototypeSemanticLoss(nn.Module):
    """Retrieval + seen-text CE + frozen photo-centroid alignment."""

    def __init__(self, labels: Tensor, text: Tensor, photo: Tensor, metadata: dict):
        super().__init__()
        order = labels.argsort()
        self.register_buffer("labels", labels[order].long().detach())
        self.register_buffer("text", text[order].float().detach())
        self.register_buffer("photo", photo[order].float().detach())
        self.experiment_metadata = metadata
        self.last_components: dict[str, Tensor] = {}

    @classmethod
    def from_cache(
        cls, cache: Path, data_config: Path, clip_weights: Path
    ) -> PrototypeSemanticLoss:
        metadata = json.loads(cache.with_name("metadata.json").read_text())
        data = load_data_config(data_config)
        names = read_class_map(data.train.class_map)
        expected = {
            "cache_sha256": _sha(cache),
            "data_config_sha256": _sha(data_config),
            "train_photo_manifest_sha256": _sha(data.train.photo_manifest),
            "train_class_map_sha256": _sha(data.train.class_map),
            "clip_weights_sha256": _sha(clip_weights),
            "source_sha256": _sha(Path(__file__)),
            "class_names": {str(label): name for label, name in names.items()},
            "class_count": 220,
            "temperature": TEMPERATURE,
            "weights": WEIGHTS,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Semantic prototype cache does not match active TRAIN/CLIP/source"
            )
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        if not torch.equal(payload["labels"], torch.tensor(sorted(names))) or any(
            payload[key].shape != (220, 512)
            or not torch.isfinite(payload[key]).all()
            or not torch.allclose(payload[key].norm(dim=-1), torch.ones(220), atol=1e-4)
            for key in ("text", "photo")
        ):
            raise ValueError("Invalid semantic bank labels or unit embeddings")
        metadata["bank_path"] = str(cache)
        return cls(payload["labels"], payload["text"], payload["photo"], metadata)

    def forward(
        self,
        query: Tensor,
        photos: Tensor,
        labels: Tensor,
        photo_labels: Tensor,
    ) -> Tensor:
        from .train_student import multi_positive_loss

        indices = torch.searchsorted(self.labels, labels)
        q = F.normalize(query, dim=-1)
        text_loss = F.cross_entropy(q @ self.text.T / TEMPERATURE, indices)
        photo_targets = self.photo[indices]
        photo_loss = (1 - (q * photo_targets).sum(-1)).mean()
        retrieval = multi_positive_loss(query, photos, labels, photo_labels)
        total = retrieval + text_loss + photo_loss
        self.last_components = {
            "retrieval": retrieval.detach(),
            "text": text_loss.detach(),
            "photo": photo_loss.detach(),
        }
        return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_frozen_clip(
        pretrained=str(args.clip_weights.expanduser().absolute()), device="cuda"
    )
    cache = prepare_cache(
        bundle, args.data_config, args.output, _sha(args.clip_weights)
    )
    print(cache)


if __name__ == "__main__":
    main()
