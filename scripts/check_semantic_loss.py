"""Small standalone CPU check for frozen prototype semantic loss."""

import json
import tempfile
from pathlib import Path

import torch
import yaml

from spica.semantic_loss import PrototypeSemanticLoss, _sha, photo_centroids


def check_cache_boundary():
    with tempfile.TemporaryDirectory(
        dir="outputs", prefix="semantic_cpu_"
    ) as directory:
        root = Path(directory).absolute()
        for split, count in (("train", 220), ("test", 30)):
            (root / f"{split}_classes.txt").write_text(
                "".join(f"{split}_class_{i} {i}\n" for i in range(count))
            )
            (root / f"{split}_photos.txt").write_text(
                "".join(f"{split}/{i}.png {i}\n" for i in range(count))
            )
        config = root / "data.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "name": "tuberlin_220_30",
                    "root": str(root),
                    **{
                        split: {
                            "sketch_manifest": f"{split}_photos.txt",
                            "photo_manifest": f"{split}_photos.txt",
                            "class_map": f"{split}_classes.txt",
                        }
                        for split in ("train", "test")
                    },
                }
            )
        )
        clip = root / "mock.safetensors"
        clip.write_bytes(b"CPU boundary fixture only, never load a model")
        bank = torch.nn.functional.normalize(torch.randn(220, 512), dim=-1)
        cache = root / "semantic_prototypes.pt"
        torch.save({"labels": torch.arange(220), "text": bank, "photo": bank}, cache)
        source = Path("src/spica/semantic_loss.py")
        metadata = {
            "cache_sha256": _sha(cache),
            "data_config_sha256": _sha(config),
            "train_photo_manifest_sha256": _sha(root / "train_photos.txt"),
            "train_class_map_sha256": _sha(root / "train_classes.txt"),
            "clip_weights_sha256": _sha(clip),
            "source_sha256": _sha(source),
            "class_names": {str(i): f"train_class_{i}" for i in range(220)},
            "class_count": 220,
            "temperature": 0.07,
            "weights": {"retrieval": 1.0, "text": 1.0, "photo": 1.0},
        }
        (root / "metadata.json").write_text(json.dumps(metadata))
        assert (
            PrototypeSemanticLoss.from_cache(cache, config, clip).labels.numel() == 220
        )
        metadata["clip_weights_sha256"] = "stale"
        (root / "metadata.json").write_text(json.dumps(metadata))
        try:
            PrototypeSemanticLoss.from_cache(cache, config, clip)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale bank must fail closed")


def main():
    torch.manual_seed(4)
    labels = torch.tensor([2, 7])
    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    photos = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0], [0.0, 1.0]]), dim=-1
    )
    centroid = photo_centroids(torch.eye(2), torch.tensor([2, 2]), labels[:1])
    assert torch.allclose(centroid, torch.tensor([[2**-0.5, 2**-0.5]]))
    bank = PrototypeSemanticLoss(labels, text, photos, {"bank_sha256": "mock"})
    query = torch.tensor([[0.8, 0.6], [0.6, 0.8]], requires_grad=True)
    gallery = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    query_labels = torch.tensor([2, 7])
    gallery_labels = torch.tensor([2, 2, 7, 7])

    loss = bank(query, gallery, query_labels, gallery_labels)
    pieces = bank.last_components
    assert torch.allclose(
        loss.detach(), pieces["retrieval"] + pieces["text"] + pieces["photo"]
    )
    assert torch.allclose(
        pieces["photo"],
        (1 - (torch.nn.functional.normalize(query, dim=-1) * photos).sum(-1)).mean(),
    )
    loss.backward()
    assert (
        query.grad is not None
        and torch.isfinite(query.grad).all()
        and query.grad.norm() > 0
    )
    assert bank.text.grad is None and bank.photo.grad is None

    # Assigning another class's text to each query must increase text CE.
    flipped = PrototypeSemanticLoss(labels, text.flip(0), photos, {})
    flipped(query.detach(), gallery, query_labels, gallery_labels)
    assert flipped.last_components["text"] > pieces["text"]
    assert not any(
        name in {"query_labels", "class_ids"} for name, _ in bank.named_buffers()
    )
    check_cache_boundary()
    print("semantic loss CPU check: PASS")


if __name__ == "__main__":
    main()
