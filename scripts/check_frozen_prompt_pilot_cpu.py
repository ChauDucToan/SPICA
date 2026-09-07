"""CPU-only check of the configured hard-text/photo-prompt loss path."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from open_clip.model import VisionTransformer

from spica.models.frozen_prompt import FrozenPromptModel
from spica.models.jepa import jepa_text_classification_loss
from spica.frozen_prompt_artifacts import ensure_manifest, manifest_entry_identity
from spica.train_frozen_prompt import (
    PROJECT_ROOT, _assert_optimizer_gradients, _validate, build_optimizer,
)


def main() -> None:
    torch.set_default_device("cpu")
    torch.manual_seed(42)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(PROJECT_ROOT / "configs")
    ):
        args = compose(
            config_name="train_frozen_prompt",
            overrides=[
                "+experiments=frozen_prompt_pilot_hard_text_photo_prompt",
                "device=cpu",
            ],
        )
    _validate(args)
    with TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest, manifest_sha256 = ensure_manifest(
            manifest_path,
            dataset="toy",
            data_config=str(args.data_config),
            campaign=str(args.experiment_campaign),
        )
        identity = manifest_entry_identity(
            manifest_path,
            manifest,
            role=str(args.experiment_role),
            manifest_sha256=manifest_sha256,
        )
        assert identity["manifest_path"] == str(manifest_path)
        assert manifest["campaign"] == str(args.experiment_campaign)
    visual = VisionTransformer(
        image_size=8,
        patch_size=4,
        width=8,
        layers=2,
        heads=2,
        mlp_ratio=2,
        output_dim=6,
    )
    model = FrozenPromptModel(
        visual,
        prompt_length=int(args.visual_prompt_length),
        train_visual_layernorm=bool(args.train_visual_layernorm),
        train_sketch_prompt=bool(args.train_sketch_prompt),
        train_photo_prompt=bool(args.train_photo_prompt),
    )
    optimizer, mapping = build_optimizer(model, None, args)
    assert optimizer is not None
    names = {name for group in mapping for name in group["parameter_names"]}
    assert names == {"sketch_prompt", "photo_prompt"}
    clip_before = {
        name: value.detach().clone() for name, value in visual.named_parameters()
    }
    sketch_before = model.sketch_prompt.detach().clone()
    photo_before = model.photo_prompt.detach().clone()

    sketches = torch.randn(2, 3, 8, 8)
    positive_photos = torch.randn(2, 3, 8, 8)
    negative_photos = torch.randn(2, 3, 8, 8)
    text = torch.randn(2, 6)
    labels = torch.tensor([0, 1])
    query = model(sketches)
    positive = model.encode_photo(positive_photos)
    negative = model.encode_photo(negative_photos)
    # Match train_frozen_prompt's inline loss: unlike JEPA, photos are NOT detached.
    rank = F.softplus(
        float(args.margin)
        - (F.normalize(query, dim=-1) * F.normalize(positive, dim=-1)).sum(-1)
        + (F.normalize(query, dim=-1) * F.normalize(negative, dim=-1)).sum(-1)
    ).mean()
    classification, _ = jepa_text_classification_loss(
        query,
        text,
        torch.tensor([0, 1]),
        labels,
        temperature=float(args.tau_cls),
        detach_text=True,
    )
    loss = float(args.lambda_rank) * rank + float(args.lambda_cls) * classification
    assert rank.requires_grad and classification.requires_grad
    loss.backward()
    _assert_optimizer_gradients(model, None, mapping)
    assert model.sketch_prompt.grad is not None and model.sketch_prompt.grad.norm() > 0
    assert model.photo_prompt.grad is not None and model.photo_prompt.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(model.sketch_prompt.detach(), sketch_before)
    assert not torch.equal(model.photo_prompt.detach(), photo_before)
    assert all(
        torch.equal(value, dict(visual.named_parameters())[name].detach())
        for name, value in clip_before.items()
    )
    assert all(not value.requires_grad for value in visual.parameters())
    assert mapping[0]["parameter_names"] == ["sketch_prompt", "photo_prompt"]
    print("PASS: CPU rank+hard-CE updates both prompts; frozen visual weights unchanged")


if __name__ == "__main__":
    main()
