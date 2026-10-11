"""Standalone check for MambaSketchStem and CLIPWithSketchStem."""

import argparse
import io

import torch
import torch.nn.functional as F
from open_clip.transformer import VisionTransformer
from torch import nn

from spica.models.sketch_stem import CLIPWithSketchStem, MambaSketchStem
from spica.train_student import multi_positive_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device

    # Tiny mock ViT for CPU testing, or full ViT for CUDA
    if device == "cpu":
        visual = VisionTransformer(
            image_size=224,
            patch_size=32,
            width=64,
            layers=2,
            heads=4,
            mlp_ratio=2,
            output_dim=32,
        ).cpu().eval()
        clip_dim = 64
        output_dim = 32
    else:
        visual = VisionTransformer(
            image_size=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            mlp_ratio=4,
            output_dim=512,
        ).cuda().eval()
        clip_dim = 768
        output_dim = 512

    # 1. Test MambaSketchStem
    stem = MambaSketchStem(in_channels=192, clip_dim=clip_dim).to(device)
    sketches = torch.rand(2, 3, 224, 224, device=device)
    patches = stem(sketches)
    assert patches.shape == (2, 49, clip_dim)
    assert torch.isfinite(patches).all()

    # 2. Test CLIPWithSketchStem
    model = CLIPWithSketchStem(visual, stem_channels=192).to(device)
    query, stem_patches = model(sketches, return_patches=True)
    assert query.shape == (2, output_dim)
    assert stem_patches.shape == (2, 49, clip_dim)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2, device=device), atol=1e-5)

    targets = model.target_patch_tokens(sketches)
    assert targets.shape == (2, 49, clip_dim)
    patch_mse = F.mse_loss(stem_patches, targets)
    assert patch_mse.ndim == 0 and torch.isfinite(patch_mse)

    # 3. Test Gradient Isolation
    photos = F.normalize(torch.randn(4, output_dim, device=device), dim=-1)
    labels = torch.tensor([0, 1], device=device)
    photo_labels = torch.tensor([0, 1, 0, 2], device=device)
    loss = multi_positive_loss(query, photos, labels, photo_labels)
    loss.backward()

    # Verify model.visual has ZERO gradients
    for name, p in model.visual.named_parameters():
        assert p.grad is None, f"Frozen visual parameter received gradient: {name}"

    # Verify model.stem has valid gradients
    stem_grads = [p.grad for p in model.stem.parameters() if p.requires_grad]
    assert len(stem_grads) > 0
    assert all(g is not None and torch.isfinite(g).all() for g in stem_grads)
    assert any(g.abs().sum() > 0 for g in stem_grads)

    # 4. Test State Dict restoration
    buffer = io.BytesIO()
    torch.save(model.stem.state_dict(), buffer)
    buffer.seek(0)
    restored_stem = MambaSketchStem(in_channels=192, clip_dim=clip_dim).to(device)
    restored_stem.load_state_dict(
        torch.load(buffer, map_location=device, weights_only=True)
    )

    with torch.no_grad():
        assert torch.equal(model.stem(sketches), restored_stem(sketches))

    stem_params = sum(p.numel() for p in model.stem.parameters())
    visual_params = sum(p.numel() for p in model.visual.parameters())
    print(
        {
            "status": "PASS",
            "scope": f"CLIPWithSketchStem_{device}",
            "stem_parameters": stem_params,
            "frozen_visual_parameters": visual_params,
            "loss": loss.item(),
        }
    )


if __name__ == "__main__":
    main()
