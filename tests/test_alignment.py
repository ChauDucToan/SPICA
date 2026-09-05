import torch
import torch.nn.functional as F
from spica.models.alignment import (
    class_conditional_alignment_loss,
    spherical_exp_map,
    spherical_log_map,
)


def test_spherical_log_exp_round_trip_and_tangent_constraint() -> None:
    anchor = F.normalize(torch.tensor([1.0, 2.0, -1.0]), dim=0)
    points = F.normalize(torch.tensor([[1.0, -1.0, 2.0], [-2.0, 1.0, 1.0]]), dim=-1)
    tangent = spherical_log_map(anchor, points)
    assert torch.allclose((tangent * anchor).sum(-1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(spherical_exp_map(anchor, tangent), points, atol=1e-5)
    assert torch.equal(spherical_log_map(anchor, anchor), torch.zeros_like(anchor))


def test_alignment_matches_equal_class_moments() -> None:
    torch.manual_seed(7)
    text = F.normalize(torch.randn(2, 8), dim=-1)
    labels = torch.tensor([3, 3, 9, 9])
    sketches = F.normalize(torch.randn(4, 8), dim=-1)
    photos = sketches.reshape(4, 1, 8).expand(-1, 3, -1).clone()
    result = class_conditional_alignment_loss(
        sketches,
        photos,
        labels,
        text_embeddings=text,
        text_labels=torch.tensor([3, 9]),
        mean_weight=1.0,
        covariance_weight=1.0,
    )
    assert result.num_classes == 2
    assert result.num_photos == 12
    assert result.total.item() < 1e-10


def test_alignment_detaches_photo_target_and_text_anchor() -> None:
    torch.manual_seed(9)
    sketches = F.normalize(torch.randn(4, 6), dim=-1).requires_grad_()
    photos = F.normalize(torch.randn(4, 2, 6), dim=-1).requires_grad_()
    text = F.normalize(torch.randn(2, 6), dim=-1).requires_grad_()
    result = class_conditional_alignment_loss(
        sketches,
        photos,
        torch.tensor([0, 0, 1, 1]),
        text_embeddings=text,
        text_labels=torch.tensor([0, 1]),
    )
    result.total.backward()
    assert sketches.grad is not None
    assert photos.grad is None
    assert text.grad is None


def test_alignment_photo_anchor_does_not_need_text() -> None:
    sketches = F.normalize(torch.eye(4), dim=-1).requires_grad_()
    photos = sketches.detach().reshape(4, 1, 4).expand(-1, 2, -1).clone()
    result = class_conditional_alignment_loss(
        sketches,
        photos,
        torch.tensor([0, 0, 1, 1]),
        anchor="photo_mean",
    )
    assert result.num_classes == 2
    assert torch.isfinite(result.total)


def test_alignment_skips_singleton_class_for_covariance() -> None:
    result = class_conditional_alignment_loss(
        torch.eye(2),
        torch.eye(2).reshape(2, 1, 2),
        torch.tensor([0, 1]),
        text_embeddings=torch.eye(2),
        text_labels=torch.tensor([0, 1]),
    )
    assert result.num_classes == 0
    assert result.total.item() == 0.0
