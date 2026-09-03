import torch
import torch.nn.functional as F
from torch import nn

from spica.data.manifest import ManifestEntry
from spica.data.splits import make_classwise_retrieval_split
from spica.evaluation.embeddings import EncodedRetrievalSet
from spica.evaluation.jepa import (
    JepaFeatureSet,
    feature_geometry,
    photo_target_alignment_diagnostics,
    semantic_query_diagnostics,
)
from spica.models.clip import TrainableSketchContextEncoder
from spica.models.jepa import (
    SignatureRegularizer,
    SketchPhotoJepa,
    SpicaJepaPredictor,
    classification_accuracy,
    jepa_prediction_loss,
    jepa_ranking_loss,
    jepa_text_classification_loss,
    photo_semantic_target,
    vicreg_latent_regularization,
)


class _DummyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])


class _DummyVisual(nn.Module):
    output_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _DummyTransformer()
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = images.mean(dim=(2, 3))
        values = torch.cat((values, values[:, :1]), dim=-1)
        for block in self.transformer.resblocks:
            values = block(values)
        return self.ln_post(values) @ self.proj


def test_context_encoder_modes_control_trainable_parameters() -> None:
    frozen = TrainableSketchContextEncoder(
        _DummyVisual(), embedding_dim=4, mode="frozen"
    )
    partial = TrainableSketchContextEncoder(
        _DummyVisual(), embedding_dim=4, mode="partial", unfreeze_depth=2
    )
    full = TrainableSketchContextEncoder(_DummyVisual(), embedding_dim=4, mode="full")
    assert frozen.trainable_parameter_count == 0
    assert partial.trainable_parameter_count > 0
    assert partial.trainable_parameter_count < full.trainable_parameter_count
    assert full.trainable_parameter_count == full.total_parameter_count


def test_jepa_forward_has_sketch_only_contract_and_unit_query() -> None:
    context = TrainableSketchContextEncoder(
        _DummyVisual(), embedding_dim=4, mode="full"
    )
    model = SketchPhotoJepa(context, SpicaJepaPredictor(4, 8))
    prediction = model(torch.randn(3, 3, 8, 8))
    assert prediction.h.shape == (3, 4)
    assert prediction.u.shape == (3, 4)
    assert torch.allclose(prediction.q.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert not hasattr(prediction, "text")


def test_photo_target_is_normalized_and_detached() -> None:
    photos = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    target = photo_semantic_target(photos)
    assert target.shape == (2, 2)
    assert torch.allclose(target.norm(dim=-1), torch.ones(2))
    assert not target.requires_grad


def test_jepa_losses_and_frozen_text_bank() -> None:
    queries = F.normalize(torch.randn(4, 6), dim=-1).requires_grad_()
    targets = F.normalize(torch.randn(4, 6), dim=-1)
    negatives = F.normalize(torch.randn(4, 6), dim=-1)
    prediction = jepa_prediction_loss(queries, targets)
    ranking = jepa_ranking_loss(queries, targets, negatives)
    text = F.normalize(torch.randn(3, 6), dim=-1).requires_grad_()
    text_labels = torch.tensor([2, 7, 11])
    class_labels = torch.tensor([11, 2, 7, 2])
    classification, logits = jepa_text_classification_loss(
        queries, text, text_labels, class_labels
    )
    total = prediction + ranking + classification
    total.backward()
    assert torch.isfinite(total)
    assert text.grad is None
    assert torch.isfinite(queries.grad).all()
    assert 0 <= classification_accuracy(logits, text_labels, class_labels) <= 1

    trainable_text = text.detach().clone().requires_grad_()
    soft_loss, _ = jepa_text_classification_loss(
        queries.detach(),
        trainable_text,
        text_labels,
        class_labels,
        detach_text=False,
    )
    soft_loss.backward()
    assert torch.isfinite(soft_loss)
    assert torch.isfinite(trainable_text.grad).all()


def test_vicreg_and_sigreg_are_internal_latent_regularizers() -> None:
    latent = torch.randn(16, 8, requires_grad=True)
    vicreg = vicreg_latent_regularization(latent)
    sigreg = SignatureRegularizer(8)(latent)
    (vicreg.total + sigreg).backward()
    assert torch.isfinite(vicreg.variance)
    assert torch.isfinite(vicreg.covariance)
    assert torch.isfinite(sigreg)
    assert torch.isfinite(latent.grad).all()


def test_feature_geometry_and_semantic_diagnostics() -> None:
    torch.manual_seed(4)
    q = F.normalize(torch.randn(8, 4), dim=-1)
    features = torch.randn(8, 4)
    geometry = feature_geometry(features)
    gallery = EncodedRetrievalSet(
        embeddings=F.normalize(torch.randn(12, 4), dim=-1),
        labels=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        paths=tuple(f"p{i}" for i in range(12)),
    )
    jepa_features = JepaFeatureSet(
        h=features,
        u=features,
        q=q,
        labels=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        paths=tuple(f"q{i}" for i in range(8)),
    )
    diagnostics = semantic_query_diagnostics(jepa_features, gallery)
    target_diagnostics = photo_target_alignment_diagnostics(jepa_features, gallery)
    assert 0 < geometry.effective_rank <= 4
    assert len(geometry.singular_values) == 4
    assert torch.isfinite(
        torch.tensor(
            list(
                diagnostics.values(),
            ),
            dtype=torch.float32,
        )[0]
    )
    assert "semantic_margin" in diagnostics
    assert "individual_positive_cosine" in target_diagnostics
    assert "positive_negative_margin" in target_diagnostics


def test_classwise_split_is_deterministic_and_disjoint(tmp_path) -> None:
    classes = {i: f"class_{i}" for i in range(5)}
    sketches = tuple(
        ManifestEntry(tmp_path / f"s{i}_{c}.png", c) for c in classes for i in range(2)
    )
    photos = tuple(
        ManifestEntry(tmp_path / f"p{i}_{c}.jpg", c) for c in classes for i in range(3)
    )
    first = make_classwise_retrieval_split(
        sketches, photos, classes, num_validation_classes=2, seed=3407
    )
    second = make_classwise_retrieval_split(
        sketches, photos, classes, num_validation_classes=2, seed=3407
    )
    assert first.validation_class_ids == second.validation_class_ids
    assert set(first.train_class_ids).isdisjoint(first.validation_class_ids)
    assert all(
        entry.label in first.train_class_ids for entry in first.train_sketch_entries
    )
    assert all(
        entry.label in first.validation_class_ids
        for entry in first.validation_photo_entries
    )
