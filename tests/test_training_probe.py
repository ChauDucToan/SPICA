import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from spica.evaluation import training_probe as probe_module
from spica.evaluation.training_probe import FRACTIONS, SEEDS, TrainingProbe


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Conv2d(3, 4, 1)
        self.pooled_head = nn.Linear(4, 4)
        self.predictor_calls = 0
        self.predictor = nn.Identity()

    def _context_tokens(self, images):
        value = self.visual(images).mean(dim=(-1, -2))
        return value[:, None, :].expand(-1, 2, -1)

    def encode_photo(self, images):
        return self.visual(images).mean(dim=(-1, -2))

    def forward(self, images):
        self.predictor_calls += 1
        return self.pooled_head(self._context_tokens(images).mean(dim=1))


def make_probe():
    labels = torch.arange(32).repeat_interleave(8)
    gallery = torch.zeros(256, 3, 4, 4)
    gallery[:, 0] = labels[:, None, None].float() / 32
    queries = gallery[::8].clone()
    masked = {(fraction, seed): queries.clone() for fraction in FRACTIONS for seed in SEEDS}
    return TrainingProbe(queries, gallery, torch.arange(32), labels, masked_queries=masked)


def independent(query, qlabels, gallery, glabels):
    query = query.flatten(1)
    gallery = gallery.flatten(1)
    scores = torch.nn.functional.normalize(query, dim=1) @ torch.nn.functional.normalize(gallery, dim=1).T
    ranking = torch.argsort(scores, dim=1, descending=True, stable=True)
    relevant = glabels[ranking].eq(qlabels[:, None])
    position = torch.arange(1, gallery.shape[0] + 1, dtype=torch.float32)
    precision = relevant.float().cumsum(1) / position
    top = relevant[:, :200]
    return {
        "mAP@200": float(((precision[:, :200] * top).sum(1) / top.sum(1).clamp_min(1)).mean()),
        "mAP@all": float(((precision * relevant).sum(1) / relevant.sum(1)).mean()),
        "P@200": float(top.float().mean()),
    }


def test_metrics_are_three_keys_and_mask_macro_uses_nine_conditions():
    probe = make_probe()
    model = TinyModel()
    result = probe(model, "cpu")
    assert set(result["cleaned"]) == {"mAP@200", "mAP@all", "P@200"}
    assert set(result["masked"]) == {"mAP@200", "mAP@all", "P@200"}
    expected = independent(
        model.pooled_head(model._context_tokens(probe.query_images).mean(dim=1)), probe.query_labels,
        model.encode_photo(probe.gallery_images), probe.gallery_labels,
    )
    assert result["cleaned"] == pytest.approx(expected)
    assert result["masked"] == pytest.approx(expected)
    assert result["summary"]["P@all"]["clean"] == pytest.approx(8 / 256)
    assert result["summary"]["P@all"]["masked"] == pytest.approx(8 / 256)


def test_probe_preserves_rng_modes_grads_and_bypasses_predictor(monkeypatch):
    probe = make_probe()
    model = TinyModel()
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before_grads = [parameter.grad.clone() for parameter in model.parameters()]
    py = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    monkeypatch.setattr(probe_module, "_load_rgb_image", lambda _: (_ for _ in ()).throw(AssertionError("opened on call")))
    probe(model, "cpu")
    assert model.training
    assert model.predictor_calls == 0
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    for parameter, before in zip(model.parameters(), before_grads):
        assert torch.equal(parameter.grad, before)


def test_from_protocol_selects_sorted_train_entries_and_masks_normalized_images(monkeypatch, tmp_path):
    entries = []
    for label in range(32):
        entries.append(SimpleNamespace(path=Path(f"z/{label}/sketch.png"), label=label))
        entries.append(SimpleNamespace(path=Path(f"a/{label}/sketch.png"), label=label))
    photos = [SimpleNamespace(path=Path(f"photos/{label}/z{i}.png"), label=label)
              for label in range(32) for i in range(8)]
    photos += [SimpleNamespace(path=Path(f"photos/{label}/a.png"), label=label)
               for label in range(32)]
    image = Image.new("RGB", (8, 8), "white")
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), (0, 0, 0))
    monkeypatch.setattr(probe_module, "_load_rgb_image", lambda _: image)
    def transform(image):
        return (torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255 - .5) / .25
    transform.transforms = [SimpleNamespace(mean=(.5, .5, .5), std=(.25, .25, .25))]
    protocol = SimpleNamespace(root=tmp_path, train=SimpleNamespace(
        sketch_entries=entries, photo_entries=photos))
    probe = TrainingProbe.from_protocol(protocol, transform)
    assert probe.query_ids[0] == "a/0/sketch.png"
    assert probe.gallery_ids[0] == "photos/0/a.png"
    assert probe.gallery_ids[1] == "photos/0/z0.png"
    assert not torch.equal(probe.query_images, probe.masked_queries[(.75, 101)])


def test_constructor_rejects_wrong_fixed_scope():
    with pytest.raises(ValueError, match="exactly 32"):
        TrainingProbe(torch.zeros(1, 3, 2, 2), torch.zeros(8, 3, 2, 2), [0], [0] * 8)
