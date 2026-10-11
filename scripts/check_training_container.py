"""Small CPU-only provider/loss check; no images, CUDA, training or W&B.

PYTHONPATH=src .venv/bin/python scripts/check_training_container.py
"""

from functools import partial

import torch
import torch.nn.functional as F
from dependency_injector import providers

from spica.containers import active_provider, create_training_container
from spica.train_student import multi_positive_loss


def main():
    factory = partial(torch.nn.Linear, 3, 512)
    container = create_training_container(factory, multi_positive_loss)
    model = container.student()
    assert model is not container.student()
    assert active_provider(container.student).provides is factory
    with container.student.override(
        providers.Factory(torch.nn.Linear, 3, 512, bias=False)
    ):
        assert container.student().bias is None
    assert container.student().bias is not None
    assert active_provider(container.student).provides is factory

    query = F.normalize(model(torch.randn(2, 3)), dim=-1)
    photos = F.normalize(torch.randn(4, 512), dim=-1)
    labels = torch.tensor([0, 1])
    photo_labels = torch.tensor([0, 1, 0, 1])
    values = query, photos, labels, photo_labels
    actual = container.retrieval_loss(*values)
    expected = multi_positive_loss(*values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, query)[0],
        torch.autograd.grad(expected, query)[0],
        atol=0,
        rtol=0,
    )
    print("PASS: fresh Factory, override/reset, exact CPU loss and gradient")


if __name__ == "__main__":
    main()
