"""Experiment composition using the actual dependency-injector framework."""

from collections.abc import Callable

from dependency_injector import containers, providers
from torch import Tensor, nn


class TrainingContainer(containers.DeclarativeContainer):
    # Factories create fresh encoders; Callable evaluates the loss on each batch.
    student = providers.Factory()
    retrieval_loss = providers.Callable()


def create_training_container(
    student_factory: Callable[[], nn.Module], retrieval_loss: Callable[..., Tensor]
) -> containers.DynamicContainer:
    container = TrainingContainer()
    container.student.set_provides(student_factory)
    container.retrieval_loss.set_provides(retrieval_loss)
    return container


def active_provider(provider: providers.Provider) -> providers.Provider:
    """Get the active override for provenance without constructing an object."""
    while provider.last_overriding is not None:
        provider = provider.last_overriding
    return provider
