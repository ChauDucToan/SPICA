from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import open_clip
import torch
from torch import Tensor, nn

from ..data.datasets import ImageTransform

TextTokenizer = Callable[[Sequence[str]], Tensor]


class FrozenClipEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.train(False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> Self:
        """Keep the wrapped CLIP model in evaluation mode."""
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        return self.encode_image(images)

    def encode_image(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "CLIP images must have shape [batch, channels, height, width], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] != 3:
            raise ValueError(f"CLIP expects three RGB channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError(f"CLIP expects floating-point images, got {images.dtype}")

        embeddings = self.model.encode_image(images, normalize=True)
        self._validate_embeddings(embeddings, expected_batch_size=images.shape[0])
        return embeddings

    def encode_text(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(
                "CLIP text tokens must have shape [batch, context_length], "
                f"got {tuple(tokens.shape)}"
            )
        if tokens.is_floating_point():
            raise TypeError(f"CLIP text tokens must be integer IDs, got {tokens.dtype}")

        embeddings = self.model.encode_text(tokens, normalize=True)
        self._validate_embeddings(embeddings, expected_batch_size=tokens.shape[0])
        return embeddings

    @staticmethod
    def _validate_embeddings(
        embeddings: Tensor,
        *,
        expected_batch_size: int,
    ) -> None:
        if embeddings.ndim != 2:
            raise RuntimeError(
                "OpenCLIP encoder must return [batch, embedding_dim], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] != expected_batch_size:
            raise RuntimeError(
                "OpenCLIP changed the batch size from "
                f"{expected_batch_size} to {embeddings.shape[0]}"
            )


@dataclass(frozen=True, slots=True)
class FrozenClipBundle:
    encoder: FrozenClipEncoder
    transform: ImageTransform
    tokenizer: TextTokenizer
    model_name: str
    pretrained: str | None


class TrainableSketchContextEncoder(nn.Module):
    """A CLIP-initialized image encoder used as the sketch context encoder.

    The wrapper deliberately exposes the visual tower only.  In the predictive
    JEPA path its input is an image tensor and its output is an *unnormalized*
    latent; no cached frozen sketch embedding is used.  ``mode`` controls
    which visual parameters receive gradients while preserving the original
    CLIP initialization.
    """

    MODES = {"frozen", "partial", "full"}

    def __init__(
        self,
        visual: nn.Module,
        *,
        embedding_dim: int,
        mode: str,
        unfreeze_depth: int = 0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.MODES)}, got {mode!r}"
            )
        if unfreeze_depth < 0:
            raise ValueError("unfreeze_depth must be non-negative")
        self.visual = visual
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.unfreeze_depth = unfreeze_depth
        self._trainable_modules: tuple[nn.Module, ...] = ()
        self._configure_trainability()

    def _configure_trainability(self) -> None:
        self.visual.requires_grad_(False)
        if self.mode == "frozen":
            self._trainable_modules = ()
            return
        if self.mode == "full":
            self.visual.requires_grad_(True)
            self._trainable_modules = (self.visual,)
            return

        # OpenCLIP ViT visual towers expose their transformer blocks through
        # ``visual.transformer.resblocks``.  Requiring this structure avoids a
        # silently misconfigured partial experiment on an unsupported backbone.
        blocks = getattr(getattr(self.visual, "transformer", None), "resblocks", None)
        if blocks is None:
            raise ValueError(
                "partial sketch encoder mode requires a visual transformer with "
                "a resblocks collection"
            )
        if self.unfreeze_depth <= 0 or self.unfreeze_depth > len(blocks):
            raise ValueError(
                "unfreeze_depth must be between 1 and the number of visual "
                f"transformer blocks ({len(blocks)}), got {self.unfreeze_depth}"
            )

        start = len(blocks) - self.unfreeze_depth
        trainable_modules: list[nn.Module] = []
        for block in blocks[start:]:
            block.requires_grad_(True)
            trainable_modules.append(block)

        # The projection and final normalization are part of the CLIP image
        # representation and are sensible small partial-adaptation parameters.
        for name in ("ln_post",):
            module = getattr(self.visual, name, None)
            if module is not None:
                module.requires_grad_(True)
                trainable_modules.append(module)
        projection = getattr(self.visual, "proj", None)
        if isinstance(projection, nn.Parameter):
            projection.requires_grad_(True)
        self._trainable_modules = tuple(trainable_modules)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def train(self, mode: bool = True) -> Self:
        """Keep frozen visual blocks in evaluation mode during training."""
        super().train(mode)
        if self.mode == "frozen":
            self.visual.eval()
        elif self.mode == "partial":
            # Most CLIP ViTs have no stochastic layers, but enforcing this makes
            # the frozen/partial contract robust to future backbones.
            self.visual.eval()
            for module in self._trainable_modules:
                module.train(mode)
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Sketch images must have shape [batch, channels, height, width], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] != 3:
            raise ValueError(f"Sketch encoder expects three RGB channels, got {images.shape[1]}")
        if not images.is_floating_point():
            raise TypeError(f"Sketch encoder expects floating-point images, got {images.dtype}")
        embeddings = self.visual(images)
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 2:
            raise RuntimeError(
                "The CLIP visual tower must return [batch, embedding_dim], "
                f"got {getattr(embeddings, 'shape', type(embeddings))}"
            )
        if embeddings.shape[0] != images.shape[0] or embeddings.shape[1] != self.embedding_dim:
            raise RuntimeError(
                "Unexpected sketch context shape: expected "
                f"[{images.shape[0]}, {self.embedding_dim}], got {tuple(embeddings.shape)}"
            )
        return embeddings


@dataclass(frozen=True, slots=True)
class SketchContextBundle:
    encoder: TrainableSketchContextEncoder
    transform: ImageTransform
    model_name: str
    pretrained: str | None


def load_frozen_clip(
    *,
    # OpenAI ViT-B/32 was pretrained with QuickGELU, not standard GELU.
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str | None = "openai",
    device: str | torch.device = "cpu",
    cache_dir: Path | None = None,
) -> FrozenClipBundle:
    model, _, eval_transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    tokenizer = open_clip.get_tokenizer(
        model_name,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )

    return FrozenClipBundle(
        encoder=FrozenClipEncoder(model),
        transform=eval_transform,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
    )


def load_trainable_sketch_encoder(
    *,
    model_name: str = "ViT-B-32-quickgelu",
    pretrained: str | None = "openai",
    device: str | torch.device = "cpu",
    cache_dir: Path | None = None,
    mode: str = "full",
    unfreeze_depth: int = 0,
) -> SketchContextBundle:
    """Load a CLIP-initialized, image-input sketch context encoder.

    Only the visual tower is retained, so the text tower cannot accidentally
    become part of the JEPA predictor.  The returned transform is the regular
    CLIP image transform; using it is preprocessing of the raw image, not a
    frozen sketch-feature bottleneck.
    """
    model, _, eval_transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    visual = model.visual
    embedding_dim = int(getattr(visual, "output_dim", 0))
    if embedding_dim <= 0:
        raise RuntimeError("Could not determine the CLIP visual output dimension")
    encoder = TrainableSketchContextEncoder(
        visual,
        embedding_dim=embedding_dim,
        mode=mode,
        unfreeze_depth=unfreeze_depth,
    )
    # The full CLIP object is no longer needed once its visual tower is owned by
    # the wrapper.  Deleting this reference releases the unused text tower.
    del model
    encoder.train(mode != "frozen")
    return SketchContextBundle(
        encoder=encoder,
        transform=eval_transform,
        model_name=model_name,
        pretrained=pretrained,
    )
