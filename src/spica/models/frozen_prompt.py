from typing import Literal, Self

import torch
import torch.nn.functional as F
from open_clip.transformer import ResidualAttentionBlock
from torch import Tensor, nn


class FrozenPromptModel(nn.Module):
    """OpenCLIP ViT with separate shallow sketch and photo prompts."""

    def __init__(
        self,
        visual: nn.Module,
        *,
        prompt_length: int,
        train_visual_layernorm: bool = False,
        train_sketch_prompt: bool = True,
        train_photo_prompt: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(prompt_length, bool) or not isinstance(prompt_length, int):
            raise TypeError("prompt_length must be an integer")
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if not isinstance(train_visual_layernorm, bool):
            raise TypeError("train_visual_layernorm must be a bool")
        if not isinstance(train_sketch_prompt, bool) or not isinstance(
            train_photo_prompt, bool
        ):
            raise TypeError("train_sketch_prompt and train_photo_prompt must be bools")
        self._validate_visual(visual)

        self.visual = visual
        self.prompt_length = prompt_length
        self.train_visual_layernorm = train_visual_layernorm
        self.train_sketch_prompt = train_sketch_prompt
        self.train_photo_prompt = train_photo_prompt
        width = int(visual.positional_embedding.shape[1])
        options = {
            "device": visual.positional_embedding.device,
            "dtype": visual.positional_embedding.dtype,
        }
        if prompt_length:
            self.sketch_prompt = nn.Parameter(
                torch.empty(prompt_length, width, **options)
            )
            self.photo_prompt = nn.Parameter(
                torch.empty(prompt_length, width, **options)
            )
            nn.init.normal_(self.sketch_prompt, std=0.02)
            nn.init.normal_(self.photo_prompt, std=0.02)
        else:
            self.register_buffer("sketch_prompt", torch.empty(0, width, **options))
            self.register_buffer("photo_prompt", torch.empty(0, width, **options))

        self.visual.requires_grad_(False)
        if train_visual_layernorm:
            for module in self.visual.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        self.sketch_prompt.requires_grad_(prompt_length > 0 and train_sketch_prompt)
        self.photo_prompt.requires_grad_(prompt_length > 0 and train_photo_prompt)
        self.train(False)

    @staticmethod
    def _validate_visual(visual: nn.Module) -> None:
        """Reject towers whose execution or token layout is not the OpenCLIP ViT one."""
        embeds = getattr(visual, "_embeds", None)
        pool = getattr(visual, "_pool", None)
        transformer = getattr(visual, "transformer", None)
        blocks = getattr(transformer, "resblocks", None)
        positional = getattr(visual, "positional_embedding", None)
        projection = getattr(visual, "proj", None)
        supported = (
            callable(embeds)
            and callable(pool)
            and isinstance(blocks, nn.ModuleList)
            and len(blocks) > 0
            and getattr(transformer, "batch_first", None) is True
            and isinstance(positional, Tensor)
            and positional.ndim == 2
            and positional.shape[0] > 1
            and positional.shape[1] > 0
            and isinstance(projection, Tensor)
            and projection.ndim == 2
            and projection.shape[0] == positional.shape[1]
            and getattr(visual, "pool_type", None) == "tok"
            and getattr(visual, "attn_pool", None) is None
            and getattr(visual, "output_tokens", None) is False
        )
        if not supported:
            raise TypeError(
                "FrozenPromptModel supports only batch-first OpenCLIP ViT visual "
                "towers with CLS pooling, a tensor projection, and no attention pool"
            )

    @property
    def device(self) -> torch.device:
        return self.sketch_prompt.device

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, value in self.named_parameters() if value.requires_grad
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @property
    def clip_parameter_names(self) -> tuple[str, ...]:
        return tuple(f"visual.{name}" for name, _ in self.visual.named_parameters())

    @property
    def visual_layernorm_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"visual.{module_name}.{parameter_name}"
            for module_name, module in self.visual.named_modules()
            if isinstance(module, nn.LayerNorm)
            for parameter_name, parameter in module.named_parameters(recurse=False)
            if parameter.requires_grad
        )

    @property
    def visual_prompt_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("sketch_prompt", "photo_prompt")
            if getattr(self, name).requires_grad
        )

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.visual.eval()
        return self

    def _prompted_tokens(self, images: Tensor, prompt: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [batch, channels, height, width], "
                f"got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError("images must be floating-point")
        tokens = self.visual._embeds(images)
        width = self.sketch_prompt.shape[1]
        if (
            not isinstance(tokens, Tensor)
            or tokens.ndim != 3
            or tokens.shape[0] != images.shape[0]
            or tokens.shape[1] < 2
            or tokens.shape[2] != width
        ):
            raise RuntimeError(
                "OpenCLIP ViT _embeds must return [batch, CLS+patches, width], "
                f"got {getattr(tokens, 'shape', type(tokens))}"
            )
        # ``_embeds`` has already applied the frozen token-wise ``ln_pre`` to
        # CLS and patches. Apply the same operation to prompts before insertion.
        expanded = (
            self.visual.ln_pre(prompt.to(dtype=tokens.dtype))
            .unsqueeze(0)
            .expand(tokens.shape[0], -1, -1)
        )
        return torch.cat((tokens[:, :1], expanded, tokens[:, 1:]), dim=1)

    def _encode(self, images: Tensor, prompt: Tensor) -> Tensor:
        tokens = self.visual.transformer(self._prompted_tokens(images, prompt))
        pooled_and_tokens = self.visual._pool(tokens)
        if not isinstance(pooled_and_tokens, tuple) or len(pooled_and_tokens) != 2:
            raise RuntimeError("OpenCLIP ViT _pool must return (pooled, tokens)")
        pooled = pooled_and_tokens[0]
        if not isinstance(pooled, Tensor) or pooled.ndim != 2:
            raise RuntimeError("OpenCLIP ViT _pool must return a [batch, width] tensor")
        return F.normalize(pooled @ self.visual.proj, dim=-1)

    def forward(self, sketch_images: Tensor) -> Tensor:
        return self._encode(sketch_images, self.sketch_prompt)

    def encode_photo(self, photo_images: Tensor) -> Tensor:
        return self._encode(photo_images, self.photo_prompt)

    @staticmethod
    def _attention_mass(weights: Tensor, rows: slice, columns: slice) -> Tensor:
        selected = weights[..., rows, columns]
        return selected.sum(dim=-1).mean()

    @torch.no_grad()
    def attention_diagnostics(
        self,
        images: Tensor,
        *,
        prompt: Literal["sketch", "photo"] = "sketch",
        block_index: int = 0,
    ) -> dict[str, float | int]:
        """Return attention mass for one exact transformer block."""
        result = self.attention_diagnostics_by_block(
            images, prompt=prompt, block_indices=(block_index,)
        )["blocks"][0]
        return result

    @torch.no_grad()
    def attention_diagnostics_by_block(
        self,
        images: Tensor,
        *,
        prompt: Literal["sketch", "photo"] = "sketch",
        block_indices: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        """Measure all prompt attention directions at explicitly named blocks."""
        if prompt not in {"sketch", "photo"}:
            raise ValueError("prompt must be 'sketch' or 'photo'")
        blocks = self.visual.transformer.resblocks
        if block_indices is None:
            if len(blocks) == 1:
                block_indices = (0,)
            elif len(blocks) == 2:
                block_indices = (0, 1)
            else:
                block_indices = (0, len(blocks) // 2, len(blocks) - 1)
        if not block_indices:
            raise ValueError("block_indices must contain at least one block")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in block_indices
        ):
            raise TypeError("block_indices must contain integers")
        if any(index < 0 or index >= len(blocks) for index in block_indices):
            raise ValueError(f"block indices must be between 0 and {len(blocks) - 1}")
        if len(set(block_indices)) != len(block_indices):
            raise ValueError("block_indices must be unique")
        if self.prompt_length == 0:
            return {
                "prompt": prompt,
                "prompt_token_count": 0,
                "blocks": [
                    {
                        "block_index": index,
                        "cls_to_prompt_mass": 0.0,
                        "patch_to_prompt_mass": 0.0,
                        "prompt_to_cls_mass": 0.0,
                        "prompt_to_patch_mass": 0.0,
                    }
                    for index in block_indices
                ],
            }

        block_set = set(block_indices)
        selected = self.sketch_prompt if prompt == "sketch" else self.photo_prompt
        tokens = self._prompted_tokens(images, selected)
        results: list[dict[str, float | int]] = []
        for index, block in enumerate(blocks):
            if index in block_set:
                if not isinstance(block, ResidualAttentionBlock):
                    raise TypeError(
                        "attention diagnostics support only OpenCLIP "
                        "ResidualAttentionBlock instances"
                    )
                normalized = block.ln_1(tokens)
                _, weights = block.attn(
                    normalized,
                    normalized,
                    normalized,
                    need_weights=True,
                    average_attn_weights=False,
                )
                prompt_columns = slice(1, 1 + self.prompt_length)
                patch_rows = slice(1 + self.prompt_length, tokens.shape[1])
                prompt_rows = slice(1, 1 + self.prompt_length)
                patch_columns = slice(1 + self.prompt_length, tokens.shape[1])
                results.append(
                    {
                        "block_index": index,
                        "cls_to_prompt_mass": self._attention_mass(
                            weights, slice(0, 1), prompt_columns
                        ).item(),
                        "patch_to_prompt_mass": self._attention_mass(
                            weights, patch_rows, prompt_columns
                        ).item(),
                        "prompt_to_cls_mass": self._attention_mass(
                            weights, prompt_rows, slice(0, 1)
                        ).item(),
                        "prompt_to_patch_mass": self._attention_mass(
                            weights, prompt_rows, patch_columns
                        ).item(),
                    }
                )
            tokens = block(tokens, attn_mask=None)
        results.sort(key=lambda value: int(value["block_index"]))
        return {
            "prompt": prompt,
            "prompt_token_count": self.prompt_length,
            "blocks": results,
        }
