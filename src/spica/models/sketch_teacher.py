"""CLIP Q/V-LoRA teacher with optional shallow visual prompts; raw RGB input."""

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.utils import parametrize

PackedWeights = Float[Tensor, "3*width width"]
Tokens = Float[Tensor, "batch length width"]
PromptedTokens = Float[Tensor, "batch length+4 width"]
SketchImages = Float[Tensor, "batch 3 height width"]
Embeddings = Float[Tensor, "batch embedding"]
PatchMask = Bool[Tensor, "batch tokens"]
Scalar = Float[Tensor, ""]


class QVLoRA(nn.Module):
    """Add rank-8, alpha-8 updates to packed attention Q/V, leaving K intact."""

    def __init__(self, width: int):
        super().__init__()
        self.q_a = nn.Parameter(torch.empty(8, width))
        self.q_b = nn.Parameter(torch.zeros(width, 8))
        self.v_a = nn.Parameter(torch.empty(8, width))
        self.v_b = nn.Parameter(torch.zeros(width, 8))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))

    def forward(self, weight: PackedWeights) -> PackedWeights:
        q = self.q_b @ self.q_a
        v = self.v_b @ self.v_a
        return weight + torch.cat((q, torch.zeros_like(q), v))  # alpha/rank = 1


class VisualPrompt(nn.Module):
    """Four tokens after CLS, before pre-LN; optional dynamic conditioning from sketch ink features."""

    def __init__(self, width: int, dynamic: bool = False, in_channels: int = 192):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(4, width))
        nn.init.normal_(self.tokens, std=0.02)
        self.dynamic = dynamic
        if dynamic:
            self.generator = nn.Sequential(
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Conv2d(in_channels, width, 1, bias=False),
            )
        self.current_ink = None

    def forward(self, tokens: Tokens) -> PromptedTokens:
        base = self.tokens.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        if self.dynamic and self.current_ink is not None:
            dyn = self.generator(self.current_ink).flatten(2).transpose(1, 2)
            prompts = base + dyn
            self.current_ink = None
        else:
            prompts = base
        return torch.cat((tokens[:, :1], prompts, tokens[:, 1:]), dim=1)


class SketchTeacher(nn.Module):
    def __init__(
        self,
        visual,
        mean,
        std,
        *,
        visual_prompt=False,
        dynamic_prompt=False,
        mask_white_patches=False,
        deep_visual_prompt=False,
    ):
        super().__init__()
        if deep_visual_prompt and not visual_prompt:
            raise ValueError("Deep prompts require visual prompts")
        self.mask_white_patches = mask_white_patches
        self.deep_visual_prompt = deep_visual_prompt
        self.dynamic_prompt = dynamic_prompt
        self.prompt_count = 4 if (visual_prompt or dynamic_prompt) and not deep_visual_prompt else 0
        self.visual = deepcopy(visual).requires_grad_(False)
        self.prompt = None
        object.__setattr__(self, "_student_ref", None)
        # Isolate adapter initialization from the Student/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            for block in self.visual.transformer.resblocks:
                parametrize.register_parametrization(
                    block.attn,
                    "in_proj_weight",
                    QVLoRA(block.attn.embed_dim).to(block.attn.in_proj_weight.device),
                )
            if deep_visual_prompt:
                self.deep_prompts = nn.ModuleList(
                    VisualPrompt(self.visual.conv1.out_channels)
                    for _ in self.visual.transformer.resblocks
                ).to(self.visual.conv1.weight.device)
            elif visual_prompt or dynamic_prompt:
                self.prompt = VisualPrompt(
                    self.visual.conv1.out_channels, dynamic=dynamic_prompt
                )
                self.visual.ln_pre = nn.Sequential(
                    self.prompt.to(self.visual.conv1.weight.device), self.visual.ln_pre
                )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def device(self):
        return self.mean.device

    def white_patch_mask(self, sketches: SketchImages) -> PatchMask:
        ph, pw = self.visual.conv1.kernel_size
        patches = sketches.unfold(2, ph, ph).unfold(3, pw, pw)
        blank = patches.eq(1).all(dim=(1, 4, 5)).flatten(1)
        # CLS and visual prompts are always valid, including all-white sketches.
        return F.pad(blank, (1 + self.prompt_count, 0), value=False)

    def forward(self, sketches: SketchImages) -> Embeddings:
        if self.prompt is not None and self.prompt.dynamic:
            if self.student is not None:
                with torch.no_grad():
                    ink = 1.0 - sketches.mean(dim=1, keepdim=True)
                    self.prompt.current_ink = self.student.stem(ink).detach()
        normalized = (sketches - self.mean) / self.std
        if not self.mask_white_patches and not self.deep_visual_prompt:
            return F.normalize(self.visual(normalized), dim=-1)
        tokens = self.visual._embeds(normalized)
        mask = None
        if self.mask_white_patches:
            blocked = self.white_patch_mask(sketches)
            batch, length = blocked.shape
            if self.deep_visual_prompt:
                blocked = torch.cat(
                    (blocked[:, :1], blocked.new_zeros(batch, 4), blocked[:, 1:]), dim=1
                )
                length += 4
            heads = self.visual.transformer.resblocks[0].attn.num_heads
            # OpenCLIP casts attn_mask to the input dtype: use additive -inf, not bool.
            mask = (
                tokens.new_zeros(batch, 1, 1, length)
                .masked_fill(blocked[:, None, None, :], float("-inf"))
                .expand(batch, heads, length, length)
                .reshape(batch * heads, length, length)
            )
        if self.deep_visual_prompt:
            for block, prompt in zip(
                self.visual.transformer.resblocks, self.deep_prompts
            ):
                # Native pre-LN remains applied to original CLS/patch tokens exactly once.
                learned = self.visual.ln_pre(prompt.tokens).to(tokens.dtype)
                augmented = torch.cat(
                    (
                        tokens[:, :1],
                        learned.unsqueeze(0).expand(tokens.shape[0], -1, -1),
                        tokens[:, 1:],
                    ),
                    dim=1,
                )
                tokens = block(augmented, attn_mask=mask)
                tokens = torch.cat((tokens[:, :1], tokens[:, 5:]), dim=1)
        else:
            tokens = self.visual.transformer(tokens, attn_mask=mask)
        pooled, _ = self.visual._pool(tokens)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        return F.normalize(pooled, dim=-1)

    @property
    def student(self):
        return getattr(self, "_student_ref", None)

    def set_student(self, student):
        object.__setattr__(self, "_student_ref", student)

    def adapter_state_dict(self):
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {
            name: value
            for name, value in self.state_dict().items()
            if name in trainable
        }


def cross_modal_relational_kd(
    student_query: Embeddings,
    teacher_query: Embeddings,
    photos: Embeddings,
    temperature: float = 0.2,
) -> Scalar:
    logits_s = student_query @ photos.T / temperature
    logits_t = teacher_query.detach() @ photos.T / temperature
    p_t = F.softmax(logits_t, dim=-1)
    log_p_s = F.log_softmax(logits_s, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean")


def intra_modal_affinity_kd(
    student_query: Embeddings,
    teacher_query: Embeddings,
) -> Scalar:
    aff_s = student_query @ student_query.T
    aff_t = teacher_query.detach() @ teacher_query.detach().T
    return F.mse_loss(aff_s, aff_t)


def feature_alignment_kd(
    student_query: Embeddings,
    teacher_query: Embeddings,
) -> Scalar:
    """Point-wise Euclidean MSE anchor per vector: ||q_s - q_t||_2^2."""
    return ((student_query - teacher_query.detach()) ** 2).sum(dim=-1).mean()


def relational_distillation(
    student_query: Embeddings,
    teacher_query: Embeddings,
    photos: Embeddings,
    temperature: float = 0.2,
    weight_fd: float = 1.0,
) -> tuple[Scalar, Scalar, Scalar, Scalar]:
    # ponytail: Combined relational + affinity + feature MSE anchor.
    rel = cross_modal_relational_kd(student_query, teacher_query, photos, temperature)
    aff = intra_modal_affinity_kd(student_query, teacher_query)
    fd = feature_alignment_kd(student_query, teacher_query)
    total = rel + aff + weight_fd * fd
    return total, rel, aff, fd


def cosine_distillation(student: Embeddings, teacher: Embeddings) -> Scalar:
    return (1 - (student * teacher.detach()).sum(dim=-1)).mean()
