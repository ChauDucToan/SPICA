"""CLIP Q/V-LoRA teacher with optional shallow visual prompts; raw RGB input."""

import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize


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

    def forward(self, weight):
        q = self.q_b @ self.q_a
        v = self.v_b @ self.v_a
        return weight + torch.cat((q, torch.zeros_like(q), v))  # alpha/rank = 1


class VisualPrompt(nn.Module):
    """Four tokens after CLS, before pre-LN; original patch positions stay intact."""

    def __init__(self, width):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(4, width))
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, tokens):
        prompts = self.tokens.to(tokens.dtype).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return torch.cat((tokens[:, :1], prompts, tokens[:, 1:]), dim=1)


class SketchTeacher(nn.Module):
    def __init__(self, visual, mean, std, *, visual_prompt=False, mask_white_patches=False):
        super().__init__()
        self.mask_white_patches = mask_white_patches
        self.prompt_count = 4 if visual_prompt else 0
        self.visual = deepcopy(visual).requires_grad_(False)
        # Isolate adapter initialization from the Student/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            for block in self.visual.transformer.resblocks:
                parametrize.register_parametrization(
                    block.attn, "in_proj_weight",
                    QVLoRA(block.attn.embed_dim).to(block.attn.in_proj_weight.device),
                )
            if visual_prompt:
                prompt = VisualPrompt(self.visual.conv1.out_channels)
                self.visual.ln_pre = nn.Sequential(
                    prompt.to(self.visual.conv1.weight.device), self.visual.ln_pre
                )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def device(self):
        return self.mean.device

    def white_patch_mask(self, sketches):
        ph, pw = self.visual.conv1.kernel_size
        patches = sketches.unfold(2, ph, ph).unfold(3, pw, pw)
        blank = patches.eq(1).all(dim=(1, 4, 5)).flatten(1)
        # CLS and visual prompts are always valid, including all-white sketches.
        return F.pad(blank, (1 + self.prompt_count, 0), value=False)

    def forward(self, sketches):
        normalized = (sketches - self.mean) / self.std
        if not self.mask_white_patches:
            return F.normalize(self.visual(normalized), dim=-1)
        tokens = self.visual._embeds(normalized)
        blocked = self.white_patch_mask(sketches)
        batch, length = blocked.shape
        heads = self.visual.transformer.resblocks[0].attn.num_heads
        # OpenCLIP casts attn_mask to the input dtype: use additive -inf, not bool.
        mask = tokens.new_zeros(batch, 1, 1, length).masked_fill(
            blocked[:, None, None, :], float("-inf")
        ).expand(batch, heads, length, length).reshape(batch * heads, length, length)
        tokens = self.visual.transformer(tokens, attn_mask=mask)
        pooled, _ = self.visual._pool(tokens)
        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        return F.normalize(pooled, dim=-1)

    def adapter_state_dict(self):
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value for name, value in self.state_dict().items() if name in trainable}


def cosine_distillation(student, teacher):
    return (1 - (student * teacher.detach()).sum(dim=-1)).mean()
