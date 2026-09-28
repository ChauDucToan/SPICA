"""CLIP visual teacher with Q/V-only LoRA; accepts raw RGB in [0, 1]."""

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


class SketchTeacher(nn.Module):
    def __init__(self, visual, mean, std):
        super().__init__()
        self.visual = deepcopy(visual).requires_grad_(False)
        # Isolate adapter initialization from the Student/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            for block in self.visual.transformer.resblocks:
                parametrize.register_parametrization(
                    block.attn, "in_proj_weight",
                    QVLoRA(block.attn.embed_dim).to(block.attn.in_proj_weight.device),
                )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def device(self):
        return self.mean.device

    def forward(self, sketches):
        return F.normalize(self.visual((sketches - self.mean) / self.std), dim=-1)

    def adapter_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if ".parametrizations.in_proj_weight.0." in name}


def cosine_distillation(student, teacher):
    return (1 - (student * teacher.detach()).sum(dim=-1)).mean()
