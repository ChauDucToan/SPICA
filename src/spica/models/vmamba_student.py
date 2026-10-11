"""Official VMamba-Tiny s1l8 backbone and white-patch-masked adaptation."""

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

UPSTREAM_COMMIT = "2ed52ead062a51a64521ed3871d52914bf532876"
SketchImages = Float[Tensor, "batch 3 224 224"]
Embeddings = Float[Tensor, "batch 512"]


class VMambaStudent(nn.Module):
    def __init__(self, masked: bool = False) -> None:
        super().__init__()
        from ..vendor.vmamba.vmamba import VSSM

        backbone = VSSM
        if masked:
            from .vmamba_white_mask import MaskedSS2D, MaskedVSSM

            backbone = MaskedVSSM
        self.backbone = backbone(
            depths=[2, 2, 8, 2],
            dims=96,
            drop_path_rate=0.2,
            patch_size=4,
            in_chans=3,
            num_classes=512,
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            patch_norm=True,
            norm_layer="ln2d",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
            posembed=False,
            imgsize=224,
            **({"_SS2D": MaskedSS2D} if masked else {}),
        )

    @property
    def device(self) -> torch.device:
        return self.backbone.classifier.head.weight.device

    def forward(self, sketches: SketchImages) -> Embeddings:
        return F.normalize(self.backbone(sketches).float(), dim=-1)


def _factory(masked: bool) -> VMambaStudent:
    import selective_scan_cuda_oflex as kernel

    from ..vendor.vmamba import csms6s

    if not csms6s.WITH_SELECTIVESCAN_OFLEX:
        raise RuntimeError("VMamba requires the official oflex backend in this process")
    model = VMambaStudent(masked)
    model.experiment_metadata = {
        "method": "vmamba_official_ss2d_tiny_s1l8_white_mask_adaptation_v1"
        if masked
        else "vmamba_official_ss2d_tiny_s1l8_student_v1",
        "adaptation": "all14_SS2D_and_masked_pool;white_exact_RGB1;native_grid_occupancy"
        if masked
        else "unmodified_native_forward",
        "upstream_repository": "https://github.com/MzeroMiko/VMamba",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_config": "classification/configs/vssm/vmambav2v_tiny_224.yaml",
        "depths": [2, 2, 8, 2],
        "dims": [96, 192, 384, 768],
        "ssm_d_state": 1,
        "ssm_ratio": 1.0,
        "ss2d_forward_type": "v05_noz",
        "drop_path_rate": 0.2,
        "readout": "native_LayerNorm_and_masked_global_average_pool;512D_linear;FP32L2"
        if masked
        else "native_LayerNorm_and_global_average_pool;512D_linear;L2",
        "input": "historical_Student_resize_crop224_RGB_0to1_no_normalization",
        "initialization": "native_upstream_random;no_pretrained_VMamba",
        "selective_scan_backend": "official_selective_scan_cuda_oflex",
        "kernel_path": str(Path(kernel.__file__).absolute()),
        "kernel_sha256": hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
        "custom_scan_or_Mamba2": False,
    }
    return model


def build_vmamba_tiny() -> VMambaStudent:
    """Production factory: native unmodified forward."""
    return _factory(False)


def build_vmamba_tiny_white_masked() -> VMambaStudent:
    """Production factory for all-block white-mask adaptation."""
    return _factory(True)
