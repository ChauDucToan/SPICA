from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .datasets import ImageTransform

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def build_clip_eval_transform(image_size: int = 224) -> ImageTransform:
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")

    return transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CLIP_IMAGE_MEAN,
                std=CLIP_IMAGE_STD,
            ),
        ]
    )
