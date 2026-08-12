from collections.abc import Callable, Sequence
from typing import TypedDict

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifest import ManifestEntry

ImageTransform = Callable[[Image.Image], Tensor]


class EvalSample(TypedDict):
    image: Tensor
    label: int
    path: str


class RetrievalEvalDataset(Dataset[EvalSample]):
    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        transform: ImageTransform,
    ) -> None:
        if not entries:
            raise ValueError("Evaluation dataset cannot be empty")

        self.entries = tuple(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> EvalSample:
        entry = self.entries[index]

        try:
            with Image.open(entry.path) as image:
                rgb_image = image.convert("RGB")
        except OSError as error:
            raise OSError(f"Could not read image: {entry.path}") from error

        return {
            "image": self.transform(rgb_image),
            "label": entry.label,
            "path": str(entry.path),
        }
