import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


# Create Sketchy with split 104/21
class Sketchy(Dataset):
    def __init__(self, opts, transform, mode="train", used_cat=None, return_orig=False):

        self.opts = opts
        self.config = opts.data
        self.transform = transform
        self.return_orig = return_orig

        sketch_root = self.config.root / self.config.sketch_dir
        photo_root = self.config.root / self.config.photo_dir

        if not sketch_root.is_dir():
            raise FileNotFoundError(f"Sketch directory not found: {sketch_root}")

        if not photo_root.is_dir():
            raise FileNotFoundError(f"Photo directory not found: {photo_root}")

        available_categories = {
            entry.name
            for entry in sketch_root.iterdir()
            if entry.is_dir() and entry.name not in self.config.ignore_dirs
        }

        unseen_categories = set(self.config.unseen_classes)
        missing_unseen = unseen_categories - available_categories
        if missing_unseen:
            raise ValueError(
                f"Unseen categories missing from dataset: {sorted(missing_unseen)}"
            )

        if mode == "train":
            self.all_categories = sorted(available_categories - unseen_categories)
        elif mode in {"val", "test"}:
            self.all_categories = list(self.config.unseen_classes)
        else:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketches = sorted((sketch_root / category).glob("*.png"))
            photos = sorted((photo_root / category).glob("*.jpg"))

            self.all_sketches_path.extend(sketches)
            self.all_photos_path[category] = photos

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]
        p = Path(filepath)
        category = p.parent.name
        filename = p.name

        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path = filepath
        img_path = random.choice(self.all_photos_path[category])
        neg_path = random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data = ImageOps.pad(
            Image.open(sk_path).convert("RGB"),
            size=(self.opts.max_size, self.opts.max_size),
        )
        img_data = ImageOps.pad(
            Image.open(img_path).convert("RGB"),
            size=(self.opts.max_size, self.opts.max_size),
        )
        neg_data = ImageOps.pad(
            Image.open(neg_path).convert("RGB"),
            size=(self.opts.max_size, self.opts.max_size),
        )

        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        if self.return_orig:
            return (
                sk_tensor,
                img_tensor,
                neg_tensor,
                category,
                filename,
                sk_data,
                img_data,
                neg_data,
            )
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose(
            [
                transforms.Resize((opts.max_size, opts.max_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return dataset_transforms


if __name__ == "__main__":
    from .args.dataset_options import opts

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode="train", return_orig=True)
    dataset_val = Sketchy(
        opts,
        dataset_transforms,
        mode="val",
        used_cat=dataset_train.all_categories,
        return_orig=True,
    )

    for idx in range(len(dataset_val)):
        data = dataset_val[idx]

        (
            sk_tensor,
            img_tensor,
            neg_tensor,
            category,
            filename,
            sk_data,
            img_data,
            neg_data,
        ) = data

        canvas = Image.new("RGB", (224 * 3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save(f"output/{idx}.jpg")

        if idx == 20:
            break
