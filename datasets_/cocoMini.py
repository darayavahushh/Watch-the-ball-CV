import yaml
import torch
import albumentations as A
import numpy as np
import random
import os

from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.datasets import CocoDetection
from albumentations.pytorch import ToTensorV2


class MiniCocoDataset(Dataset):
    def __init__(self, coco_dataset, cfg, train=True):
        self.cfg = cfg
        self.train = train
        self.coco_dataset = coco_dataset  # this is a torchvision CocoDetection object

        # Transforms
        if self.train and self.cfg["dataset"]["use_augmentations"]:
            self.transform = A.Compose([
                A.HorizontalFlip(p=self.cfg["augmentations"]["horizontal_flip"]),
                A.RandomBrightnessContrast(p=self.cfg["augmentations"]["brightness_contrast"]),
                A.HueSaturationValue(p=self.cfg["augmentations"]["hue_saturation"]),
                A.Blur(blur_limit=3, p=self.cfg["augmentations"]["blur"]),
                A.Resize(*self.cfg["training"]["image_size"]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))
        else:
            self.transform = A.Compose([
                A.Resize(*self.cfg["training"]["image_size"]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        image, target = self.coco_dataset[idx]  # image = PIL, target = list of dicts

        image = np.array(image)

        # Extract boxes and labels
        boxes = [ann["bbox"] for ann in target]  # [x, y, w, h]
        labels = [ann["category_id"] for ann in target]

        if len(boxes) == 0:
            # if no annotations, return an empty target
            boxes = []
            labels = []

        transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)

        image = transformed["image"]
        boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["class_labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target


def get_dataloaders(config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = cfg["dataset"]["root"]
    categories = set(cfg["dataset"]["categories"])
    seed = cfg["dataset"]["seed"]
    split_ratio = cfg["dataset"]["split_ratio"]

    # Load raw COCO dataset
    coco_train = CocoDetection(
        root=os.path.join(root, "train"),
        annFile=os.path.join(root, "annotations", "instances_train.json"),
    )

    # Filter images by categories
    filtered_indices = []
    for idx in range(len(coco_train)):
        _, anns = coco_train[idx]
        # Keep image if at least one annotation is in categories
        if any(ann["category_id"] in categories for ann in anns):
            filtered_indices.append(idx)

    print(f"Filtered dataset size: {len(filtered_indices)} "
          f"(out of {len(coco_train)})")

    # Wrap into MiniCocoDataset
    filtered_dataset = Subset(coco_train, filtered_indices)
    train_dataset = MiniCocoDataset(filtered_dataset, cfg, train=True)

    # Split train/test
    total_len = len(train_dataset)
    train_len = int(split_ratio * total_len)
    test_len = total_len - train_len

    g = torch.Generator().manual_seed(seed)
    train_subset, test_subset = torch.utils.data.random_split(
        train_dataset, [train_len, test_len], generator=g
    )

    # Build loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=lambda x: tuple(zip(*x)),
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=lambda x: tuple(zip(*x)),
    )

    return train_loader, test_loader