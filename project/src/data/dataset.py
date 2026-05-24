import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def default_train_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


def default_val_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


class SteelDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, is_train: bool = True, image_size: int = 224):
        self.root = Path(root_dir)
        self.transform = transform or (
            default_train_transforms(image_size) if is_train else default_val_transforms(image_size)
        )
        self.samples = []

        images_dir = self.root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        for class_dir in sorted(images_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in CLASS_TO_IDX:
                logger.warning(f"Unknown class directory: {class_name}, skipping")
                continue
            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), label))

        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def parse_voc_xml(xml_path: str) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        objects.append({
            "name": name,
            "xmin": int(bbox.find("xmin").text),
            "ymin": int(bbox.find("ymin").text),
            "xmax": int(bbox.find("xmax").text),
            "ymax": int(bbox.find("ymax").text),
            "difficult": int(obj.find("difficult").text),
        })
    return objects
