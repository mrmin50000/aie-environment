import pytest
import torch
from pathlib import Path

from src.data.dataset import SteelDataset, CLASS_NAMES, CLASS_TO_IDX, parse_voc_xml
from src.data.split import stratified_split


def test_dataset_creates_correctly():
    ds = SteelDataset(root_dir="data/train", is_train=False)
    assert len(ds) > 0
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert 0 <= label < 6


def test_dataset_all_classes_present():
    ds = SteelDataset(root_dir="data/train", is_train=False)
    present_labels = {label for _, label in ds.samples}
    assert present_labels == set(range(6))


def test_dataset_class_names_match():
    assert len(CLASS_NAMES) == 6
    for name in CLASS_NAMES:
        assert name in CLASS_TO_IDX


def test_parse_voc_xml():
    xml_files = sorted(Path("data/train/annotations").glob("*.xml"))
    assert len(xml_files) > 0
    objs = parse_voc_xml(str(xml_files[0]))
    assert len(objs) > 0
    for obj in objs:
        assert "name" in obj
        assert "xmin" in obj
        assert "ymin" in obj
        assert "xmax" in obj
        assert "ymax" in obj


def test_stratified_split():
    ds = SteelDataset(root_dir="data/train", is_train=False)
    train_subset, val_subset, test_subset = stratified_split(ds)

    total = len(train_subset) + len(val_subset) + len(test_subset)
    assert total == len(ds)
    assert len(train_subset) > len(val_subset)
    assert len(val_subset) > 0
    assert len(test_subset) > 0


def test_dataloader_output():
    from torch.utils.data import DataLoader
    from src.data.dataset import default_val_transforms

    ds = SteelDataset(root_dir="data/train", is_train=False)
    ds.transform = default_val_transforms(224)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    images, labels = next(iter(loader))
    assert images.shape == (16, 3, 224, 224)
    assert labels.shape == (16,)
