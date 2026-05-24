from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
import logging

from src.data.dataset import SteelDataset

logger = logging.getLogger(__name__)


def stratified_split(
    dataset: SteelDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    labels = [label for _, label in dataset.samples]
    indices = np.arange(len(dataset))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_seed,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_size = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=temp_labels,
        random_state=random_seed,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def load_data(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    random_seed: int = 42,
):
    from torch.utils.data import DataLoader
    from src.data.dataset import default_train_transforms, default_val_transforms

    full_dataset = SteelDataset(
        root_dir=data_dir,
        is_train=True,
        image_size=image_size,
    )

    train_ds, val_ds, test_ds = stratified_split(
        full_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=random_seed,
    )

    val_transform = default_val_transforms(image_size)
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        f"Data split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
