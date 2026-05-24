import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from src.data.dataset import SteelDataset, CLASS_NAMES
from src.data.split import stratified_split
from src.models.model import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def compute_metrics(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy, preds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        acc, _ = compute_metrics(outputs, labels)
        running_acc += acc * batch_size
        total_samples += batch_size

    return running_loss / total_samples, running_acc / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        acc, preds = compute_metrics(outputs, labels)
        running_acc += acc * batch_size
        total_samples += batch_size
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return running_loss / total_samples, running_acc / total_samples, all_preds, all_labels


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    from src.data.dataset import default_train_transforms, default_val_transforms

    full_dataset = SteelDataset(
        root_dir=cfg["data"]["train_dir"],
        is_train=True,
        image_size=cfg["data"]["image_size"],
    )

    train_ds, val_ds, _ = stratified_split(
        full_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )

    train_transform = default_train_transforms(cfg["data"]["image_size"])
    val_transform = default_val_transforms(cfg["data"]["image_size"])
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    model = create_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        device=device.type,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=artifacts_dir / "logs")

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = cfg["training"]["patience"]

    for epoch in range(cfg["training"]["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        logger.info(
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "model_name": cfg["model"]["name"],
                    "num_classes": cfg["model"]["num_classes"],
                },
                artifacts_dir / "best_model.pth",
            )
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    writer.close()
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
