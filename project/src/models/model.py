import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def create_model(
    model_name: str = "resnet18",
    num_classes: int = 6,
    pretrained: bool = True,
    device: str = "auto",
) -> nn.Module:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name.startswith("resnet"):
        model = _build_resnet(model_name, num_classes, pretrained)
    elif model_name in ("mobilenet_v2", "mobilenetv2"):
        model = _build_mobilenet(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    logger.info(f"Created {model_name} with {num_classes} classes on {device}")
    return model


def _build_resnet(variant: str, num_classes: int, pretrained: bool):
    import torchvision.models as models

    weights_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
    }

    if variant not in weights_map:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

    model = getattr(models, variant)(weights=weights_map[variant])
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet(num_classes: int, pretrained: bool):
    import torchvision.models as models

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
