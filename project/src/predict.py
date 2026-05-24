import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import default_val_transforms, CLASS_NAMES
from src.models.model import create_model

logger = logging.getLogger(__name__)


class SteelPredictor:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_name = checkpoint.get("model_name", "resnet18")
        self.num_classes = checkpoint.get("num_classes", 6)

        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False,
            device=str(self.device),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = default_val_transforms(image_size=224)
        logger.info(f"Loaded model from {checkpoint_path} on {self.device}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        predicted_idx = torch.argmax(probs).item()
        predicted_class = CLASS_NAMES[predicted_idx]

        probabilities = {
            CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))
        }

        return {
            "class": predicted_class,
            "probabilities": probabilities,
        }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path> [checkpoint_path]")
        sys.exit(1)

    img_path = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "artifacts/best_model.pth"

    predictor = SteelPredictor(checkpoint)
    image = Image.open(img_path)
    result = predictor.predict(image)
    print(result)
