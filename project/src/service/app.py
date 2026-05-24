import logging
import os
from io import BytesIO
from pathlib import Path

import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from src.predict import SteelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("service")

predictor: SteelPredictor | None = None


def load_config():
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    cfg = load_config()
    model_path = cfg["service"]["model_path"]
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}. Train first: python -m src.train")
        predictor = None
    else:
        predictor = SteelPredictor(model_path)
        logger.info("Predictor loaded successfully")
    yield


app = FastAPI(
    title="Steel Defect Classification API",
    description="Классификация дефектов поверхности стали по изображению",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if predictor is None:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = file.file.read()
        image = Image.open(BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    finally:
        file.file.close()

    result = predictor.predict(image)
    return result


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "src.service.app:app",
        host=cfg["service"]["host"],
        port=cfg["service"]["port"],
        reload=True,
    )
