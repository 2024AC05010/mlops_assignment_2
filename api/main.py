from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import torch
from PIL import Image, UnidentifiedImageError
import io
import time
import logging

from src.model import CatDogCNN
from src.utils import preprocess_image
from api.monitoring import tMetrics_Instance, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
tLogger = logging.getLogger(__name__)

app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="""\
## Overview
A binary image classification API for distinguishing between cats and dogs.
Designed for pet adoption platforms to automatically categorize uploaded pet images.

## Features
- Binary classification (cat vs dog)
- Confidence scoring for predictions
- Real-time metrics tracking
- Health monitoring endpoints

## Model Information
- Architecture: Custom 4-layer CNN
- Input: 224x224 RGB images
- Output: Binary classification with probability scores
- Framework: PyTorch

## Usage
1. Send an image file to the `/predict` endpoint
2. Receive classification result with confidence score
3. Monitor system health via `/health` endpoint
4. Track performance metrics via `/metrics` endpoint
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

tModel = None


@app.on_event("startup")
def load_model():
    global tModel
    tModel = CatDogCNN()
    tModel.load_state_dict(
        torch.load("models/cat_dog_model.pt", map_location="cpu")
    )
    tModel.eval()
    tLogger.info("Model loaded successfully")


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API status and model availability.

    Returns:
        JSON response with service status and model loading state
    """
    return {"status": "healthy", "model_loaded": tModel is not None}


@app.post("/predict", tags=["Prediction"])
async def predict(pFile: UploadFile = File(...)):
    """
    Classify an uploaded image as either 'cat' or 'dog'.

    ## Parameters
    - **pFile**: UploadFile - Image file (JPG, PNG, etc.)

    ## Returns
    - **prediction**: str - "cat" or "dog"
    - **confidence**: float - Confidence score (0.0 to 1.0)
    - **probabilities**: dict - Individual class probabilities

    ## Example Response
    ```json
    {
        "prediction": "dog",
        "confidence": 0.8765,
        "probabilities": {
            "cat": 0.1235,
            "dog": 0.8765
        }
    }
    ```
    """
    tStart_Time = time.time()

    tImage_Bytes = await pFile.read()
    if not tImage_Bytes:
        tLogger.warning(f"Empty file received: {pFile.filename}")
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        tImage = Image.open(io.BytesIO(tImage_Bytes))
        tImage.verify()  # Detect corruption without full decode
    except (UnidentifiedImageError, SyntaxError, OSError) as tErr:
        tLogger.warning(f"Invalid image received: {pFile.filename} — {tErr}")
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    try:
        tImage = Image.open(io.BytesIO(tImage_Bytes)).convert("RGB")
    except Exception as tErr:
        tLogger.warning(f"Image conversion failed: {pFile.filename} — {tErr}")
        raise HTTPException(
            status_code=400,
            detail="Failed to process the image. Ensure it is a valid RGB image."
        )

    # ── 4. Preprocess and run inference ───────────────────────────
    try:
        tInput_Tensor = preprocess_image(tImage)

        with torch.no_grad():
            tOutput = tModel(tInput_Tensor)

        tProbability = tOutput.item()
        tPredicted_Class = "dog" if tProbability > 0.5 else "cat"

        tLatency = time.time() - tStart_Time
        tMetrics_Instance.record_request(tLatency, tPredicted_Class)

        tConfidence = round(
            tProbability if tPredicted_Class == "dog" else 1 - tProbability, 4
        )

        tLogger.info(
            f"Prediction: {tPredicted_Class}, "
            f"Confidence: {tConfidence}, "
            f"Latency: {tLatency:.4f}s"
        )

        return {
            "prediction": tPredicted_Class,
            "confidence": tConfidence,
            "probabilities": {
                "cat": round(1 - tProbability, 4),
                "dog": round(tProbability, 4)
            }
        }

    except Exception as tErr:
        tLogger.error(f"Prediction failed: {str(tErr)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error during prediction."}
        )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """
    Get current API performance metrics.

    ## Returns
    - **total_requests**: int - Total number of prediction requests
    - **average_latency_seconds**: float - Average request latency
    - **prediction_distribution**: dict - Distribution of predictions

    ## Example Response
    ```json
    {
        "total_requests": 150,
        "average_latency_seconds": 0.0234,
        "prediction_distribution": {
            "cat": 75,
            "dog": 75
        }
    }
    ```
    """
    return tMetrics_Instance.get_metrics()


@app.get("/metrics/prometheus", tags=["Monitoring"])
def prometheus_metrics():
    """
    Get Prometheus metrics for monitoring integration.
    
    ## Returns
    Prometheus-formatted metrics for scraping by Prometheus server.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
