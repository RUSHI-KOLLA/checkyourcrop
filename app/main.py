import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app.config import settings
from app.inference import CLASS_NAMES, Detector, run_inference
from app.ratelimit import RateLimiter

detector: Detector | None = None
rate_limiter = RateLimiter(
    requests_per_window=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def get_model_path() -> str | None:
    """Resolve model path: local MODEL_PATH if exists, else download from HF."""
    if settings.model_path and os.path.exists(settings.model_path):
        return settings.model_path

    # Try to download from Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.model_filename,
            revision=settings.model_revision,
        )
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    model_path = get_model_path()
    if model_path:
        try:
            detector = Detector(model_path, device=settings.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
            detector = None
    else:
        detector = None
    yield
    detector = None


app = FastAPI(
    title="CheckYourCrop",
    description="Crop vs Weed Detection API",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    backend: str
    class_names: dict
    registry_source: str
    confidence_threshold: float
    iou_threshold: float


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    verdict: str
    counts: dict
    detections: list
    inference_ms: float
    model_source: str
    annotated_image: str


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/static/index.html") as f:
        return f.read()


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        model_loaded=detector is not None,
    )


@app.get("/api/model", response_model=ModelInfoResponse)
async def model_info():
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        backend="onnx" if detector.is_onnx else "ultralytics",
        class_names=CLASS_NAMES,
        registry_source=settings.model_repo,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile,
):
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    # Check model
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set MODEL_PATH or configure MODEL_REPO/MODEL_FILENAME/MODEL_REVISION.",
        )

    # Check content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    # Read file
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from exc

    # Check file size
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_mb} MB",
        )

    # Check if decodable
    try:
        Image.open(io.BytesIO(content)).verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    # Run inference
    try:
        result = run_inference(
            detector,
            content,
            conf_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(
        verdict=result.verdict,
        counts=result.counts,
        detections=[
            DetectionResponse(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
            )
            for d in result.detections
        ],
        inference_ms=result.inference_ms,
        model_source=result.model_source,
        annotated_image=result.annotated_image_b64 or "",
    )


def get_detector() -> Detector | None:
    """For testing - allows injection of fake detector."""
    return detector


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
