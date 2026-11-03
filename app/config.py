import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_path: str | None
    model_repo: str
    model_filename: str
    model_revision: str
    host: str
    port: int
    max_upload_mb: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    confidence_threshold: float
    iou_threshold: float
    device: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=os.getenv("MODEL_PATH") or None,
            model_repo=os.getenv("MODEL_REPO", "YOURNAME/checkyourcrop-model"),
            model_filename=os.getenv("MODEL_FILENAME", "best.pt"),
            model_revision=os.getenv("MODEL_REVISION", "main"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "7860")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "20")),
            rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.25")),
            iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.45")),
            device=os.getenv("DEVICE", "cpu"),
        )


settings = Settings.from_env()
