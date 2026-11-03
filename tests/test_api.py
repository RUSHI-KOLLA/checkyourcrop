import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import CLASS_NAMES
from app.main import app, get_detector


class FakeDetector:
    """Fake detector for testing API endpoints."""
    def __init__(self):
        self.is_onnx = False
        self.model_path = "fake.pt"

    def __call__(self, img_input):
        import numpy as np
        nc = len(CLASS_NAMES)
        output = np.zeros((1, 3, 4 + nc), dtype=np.float32)
        # Carpetweed (class 4) - weed
        output[0, 0, :4] = [320, 240, 100, 80]
        output[0, 0, 4 + 4] = 0.9
        # Ragweed (class 5) - weed
        output[0, 1, :4] = [400, 300, 60, 60]
        output[0, 1, 4 + 5] = 0.85
        # Waterhemp (class 0) - other
        output[0, 2, :4] = [200, 200, 50, 50]
        output[0, 2, 4 + 0] = 0.8
        return output


@pytest.fixture
def client():
    # Override the detector dependency
    fake = FakeDetector()
    app.dependency_overrides[get_detector] = lambda: fake
    with TestClient(app) as c:
        # Set global detector after TestClient initialization (lifespan already ran)
        import app.main as main_module
        main_module.detector = fake
        yield c
    app.dependency_overrides.clear()
    main_module.detector = None


@pytest.fixture
def client_no_model():
    # No model loaded
    app.dependency_overrides[get_detector] = lambda: None
    with TestClient(app) as c:
        import app.main as main_module
        main_module.detector = None
        yield c
    app.dependency_overrides.clear()


def create_test_image() -> bytes:
    img = Image.new("RGB", (640, 480), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["model_loaded"] is True


def test_healthz_no_model(client_no_model):
    resp = client_no_model.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is False


def test_root_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CheckYourCrop" in resp.text


def test_model_info(client):
    resp = client.get("/api/model")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "ultralytics"
    assert "class_names" in data
    assert data["registry_source"] == "YOURNAME/checkyourcrop-model"


def test_model_info_no_model(client_no_model):
    resp = client_no_model.get("/api/model")
    assert resp.status_code == 503


def test_predict_success(client):
    img_bytes = create_test_image()
    resp = client.post("/api/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "verdict" in data
    assert data["verdict"] in ("crop", "weed", "mixed", "none")
    assert "counts" in data
    assert "detections" in data
    assert "inference_ms" in data
    assert "model_source" in data
    assert "annotated_image" in data
    assert data["annotated_image"].startswith("data:image/png;base64,")


def test_predict_no_model(client_no_model):
    img_bytes = create_test_image()
    resp = client_no_model.post("/api/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 503
    assert "Model not loaded" in resp.json()["detail"]


def test_predict_invalid_content_type(client):
    resp = client.post("/api/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 415


def test_predict_invalid_image(client):
    resp = client.post("/api/predict", files={"file": ("test.jpg", b"not an image", "image/jpeg")})
    assert resp.status_code == 400


def test_predict_large_file(client):
    # Create a file larger than 10 MB
    large_data = b"x" * (11 * 1024 * 1024)
    resp = client.post("/api/predict", files={"file": ("large.jpg", large_data, "image/jpeg")})
    assert resp.status_code == 413


def test_predict_rate_limit(client):
    img_bytes = create_test_image()
    # Make 21 requests (limit is 20)
    for i in range(21):
        resp = client.post("/api/predict", files={"file": (f"test{i}.jpg", img_bytes, "image/jpeg")})
        if i < 20:
            assert resp.status_code == 200, f"Request {i} failed: {resp.text}"
        else:
            assert resp.status_code == 429, f"Request {i} should be rate limited"
            assert "Retry-After" in resp.headers
            break


def test_predict_response_shape(client):
    img_bytes = create_test_image()
    resp = client.post("/api/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()

    # Verify all required fields
    assert isinstance(data["verdict"], str)
    assert isinstance(data["counts"], dict)
    assert "crop" in data["counts"]
    assert "weed" in data["counts"]
    assert "other" in data["counts"]
    assert isinstance(data["detections"], list)
    assert isinstance(data["inference_ms"], int | float)
    assert isinstance(data["model_source"], str)
    assert isinstance(data["annotated_image"], str)

    # Verify detection structure
    for det in data["detections"]:
        assert "class_id" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4
