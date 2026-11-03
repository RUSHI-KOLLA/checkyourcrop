import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.inference import CLASS_NAMES, Detector


@pytest.fixture
def fake_detector():
    """A fake detector that returns predictable detections."""
    detector = MagicMock(spec=Detector)
    detector.is_onnx = False
    detector.model_path = "fake.pt"

    def mock_predict(img_input):
        # Return fake output in (1, N, 4+nc) format
        # 3 detections: 2 weeds, 1 other
        import numpy as np
        nc = len(CLASS_NAMES)
        # Format: [x, y, w, h, class_scores...]
        output = np.zeros((1, 3, 4 + nc), dtype=np.float32)
        # Detection 1: Carpetweed (class 4, contains "weed")
        output[0, 0, :4] = [320, 240, 100, 80]  # xywh
        output[0, 0, 4 + 4] = 0.9  # class 4 score
        # Detection 2: Ragweed (class 5, contains "weed")
        output[0, 1, :4] = [400, 300, 60, 60]
        output[0, 1, 4 + 5] = 0.85
        # Detection 3: Waterhemp (class 0, no "weed" substring -> other)
        output[0, 2, :4] = [200, 200, 50, 50]
        output[0, 2, 4 + 0] = 0.8
        return output

    detector.side_effect = mock_predict
    detector.__call__ = mock_predict
    return detector


@pytest.fixture
def sample_image_bytes():
    """Create a simple test image as bytes."""
    import io

    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    # The rate limiter is a module-level instance in app.main, reset its internal state
    import app.main as main_module
    main_module.rate_limiter._requests.clear()
    yield
    main_module.rate_limiter._requests.clear()
