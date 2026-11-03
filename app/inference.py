import base64
import io
import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLO = None



# Class names from the checkpoint
CLASS_NAMES = {
    0: "Waterhemp",
    1: "MorningGlory",
    2: "Purslane",
    3: "SpottedSpurge",
    4: "Carpetweed",
    5: "Ragweed",
    6: "Eclipta",
    7: "PricklySida",
    8: "PalmerAmaranth",
    9: "Sicklepod",
    10: "Goosegrass",
    11: "Kena",
    12: "Lavhala",
    13: "Gajar Gavat",
    14: "Graceful Sandmart",
    15: "Sicklepod Mh",
    16: "Harali",
    17: "OtherWeed",
    18: "Lamber Quarter Plant",
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in original image coordinates


@dataclass
class VerdictResult:
    verdict: Literal["crop", "weed", "mixed", "none"]
    counts: dict
    detections: list[Detection]
    inference_ms: float
    model_source: str
    annotated_image_b64: str | None = None


class Detector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.session = None
        self.input_name = None
        self.output_names = None
        self.is_onnx = model_path.endswith(".onnx")
        self._load_model()

    def _load_model(self):
        if self.is_onnx:
            if not HAS_ORT:
                raise RuntimeError("onnxruntime not installed but model is .onnx")
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
        else:
            if not HAS_ULTRALYTICS:
                raise RuntimeError("ultralytics not installed but model is .pt")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Run inference, return raw output."""
        if self.is_onnx:
            return self.session.run(self.output_names, {self.input_name: img})[0]
        else:
            results = self.model(img, verbose=False)
            return results[0].boxes.data.cpu().numpy()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.predict(img)


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        _ = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Non-maximum suppression in numpy."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def parse_yolo_output(output: np.ndarray, nc: int = 19) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse YOLO output handling both (1, 4+nc, N) and (1, N, 4+nc) layouts,
    plus the v5 objectness column.
    Returns: boxes (N, 4), scores (N,), class_ids (N,)
    """
    # Squeeze batch dimension
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]

    # Handle (4+nc, N) or (N, 4+nc)
    if output.shape[0] == 4 + nc or output.shape[0] == 5 + nc:
        # (4+nc, N) or (5+nc, N) -> transpose to (N, 4+nc)
        output = output.T

    # Now output is (N, 4+nc) or (N, 5+nc)
    if output.shape[1] == 4 + nc:
        # No objectness column (v8+)
        boxes = output[:, :4]
        cls_scores = output[:, 4:]
    elif output.shape[1] == 5 + nc:
        # Has objectness column (v5)
        boxes = output[:, :4]
        obj_scores = output[:, 4]
        cls_scores = output[:, 5:]
        cls_scores = cls_scores * obj_scores[:, np.newaxis]
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}, expected 4+nc or 5+nc columns")

    # Get class predictions
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(cls_scores)), class_ids]

    # Filter by confidence
    mask = scores >= 0.001  # Very low threshold, NMS will filter more
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    return boxes, scores, class_ids


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xywh to xyxy format."""
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def scale_boxes(boxes: np.ndarray, ratio: float, pad: tuple[float, float], orig_shape: tuple[int, int]) -> np.ndarray:
    """Scale boxes from letterboxed image back to original image coordinates."""
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    # Clip to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    return boxes


def get_group(class_name: str) -> str:
    """Map class name to group: 'crop', 'weed', or 'other'."""
    cl = class_name.lower()
    if "weed" in cl:
        return "weed"
    crop_keywords = ["crop", "maize", "corn", "soy", "bean", "wheat", "rice", "cotton", "canola", "tomato", "potato"]
    if any(kw in cl for kw in crop_keywords):
        return "crop"
    return "other"


def build_verdict(detections: list[Detection]) -> Literal["crop", "weed", "mixed", "none"]:
    """Build verdict from detections."""
    if not detections:
        return "none"

    groups = {get_group(d.class_name) for d in detections}
    if groups == {"crop"}:
        return "crop"
    elif groups == {"weed"}:
        return "weed"
    elif "crop" in groups and "weed" in groups:
        return "mixed"
    else:
        return "mixed"  # other + crop/weed -> mixed


def annotate_image(
    img: np.ndarray,
    detections: list[Detection],
    colors: dict | None = None,
) -> np.ndarray:
    """Draw boxes with per-group colors and confidence labels."""
    if colors is None:
        colors = {
            "crop": (0, 255, 0),      # green
            "weed": (0, 0, 255),      # red
            "other": (255, 165, 0),   # orange
        }

    annotated = img.copy()
    for det in detections:
        group = get_group(det.class_name)
        color = colors.get(group, (255, 255, 255))
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 PNG data URI."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_inference(
    detector: Detector,
    image_bytes: bytes,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> VerdictResult:
    """Run full inference pipeline on image bytes."""
    start_time = time.perf_counter()

    # Load image, strip EXIF
    pil_img = Image.open(io.BytesIO(image_bytes))
    # Strip EXIF by converting
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    orig_shape = img.shape[:2]

    # Letterbox
    img_resized, ratio, pad = letterbox(img, new_shape=(640, 640))
    # Normalize and transpose to CHW
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 640, 640)

    # Inference
    output = detector(img_input)

    # Parse output
    boxes, scores, class_ids = parse_yolo_output(output, nc=len(CLASS_NAMES))

    # Convert xywh to xyxy
    boxes = xywh2xyxy(boxes)

    # Scale back to original image
    boxes = scale_boxes(boxes, ratio, pad, orig_shape)

    # NMS
    keep = nms(boxes, scores, iou_threshold, conf_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    # Build detections
    detections = []
    for box, score, cls_id in zip(boxes, scores, class_ids, strict=False):
        detections.append(Detection(
            class_id=int(cls_id),
            class_name=CLASS_NAMES.get(int(cls_id), f"class_{cls_id}"),
            confidence=float(score),
            bbox=box.tolist(),
        ))

    # Build verdict
    verdict = build_verdict(detections)

    # Counts by group
    counts = {"crop": 0, "weed": 0, "other": 0}
    for det in detections:
        group = get_group(det.class_name)
        counts[group] = counts.get(group, 0) + 1

    # Annotate
    annotated = annotate_image(img, detections)
    annotated_b64 = image_to_base64(annotated)

    inference_ms = (time.perf_counter() - start_time) * 1000

    return VerdictResult(
        verdict=verdict,
        counts=counts,
        detections=detections,
        inference_ms=inference_ms,
        model_source=detector.model_path,
        annotated_image_b64=annotated_b64,
    )
