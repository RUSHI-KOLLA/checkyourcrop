import numpy as np

from app.inference import (
    Detection,
    VerdictResult,
    annotate_image,
    build_verdict,
    get_group,
    letterbox,
    nms,
    parse_yolo_output,
    run_inference,
    scale_boxes,
    xywh2xyxy,
)


class TestLetterbox:
    def test_letterbox_square(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == 1.0
        assert pad == (0.0, 0.0)

    def test_letterbox_landscape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == 1.0  # 640/640 = 1, 640/480 = 1.33, min is 1
        assert pad[1] > 0  # vertical padding

    def test_letterbox_portrait(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result, ratio, pad = letterbox(img, (640, 640))
        assert result.shape == (640, 640, 3)
        assert ratio == 1.0
        assert pad[0] > 0  # horizontal padding

    def test_letterbox_pad_color(self):
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        result, _, _ = letterbox(img, (640, 640), color=(114, 114, 114), scaleup=False)
        # Corners should be pad color
        assert np.allclose(result[0, 0], [114, 114, 114])
        assert np.allclose(result[-1, -1], [114, 114, 114])


class TestNMS:
    def test_nms_empty(self):
        boxes = np.zeros((0, 4))
        scores = np.zeros((0,))
        keep = nms(boxes, scores)
        assert len(keep) == 0

    def test_nms_single(self):
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = nms(boxes, scores)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_nms_overlapping(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],  # High overlap with first
            [100, 100, 150, 150],  # No overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2
        assert 0 in keep  # Highest score
        assert 2 in keep  # No overlap

    def test_nms_conf_threshold(self):
        boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32)
        scores = np.array([0.9, 0.1], dtype=np.float32)
        # NMS doesn't filter by conf_threshold, it only does NMS
        # Filtering by confidence happens in run_inference
        keep = nms(boxes, scores, iou_threshold=0.5, conf_threshold=0.5)
        assert len(keep) == 2  # Both boxes kept (no overlap)
        # If we want conf filtering, we need to do it before calling nms
        # or use the conf_threshold in run_inference


class TestParseYoloOutput:
    def test_parse_v8_format_chn(self):
        # (1, 4+nc, N) format
        nc = 19
        N = 5
        np.random.seed(42)
        output = np.random.rand(1, 4 + nc, N).astype(np.float32) * 0.0001  # Very low values
        # Make some high confidence detections
        output[0, 4:4+nc, 0] = 0
        output[0, 4+0, 0] = 0.9  # class 0
        output[0, 4:4+nc, 1] = 0
        output[0, 4+5, 1] = 0.8  # class 5

        boxes, scores, class_ids = parse_yolo_output(output, nc=nc)
        assert len(boxes) == 2
        assert class_ids[0] == 0
        assert class_ids[1] == 5

    def test_parse_v8_format_nhc(self):
        # (1, N, 4+nc) format
        nc = 19
        N = 5
        np.random.seed(43)
        output = np.random.rand(1, N, 4 + nc).astype(np.float32) * 0.0001  # Very low values
        output[0, 0, 4+0] = 0.9
        output[0, 1, 4+5] = 0.8

        boxes, scores, class_ids = parse_yolo_output(output, nc=nc)
        assert len(boxes) == 2

    def test_parse_v5_format(self):
        # (1, N, 5+nc) with objectness
        nc = 19
        N = 5
        np.random.seed(44)
        output = np.random.rand(1, N, 5 + nc).astype(np.float32) * 0.0001  # Very low values
        output[0, 0, 4] = 0.9  # objectness
        output[0, 0, 5+0] = 0.9  # class 0
        output[0, 1, 4] = 0.8
        output[0, 1, 5+5] = 0.8

        boxes, scores, class_ids = parse_yolo_output(output, nc=nc)
        assert len(boxes) == 2


class TestXywh2xyxy:
    def test_conversion(self):
        boxes = np.array([[100, 100, 50, 50]], dtype=np.float32)  # cx, cy, w, h
        xyxy = xywh2xyxy(boxes)
        assert np.allclose(xyxy[0], [75, 75, 125, 125])


class TestScaleBoxes:
    def test_scale_back(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        ratio = 0.5
        pad = (10.0, 20.0)
        orig_shape = (480, 640)
        scaled = scale_boxes(boxes.copy(), ratio, pad, orig_shape)
        # (100-10)/0.5 = 180, (200-10)/0.5 = 380
        assert np.allclose(scaled[0, [0, 2]], [180, 380])
        assert np.allclose(scaled[0, [1, 3]], [160, 360])


class TestGetGroup:
    def test_weed_substring(self):
        assert get_group("Carpetweed") == "weed"
        assert get_group("Ragweed") == "weed"
        assert get_group("OtherWeed") == "weed"
        assert get_group("weed") == "weed"
        assert get_group("WeedPlant") == "weed"

    def test_crop_keywords(self):
        assert get_group("corn") == "crop"
        assert get_group("maize") == "crop"
        assert get_group("soybean") == "crop"
        assert get_group("wheat") == "crop"
        assert get_group("rice") == "crop"
        assert get_group("cotton") == "crop"
        assert get_group("canola") == "crop"
        assert get_group("tomato") == "crop"
        assert get_group("potato") == "crop"
        assert get_group("crop") == "crop"

    def test_other(self):
        assert get_group("Waterhemp") == "other"
        assert get_group("MorningGlory") == "other"
        assert get_group("UnknownPlant") == "other"


class TestBuildVerdict:
    def test_none(self):
        assert build_verdict([]) == "none"

    def test_crop_only(self):
        dets = [
            Detection(0, "corn", 0.9, [0, 0, 10, 10]),
            Detection(1, "wheat", 0.8, [20, 20, 30, 30]),
        ]
        assert build_verdict(dets) == "crop"

    def test_weed_only(self):
        dets = [
            Detection(0, "Carpetweed", 0.9, [0, 0, 10, 10]),
            Detection(1, "Ragweed", 0.8, [20, 20, 30, 30]),
        ]
        assert build_verdict(dets) == "weed"

    def test_mixed_crop_weed(self):
        dets = [
            Detection(0, "corn", 0.9, [0, 0, 10, 10]),
            Detection(1, "Carpetweed", 0.8, [20, 20, 30, 30]),
        ]
        assert build_verdict(dets) == "mixed"

    def test_mixed_other_crop(self):
        dets = [
            Detection(0, "corn", 0.9, [0, 0, 10, 10]),
            Detection(1, "Waterhemp", 0.8, [20, 20, 30, 30]),  # "other"
        ]
        assert build_verdict(dets) == "mixed"

    def test_other_only(self):
        dets = [
            Detection(0, "Waterhemp", 0.9, [0, 0, 10, 10]),
            Detection(1, "MorningGlory", 0.8, [20, 20, 30, 30]),
        ]
        assert build_verdict(dets) == "mixed"


class TestAnnotateImage:
    def test_annotate_empty(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = annotate_image(img, [])
        assert np.array_equal(result, img)

    def test_annotate_with_detections(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [Detection(0, "Carpetweed", 0.9, [10, 10, 50, 50])]
        result = annotate_image(img, dets)
        assert not np.array_equal(result, img)  # Should be modified


class TestRunInference:
    def test_run_inference_with_fake_detector(self, fake_detector, sample_image_bytes):
        result = run_inference(fake_detector, sample_image_bytes)
        assert isinstance(result, VerdictResult)
        assert result.verdict in ("crop", "weed", "mixed", "none")
        assert "crop" in result.counts
        assert "weed" in result.counts
        assert "other" in result.counts
        assert result.inference_ms > 0
        assert result.model_source == "fake.pt"
        assert result.annotated_image_b64 is not None
        assert result.annotated_image_b64.startswith("data:image/png;base64,")

    def test_run_inference_verdict_weed(self, fake_detector, sample_image_bytes):
        # Fake detector returns 2 weeds (Carpetweed, Ragweed) and 1 other (Waterhemp)
        result = run_inference(fake_detector, sample_image_bytes)
        # Weed classes: Carpetweed (4), Ragweed (5) -> "weed" group
        # Waterhemp (0) -> "other" group
        # Should be "mixed" since we have weed + other
        assert result.verdict == "mixed"
        assert result.counts["weed"] == 2
        assert result.counts["other"] == 1
