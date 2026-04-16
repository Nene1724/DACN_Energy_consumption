"""
Utilities for running MoveNet pose inference from a webcam frame and deriving
simple fall-detection heuristics that are practical on Jetson Nano.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime on device
    cv2 = None


KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

TORSO_KEYS = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
HEAD_KEYS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
ANKLE_KEYS = ("left_ankle", "right_ankle")

FALL_FRAME_SCORE_THRESHOLD = float(os.getenv("FALL_FRAME_SCORE_THRESHOLD", "0.62") or "0.62")
FALL_WINDOW_AVG_SCORE_THRESHOLD = float(os.getenv("FALL_WINDOW_AVG_SCORE_THRESHOLD", "0.72") or "0.72")
FALL_WINDOW_RATIO_THRESHOLD = float(os.getenv("FALL_WINDOW_RATIO_THRESHOLD", "0.4") or "0.4")
FALL_WINDOW_MIN_FRAMES = max(1, int(os.getenv("FALL_WINDOW_MIN_FRAMES", "4") or "4"))
FALL_WINDOW_MIN_CONSECUTIVE = max(1, int(os.getenv("FALL_WINDOW_MIN_CONSECUTIVE", "3") or "3"))

# Reuse canvases by size/dtype to reduce per-frame allocations during resize+pad.
_RESIZE_CANVAS_CACHE: Dict[Tuple[int, str], np.ndarray] = {}


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed in the agent image")


def _camera_candidates(camera_source: Any) -> Iterable[Any]:
    if camera_source is None:
        yield 0
        yield "/dev/video0"
        return

    text = str(camera_source).strip()
    if not text:
        yield 0
        yield "/dev/video0"
        return

    yield text
    if text.startswith("/dev/video"):
        suffix = text.replace("/dev/video", "", 1)
        if suffix.isdigit():
            yield int(suffix)
    elif text.isdigit():
        yield int(text)
        yield f"/dev/video{text}"


def open_camera(camera_source: Any = "/dev/video0", width: int = 640, height: int = 480):
    _ensure_cv2()

    last_error = None
    for candidate in _camera_candidates(camera_source):
        cap = None
        try:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        except Exception as exc:  # pragma: no cover - backend-specific
            last_error = str(exc)
            cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            try:
                cap = cv2.VideoCapture(candidate)
            except Exception as exc:  # pragma: no cover - backend-specific
                last_error = str(exc)
                cap = None

        if cap is not None and cap.isOpened():
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            return cap, candidate

    raise RuntimeError(last_error or f"Unable to open camera source: {camera_source}")


def capture_camera_snapshot(
    camera_source: Any = "/dev/video0",
    width: int = 640,
    height: int = 480,
    jpeg_quality: int = 85,
    overlay_lines: Optional[Iterable[str]] = None,
) -> Tuple[bytes, Any]:
    """
    Capture one JPEG frame from the camera.

    Returns:
        (jpeg_bytes, actual_source)
    """
    _ensure_cv2()

    cap, actual_source = open_camera(camera_source, width=width, height=height)
    frame_bgr = None
    try:
        warmup_reads = 0
        while warmup_reads < 2:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            warmup_reads += 1

        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise RuntimeError("Camera frame read failed")
    finally:
        try:
            cap.release()
        except Exception:
            pass

    lines = [str(line).strip() for line in (overlay_lines or []) if str(line).strip()]
    if lines:
        origin_y = 28
        for idx, line in enumerate(lines[:6]):
            y = origin_y + (idx * 24)
            cv2.putText(
                frame_bgr,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (24, 24, 24),
                1,
                cv2.LINE_AA,
            )

    quality = max(40, min(int(jpeg_quality), 95))
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG")
    return encoded.tobytes(), actual_source


def resize_with_pad_rgb(image_rgb: np.ndarray, target_size: int) -> np.ndarray:
    _ensure_cv2()

    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise RuntimeError("Invalid input frame dimensions")

    scale = min(float(target_size) / float(width), float(target_size) / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    cache_key = (int(target_size), str(resized.dtype))
    canvas = _RESIZE_CANVAS_CACHE.get(cache_key)
    if canvas is None or canvas.shape != (target_size, target_size, 3):
        canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
        _RESIZE_CANVAS_CACHE[cache_key] = canvas
    else:
        canvas.fill(0)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas.copy()


def preprocess_frame_bgr(frame_bgr: np.ndarray, input_size: int, input_dtype=np.uint8) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_cv2()

    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise RuntimeError("Empty frame received from camera")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded = resize_with_pad_rgb(frame_rgb, int(input_size))

    if input_dtype == np.uint8:
        tensor = padded.astype(np.uint8)
    else:
        tensor = padded.astype(np.float32)

    return np.expand_dims(tensor, axis=0), frame_rgb


def extract_keypoints(output: Any) -> np.ndarray:
    arr = np.array(output)
    if arr.size < 17 * 3:
        raise RuntimeError(f"Unexpected pose output shape: {arr.shape}")
    arr = arr.reshape(-1, 17, 3)[0]
    return arr.astype(np.float32)


def _keypoint(keypoints: np.ndarray, name: str) -> np.ndarray:
    return keypoints[KEYPOINT_DICT[name]]


def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _point_if_visible(keypoints: np.ndarray, name: str, min_score: float) -> Optional[np.ndarray]:
    point = _keypoint(keypoints, name)
    return point if float(point[2]) >= min_score else None


def _mean_visible_point(keypoints: np.ndarray, names: Iterable[str], min_score: float) -> Optional[np.ndarray]:
    visible: List[np.ndarray] = []
    for name in names:
        point = _point_if_visible(keypoints, name, min_score)
        if point is not None:
            visible.append(point)
    if not visible:
        return None
    return np.mean(np.stack(visible, axis=0), axis=0)


def analyze_pose(keypoints: np.ndarray, min_score: float = 0.25) -> Dict[str, Any]:
    scores = keypoints[:, 2]
    visible_mask = scores >= float(min_score)

    if int(np.sum(visible_mask)) < 4:
        return {
            "valid_pose": False,
            "visible_keypoints": int(np.sum(visible_mask)),
            "pose_confidence": float(np.mean(scores)) if scores.size else 0.0,
            "fall_score": 0.0,
            "fall_detected_frame": False,
            "label": "insufficient_pose",
        }

    visible_points = keypoints[visible_mask]
    ys = visible_points[:, 0]
    xs = visible_points[:, 1]
    bbox_height = max(float(np.max(ys) - np.min(ys)), 1e-6)
    bbox_width = max(float(np.max(xs) - np.min(xs)), 1e-6)
    aspect_ratio = bbox_width / bbox_height
    vertical_span = bbox_height

    shoulders = _mean_visible_point(keypoints, ("left_shoulder", "right_shoulder"), min_score)
    hips = _mean_visible_point(keypoints, ("left_hip", "right_hip"), min_score)
    head = _mean_visible_point(keypoints, HEAD_KEYS, min_score)
    ankles = _mean_visible_point(keypoints, ANKLE_KEYS, min_score)

    torso_angle_deg = None
    torso_vertical_delta = None
    if shoulders is not None and hips is not None:
        dy = float(hips[0] - shoulders[0])
        dx = float(hips[1] - shoulders[1])
        torso_vertical_delta = abs(dy)
        torso_angle_deg = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))

    center_y = None
    if shoulders is not None and hips is not None:
        center_y = float((shoulders[0] + hips[0]) / 2.0)
    elif hips is not None:
        center_y = float(hips[0])
    elif head is not None:
        center_y = float(head[0])

    head_to_ankle_span = None
    if head is not None and ankles is not None:
        head_to_ankle_span = max(float(ankles[0] - head[0]), 1e-6)

    pose_confidence = float(np.mean(scores[visible_mask])) if np.any(visible_mask) else float(np.mean(scores))

    fall_score = 0.0
    if torso_angle_deg is not None:
        fall_score += min(max((torso_angle_deg - 35.0) / 55.0, 0.0), 1.0) * 0.40
    if aspect_ratio > 0.8:
        fall_score += min(max((aspect_ratio - 0.8) / 0.8, 0.0), 1.0) * 0.25
    if vertical_span < 0.45:
        fall_score += min(max((0.45 - vertical_span) / 0.25, 0.0), 1.0) * 0.15
    if center_y is not None and center_y > 0.45:
        fall_score += min(max((center_y - 0.45) / 0.35, 0.0), 1.0) * 0.10
    if torso_vertical_delta is not None and torso_vertical_delta < 0.14:
        fall_score += min(max((0.14 - torso_vertical_delta) / 0.12, 0.0), 1.0) * 0.10

    fall_score = float(min(max(fall_score, 0.0), 1.0))
    fall_detected_frame = bool(fall_score >= FALL_FRAME_SCORE_THRESHOLD and pose_confidence >= min_score)

    label = "uncertain"
    if fall_detected_frame:
        label = "fall_like_posture"
    elif torso_angle_deg is not None and torso_angle_deg < 30.0 and aspect_ratio < 0.85 and vertical_span > 0.42:
        label = "upright"

    return {
        "valid_pose": True,
        "visible_keypoints": int(np.sum(visible_mask)),
        "pose_confidence": round(pose_confidence, 4),
        "bbox_height_norm": round(vertical_span, 4),
        "bbox_width_norm": round(bbox_width, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "center_y_norm": round(center_y, 4) if center_y is not None else None,
        "head_to_ankle_span_norm": round(head_to_ankle_span, 4) if head_to_ankle_span is not None else None,
        "torso_angle_deg": round(torso_angle_deg, 2) if torso_angle_deg is not None else None,
        "torso_vertical_delta": round(torso_vertical_delta, 4) if torso_vertical_delta is not None else None,
        "fall_score": round(fall_score, 4),
        "fall_detected_frame": fall_detected_frame,
        "label": label,
    }


def summarize_detection_window(frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not frame_results:
        return {
            "frames_analyzed": 0,
            "fall_detected": False,
            "fall_score": 0.0,
            "fall_frames": 0,
            "max_consecutive_fall_frames": 0,
            "label": "no_frames",
        }

    fall_frames = 0
    consecutive = 0
    max_consecutive = 0
    best_frame = None
    best_score = -1.0

    for frame in frame_results:
        score = float(frame.get("fall_score") or 0.0)
        if score > best_score:
            best_score = score
            best_frame = frame
        if frame.get("fall_detected_frame"):
            fall_frames += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    frames_analyzed = len(frame_results)
    fall_ratio = fall_frames / float(frames_analyzed)
    avg_score = float(np.mean([float(item.get("fall_score") or 0.0) for item in frame_results]))
    fall_detected = bool(
        max_consecutive >= FALL_WINDOW_MIN_CONSECUTIVE
        or (fall_frames >= FALL_WINDOW_MIN_FRAMES and fall_ratio >= FALL_WINDOW_RATIO_THRESHOLD)
        or avg_score >= FALL_WINDOW_AVG_SCORE_THRESHOLD
    )

    return {
        "frames_analyzed": frames_analyzed,
        "fall_detected": fall_detected,
        "fall_score": round(best_score if best_score >= 0 else avg_score, 4),
        "avg_fall_score": round(avg_score, 4),
        "fall_frames": fall_frames,
        "fall_frame_ratio": round(fall_ratio, 4),
        "max_consecutive_fall_frames": max_consecutive,
        "label": "fall_detected" if fall_detected else (best_frame or {}).get("label", "no_fall"),
        "best_frame": best_frame,
    }
