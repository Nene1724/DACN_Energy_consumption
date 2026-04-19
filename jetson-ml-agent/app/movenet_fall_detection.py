"""
Utilities for running MoveNet pose inference from a webcam frame and deriving
simple fall-detection heuristics that are practical on Jetson Nano.
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import select
except ImportError:  # pragma: no cover - non-POSIX fallback
    select = None

import numpy as np

CV2_IMPORT_ERROR = None
try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime on device
    CV2_IMPORT_ERROR = str(exc)
    cv2 = None


GST_ATTEMPT_HISTORY: List[Dict[str, Any]] = []
GST_ATTEMPT_HISTORY_LIMIT = 8


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


def _ensure_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed in the agent image")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() or default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_gst_attempt(attempt: Dict[str, Any]) -> None:
    GST_ATTEMPT_HISTORY.append(attempt)
    if len(GST_ATTEMPT_HISTORY) > GST_ATTEMPT_HISTORY_LIMIT:
        del GST_ATTEMPT_HISTORY[:-GST_ATTEMPT_HISTORY_LIMIT]


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


def _is_live_camera_candidate(candidate: Any) -> bool:
    if isinstance(candidate, int):
        return True
    text = str(candidate).strip()
    return text.isdigit() or text.startswith("/dev/video")


def _camera_device_path(candidate: Any) -> str:
    if isinstance(candidate, int):
        return f"/dev/video{candidate}"
    text = str(candidate).strip()
    if text.isdigit():
        return f"/dev/video{text}"
    return text


def _gstreamer_mjpeg_pipeline(device: str, width: int, height: int, fps: float, caps_mode: str = "strict") -> str:
    fps_int = max(1, int(round(float(fps or 30.0))))
    if caps_mode == "strict":
        caps = f"image/jpeg,width={int(width)},height={int(height)},framerate={fps_int}/1"
    elif caps_mode == "size":
        caps = f"image/jpeg,width={int(width)},height={int(height)}"
    else:
        caps = "image/jpeg"
    return (
        f"v4l2src device={device} do-timestamp=true ! "
        f"{caps} ! "
        "jpegparse ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "fdsink fd=1 sync=false"
    )


def _gstreamer_raw_pipeline(device: str, width: int, height: int, fps: float, caps_mode: str = "strict") -> str:
    fps_int = max(1, int(round(float(fps or 30.0))))
    if caps_mode == "strict":
        caps = f"video/x-raw,width={int(width)},height={int(height)},framerate={fps_int}/1"
    elif caps_mode == "size":
        caps = f"video/x-raw,width={int(width)},height={int(height)}"
    else:
        caps = "video/x-raw"
    return (
        f"v4l2src device={device} do-timestamp=true ! "
        f"{caps} ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert n-threads=1 ! "
        "video/x-raw,format=I420 ! "
        "jpegenc quality=85 ! "
        "fdsink fd=1 sync=false"
    )


def _gstreamer_pipelines(device: str, width: int, height: int, fps: float) -> List[Tuple[str, str]]:
    mode = _env_text("CAMERA_GSTREAMER_FORMAT", "auto").lower()
    pipelines: List[Tuple[str, str]] = []
    if mode in {"auto", "mjpeg", "jpeg"}:
        pipelines.extend([
            ("gstreamer-mjpeg-strict", _gstreamer_mjpeg_pipeline(device, width, height, fps, "strict")),
            ("gstreamer-mjpeg-size", _gstreamer_mjpeg_pipeline(device, width, height, fps, "size")),
            ("gstreamer-mjpeg-auto", _gstreamer_mjpeg_pipeline(device, width, height, fps, "auto")),
        ])
    if mode in {"auto", "raw", "yuyv", "yuy2"}:
        pipelines.extend([
            ("gstreamer-raw-strict", _gstreamer_raw_pipeline(device, width, height, fps, "strict")),
            ("gstreamer-raw-size", _gstreamer_raw_pipeline(device, width, height, fps, "size")),
            ("gstreamer-raw-auto", _gstreamer_raw_pipeline(device, width, height, fps, "auto")),
        ])
    if not pipelines:
        pipelines.append(("gstreamer-mjpeg-auto", _gstreamer_mjpeg_pipeline(device, width, height, fps, "auto")))
    return pipelines


class GStreamerJpegCapture:
    """Small VideoCapture-like wrapper around gst-launch JPEG output."""

    def __init__(self, pipeline: str):
        self.pipeline = pipeline
        self.command: List[str] = []
        self.process = None
        self.buffer = bytearray()
        self.prefetched_frame = None
        self.stderr_tail = ""
        self.returncode = None
        self._open()

    def _open(self) -> None:
        command = ["gst-launch-1.0", "-q"] + shlex.split(self.pipeline)
        self.command = command
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def isOpened(self) -> bool:
        return self.process is not None and self.process.poll() is None and self.process.stdout is not None

    def _read_chunk(self, timeout: float) -> bytes:
        if not self.isOpened() or self.process is None or self.process.stdout is None:
            return b""
        fd = self.process.stdout.fileno()
        if select is not None:
            readable, _, _ = select.select([fd], [], [], max(0.0, float(timeout)))
            if not readable:
                return b""
        return os.read(fd, 65536)

    def read(self, timeout: float = 2.0):
        _ensure_cv2()
        if self.prefetched_frame is not None:
            frame = self.prefetched_frame
            self.prefetched_frame = None
            return True, frame

        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline:
            start = self.buffer.find(b"\xff\xd8")
            if start > 0:
                del self.buffer[:start]
                start = 0
            end = self.buffer.find(b"\xff\xd9", start + 2 if start >= 0 else 0)
            if start >= 0 and end >= 0:
                jpeg_bytes = bytes(self.buffer[start:end + 2])
                del self.buffer[:end + 2]
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None and getattr(frame, "size", 0) > 0:
                    return True, frame

            chunk = self._read_chunk(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            if not chunk:
                if self.process is not None and self.process.poll() is not None:
                    self._capture_stderr_tail()
                    return False, None
                continue
            self.buffer.extend(chunk)
            if len(self.buffer) > 4 * 1024 * 1024:
                del self.buffer[:-1024 * 1024]
        return False, None

    def prefetch(self, timeout: float = 2.5) -> bool:
        ok, frame = self.read(timeout=timeout)
        if ok:
            self.prefetched_frame = frame
        return bool(ok)

    def _capture_stderr_tail(self) -> str:
        process = self.process
        if process is None or process.stderr is None:
            return self.stderr_tail
        if process.poll() is None:
            return self.stderr_tail
        self.returncode = process.poll()
        try:
            raw = process.stderr.read() or b""
        except Exception:
            raw = b""
        if raw:
            self.stderr_tail = raw.decode("utf-8", "replace")[-1200:]
        return self.stderr_tail

    def diagnostic(self) -> Dict[str, Any]:
        process = self.process
        return {
            "command": " ".join(self.command),
            "pipeline": self.pipeline,
            "returncode": process.poll() if process is not None else self.returncode,
            "stderr_tail": self.stderr_tail,
            "buffer_bytes": len(self.buffer),
        }

    def release(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
            self._capture_stderr_tail()
        except Exception:
            pass
        finally:
            self.process = None


def camera_backend_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "opencv_available": cv2 is not None,
        "opencv_import_error": CV2_IMPORT_ERROR,
        "opencv_version": None,
        "camera_use_gstreamer": _env_bool("CAMERA_USE_GSTREAMER", False),
        "camera_gstreamer_format": _env_text("CAMERA_GSTREAMER_FORMAT", "auto").lower(),
        "camera_gstreamer_prefetch_timeout_s": _env_float("CAMERA_GSTREAMER_PREFETCH_TIMEOUT", 6.0),
        "gstreamer_mode": "gst-launch-jpeg-pipe",
        "gst_launch_path": shutil.which("gst-launch-1.0"),
        "gst_launch_available": shutil.which("gst-launch-1.0") is not None,
        "opencv_gstreamer_build": None,
        "opencv_gstreamer_available": False,
        "gstreamer_available": shutil.which("gst-launch-1.0") is not None,
        "last_gstreamer_attempts": list(GST_ATTEMPT_HISTORY),
    }
    if cv2 is None:
        return info

    info["opencv_version"] = getattr(cv2, "__version__", None)
    try:
        build_info = cv2.getBuildInformation()
    except Exception as exc:  # pragma: no cover - backend-specific
        info["build_error"] = str(exc)
        return info

    for line in build_info.splitlines():
        if "GStreamer" in line:
            value = line.strip()
            info["opencv_gstreamer_build"] = value
            info["opencv_gstreamer_available"] = "YES" in value.upper()
            break
    return info


def open_camera(
    camera_source: Any = "/dev/video0",
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
):
    _ensure_cv2()

    last_error = None
    for candidate in _camera_candidates(camera_source):
        if _env_bool("CAMERA_USE_GSTREAMER", False) and _is_live_camera_candidate(candidate):
            device = _camera_device_path(candidate)
            prefetch_timeout = max(1.0, _env_float("CAMERA_GSTREAMER_PREFETCH_TIMEOUT", 6.0))
            for backend_name, pipeline in _gstreamer_pipelines(device, width, height, fps):
                cap = None
                attempt: Dict[str, Any] = {
                    "ts": _utc_now_iso(),
                    "backend": backend_name,
                    "device": device,
                    "width": int(width),
                    "height": int(height),
                    "fps": float(fps or 30.0),
                    "prefetch_timeout_s": prefetch_timeout,
                    "pipeline": pipeline,
                    "success": False,
                }
                try:
                    cap = GStreamerJpegCapture(pipeline)
                except Exception as exc:  # pragma: no cover - backend-specific
                    last_error = f"{backend_name}: {exc}"
                    attempt["error"] = str(exc)
                    _record_gst_attempt(attempt)
                    cap = None
                if cap is None:
                    continue
                if cap.isOpened() and cap.prefetch(timeout=prefetch_timeout):
                    attempt["success"] = True
                    attempt.update(cap.diagnostic())
                    _record_gst_attempt(attempt)
                    return cap, f"{backend_name}:{device}"
                attempt.update(cap.diagnostic())
                cap.release()
                attempt.update(cap.diagnostic())
                last_error = f"{backend_name} pipeline failed for {device}"
                if "error" not in attempt:
                    stderr = str(attempt.get("stderr_tail") or "").strip()
                    attempt["error"] = stderr or last_error
                _record_gst_attempt(attempt)

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
            # Request MJPEG format from USB camera for hardware-accelerated
            # decoding — dramatically reduces CPU load vs raw YUYV.
            fourcc_mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            cap.set(cv2.CAP_PROP_FPS, float(fps or 30.0))
            # Minimize internal buffer to always get the latest frame
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    canvas = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


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
    fall_detected_frame = bool(fall_score >= 0.62 and pose_confidence >= min_score)

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
    fall_detected = bool(max_consecutive >= 3 or (fall_frames >= 4 and fall_ratio >= 0.4) or avg_score >= 0.72)

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
