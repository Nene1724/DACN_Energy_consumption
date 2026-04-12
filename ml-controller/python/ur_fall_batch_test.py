import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import requests


DatasetKind = Literal["fall", "adl"]
CameraKind = Literal["cam0", "cam1"]


DATASET_BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data/"


@dataclass(frozen=True)
class SequenceSpec:
    kind: DatasetKind
    index: int
    camera: CameraKind

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.index:02d}-{self.camera}"

    @property
    def expected_fall(self) -> bool:
        return self.kind == "fall"

    @property
    def video_filename(self) -> str:
        return f"{self.kind}-{self.index:02d}-{self.camera}.mp4"

    def video_url(self, base_url: str) -> str:
        base = base_url.rstrip("/") + "/"
        return base + self.video_filename


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_to_cache(url: str, dest_path: Path, timeout_s: int = 300) -> None:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    _ensure_dir(dest_path.parent)

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            start = time.time()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Download timed out after {timeout_s}s: {url}")

    # On Windows, antivirus/indexers can briefly lock freshly written files.
    last_exc = None
    for _ in range(10):
        try:
            tmp_path.replace(dest_path)
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(0.25)

    raise last_exc


def upload_mp4(agent_url: str, local_mp4: Path, timeout_s: int = 180) -> str:
    url = agent_url.rstrip("/") + "/camera/upload-video"
    with local_mp4.open("rb") as f:
        resp = requests.post(
            url,
            files={"file": (local_mp4.name, f, "video/mp4")},
            timeout=timeout_s,
        )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"Upload failed: {payload}")
    video_path = payload.get("video_path")
    if not video_path:
        raise RuntimeError(f"Upload response missing video_path: {payload}")
    return str(video_path)


def preflight_check(agent_url: str) -> None:
    """Fail fast if the agent doesn't expose required endpoints."""
    upload_url = agent_url.rstrip("/") + "/camera/upload-video"
    try:
        # POST without a file should yield 400 if the endpoint exists.
        resp = requests.post(upload_url, timeout=15)
        if resp.status_code == 404:
            raise RuntimeError(
                "Agent is reachable but does not expose /camera/upload-video (404). "
                "This usually means your Jetson agent image is outdated; redeploy the latest jetson-ml-agent."
            )
    except requests.exceptions.HTTPError:
        raise
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Unable to reach agent upload endpoint: {exc}") from exc


def run_fall_detect(
    agent_url: str,
    camera_source: str,
    duration_s: float,
    max_frames: int,
    timeout_s: int = 120,
) -> dict:
    url = agent_url.rstrip("/") + "/camera/fall-detect"
    resp = requests.post(
        url,
        json={
            "duration_s": duration_s,
            "max_frames": max_frames,
            "camera_source": camera_source,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"Fall-detect failed: {payload}")
    return payload


def iter_specs(
    kind: Literal["fall", "adl", "all"],
    start_idx: int,
    end_idx: int,
    camera: CameraKind,
) -> Iterable[SequenceSpec]:
    if start_idx < 1:
        raise ValueError("start must be >= 1")
    if end_idx < start_idx:
        raise ValueError("end must be >= start")

    def clamp_end(max_end: int) -> int:
        return min(end_idx, max_end)

    if kind in ("fall", "all"):
        for i in range(start_idx, clamp_end(30) + 1):
            yield SequenceSpec(kind="fall", index=i, camera=camera)

    if kind in ("adl", "all"):
        # ADL sequences are recorded only for cam0.
        for i in range(start_idx, clamp_end(40) + 1):
            yield SequenceSpec(kind="adl", index=i, camera="cam0")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-test Jetson MoveNet fall detection using UR Fall Detection Dataset MP4s.",
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Jetson agent base URL, e.g. http://192.168.1.50:8000",
    )
    parser.add_argument(
        "--kind",
        choices=["fall", "adl", "all"],
        default="all",
        help="Which subset to test.",
    )
    parser.add_argument(
        "--camera",
        choices=["cam0", "cam1"],
        default="cam0",
        help="Camera stream to test for fall sequences (ADL always uses cam0).",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument(
        "--dataset-base-url",
        default=DATASET_BASE_URL,
        help="Base URL for UR Fall dataset files.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache directory for downloaded MP4s (defaults to repo .cache/ur_fall).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path (defaults to ml-controller/artifacts/report_figures).",
    )

    args = parser.parse_args()

    preflight_check(args.agent)

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    cache_dir = Path(args.cache_dir) if args.cache_dir else (repo_root / ".cache" / "ur_fall")
    output_path = Path(args.output) if args.output else (
        repo_root / "ml-controller" / "artifacts" / "report_figures" / f"ur_fall_eval_{_utc_ts()}.json"
    )

    specs = list(iter_specs(args.kind, args.start, args.end, args.camera))
    if not specs:
        raise RuntimeError("No sequences selected")

    results = []
    stats = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "errors": 0,
    }

    for spec in specs:
        url = spec.video_url(args.dataset_base_url)
        local_path = cache_dir / spec.video_filename

        item = {
            "sequence": {
                "kind": spec.kind,
                "index": spec.index,
                "camera": spec.camera,
                "url": url,
                "expected_fall": spec.expected_fall,
            },
            "downloaded_mp4": str(local_path),
        }

        try:
            t0 = time.perf_counter()
            download_to_cache(url, local_path)
            t1 = time.perf_counter()

            video_path = upload_mp4(args.agent, local_path)
            t2 = time.perf_counter()

            payload = run_fall_detect(
                args.agent,
                camera_source=video_path,
                duration_s=args.duration_s,
                max_frames=args.max_frames,
            )
            t3 = time.perf_counter()

            predicted_fall = bool(payload.get("fall_detected"))
            expected_fall = spec.expected_fall

            if predicted_fall and expected_fall:
                stats["tp"] += 1
            elif (not predicted_fall) and (not expected_fall):
                stats["tn"] += 1
            elif predicted_fall and (not expected_fall):
                stats["fp"] += 1
            else:
                stats["fn"] += 1

            item.update(
                {
                    "success": True,
                    "timing_s": {
                        "download": round(t1 - t0, 3),
                        "upload": round(t2 - t1, 3),
                        "fall_detect": round(t3 - t2, 3),
                        "total": round(t3 - t0, 3),
                    },
                    "agent": {
                        "video_path": video_path,
                        "camera_source": payload.get("camera_source"),
                        "frames_analyzed": payload.get("frames_analyzed"),
                        "fall_detected": predicted_fall,
                        "fall_score": payload.get("fall_score"),
                        "label": payload.get("label"),
                        "duration_s": payload.get("duration_s"),
                        "details": payload.get("details") or {},
                    },
                }
            )
        except Exception as exc:
            stats["errors"] += 1
            item.update({"success": False, "error": str(exc)})

        results.append(item)
        if item.get("success"):
            print(
                f"{spec.name}: expected={spec.expected_fall} -> "
                f"{item.get('agent', {}).get('fall_detected')} "
                f"score={item.get('agent', {}).get('fall_score')} "
                f"({item.get('timing_s', {}).get('total')}s)"
            )
        else:
            print(f"{spec.name}: ERROR: {item.get('error')}")

    total = len(results)
    accuracy = (stats["tp"] + stats["tn"]) / total if total else 0.0
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": args.agent,
        "dataset_base_url": args.dataset_base_url,
        "selection": {
            "kind": args.kind,
            "camera": args.camera,
            "start": args.start,
            "end": args.end,
            "duration_s": args.duration_s,
            "max_frames": args.max_frames,
        },
        "stats": {
            **stats,
            "total": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        },
        "results": results,
    }

    _ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSummary")
    print(json.dumps(report["stats"], indent=2))
    print(f"\nSaved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
