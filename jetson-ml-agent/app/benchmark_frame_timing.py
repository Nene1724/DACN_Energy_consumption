import argparse
import statistics
import time
from typing import List, Tuple

import requests


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def run_snapshot_benchmark(base_url: str, runs: int, timeout_s: float) -> Tuple[List[float], int]:
    latencies_ms: List[float] = []
    ok_count = 0

    for idx in range(runs):
        started = time.perf_counter()
        try:
            response = requests.get(
                f"{base_url}/camera/snapshot",
                params={"annotate": "0"},
                timeout=timeout_s,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            if response.ok:
                ok_count += 1
            else:
                print(f"[snapshot] run={idx + 1} status={response.status_code} latency_ms={elapsed_ms:.2f}")
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            print(f"[snapshot] run={idx + 1} error={exc} latency_ms={elapsed_ms:.2f}")

    return latencies_ms, ok_count


def run_fall_detect_probe(base_url: str, duration_s: float, max_frames: int, timeout_s: float) -> None:
    payload = {
        "duration_s": duration_s,
        "max_frames": max_frames,
    }

    started = time.perf_counter()
    response = requests.post(f"{base_url}/camera/fall-detect", json=payload, timeout=timeout_s)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print("\n[fall-detect] probe")
    print(f"- http_status: {response.status_code}")
    print(f"- roundtrip_ms: {elapsed_ms:.2f}")

    try:
        data = response.json()
    except Exception:
        print("- json: unavailable")
        return

    frames = int(data.get("frames_analyzed") or 0)
    duration = float(data.get("duration_s") or 0.0)
    details = data.get("details") or {}

    print(f"- success: {data.get('success')}")
    print(f"- label: {data.get('label')}")
    print(f"- fall_score: {data.get('fall_score')}")
    print(f"- frames_analyzed: {frames}")
    print(f"- detector_duration_s: {duration:.3f}")

    if frames > 0 and duration > 0:
        per_frame_ms = (duration / float(frames)) * 1000.0
        fps = float(frames) / duration
        print(f"- estimated_ms_per_frame: {per_frame_ms:.2f}")
        print(f"- estimated_fps: {fps:.2f}")

    if "frame_read_failures" in details:
        print(f"- frame_read_failures: {details.get('frame_read_failures')}")
    if "last_frame_error" in details:
        print(f"- last_frame_error: {details.get('last_frame_error')}")


def summarize_snapshot(latencies_ms: List[float], ok_count: int) -> None:
    if not latencies_ms:
        print("No snapshot samples collected.")
        return

    values = sorted(float(x) for x in latencies_ms)
    avg_ms = statistics.mean(values)
    min_ms = values[0]
    max_ms = values[-1]
    p50_ms = _percentile(values, 0.50)
    p95_ms = _percentile(values, 0.95)
    success_rate = (ok_count / float(len(values))) * 100.0

    print("[snapshot] benchmark")
    print(f"- samples: {len(values)}")
    print(f"- success_count: {ok_count}")
    print(f"- success_rate_pct: {success_rate:.2f}")
    print(f"- avg_ms: {avg_ms:.2f}")
    print(f"- min_ms: {min_ms:.2f}")
    print(f"- p50_ms: {p50_ms:.2f}")
    print(f"- p95_ms: {p95_ms:.2f}")
    print(f"- max_ms: {max_ms:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frame loading latency from Jetson ML Agent camera endpoints.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Agent base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=20, help="Number of snapshot requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    parser.add_argument("--probe-fall-detect", action="store_true", help="Also send one /camera/fall-detect probe")
    parser.add_argument("--duration", type=float, default=2.5, help="duration_s for /camera/fall-detect")
    parser.add_argument("--max-frames", type=int, default=16, help="max_frames for /camera/fall-detect")
    args = parser.parse_args()

    runs = max(1, int(args.runs))
    timeout_s = max(1.0, float(args.timeout))
    base_url = str(args.base_url).rstrip("/")

    print(f"Target: {base_url}")
    latencies_ms, ok_count = run_snapshot_benchmark(base_url, runs, timeout_s)
    summarize_snapshot(latencies_ms, ok_count)

    if args.probe_fall_detect:
        run_fall_detect_probe(base_url, float(args.duration), int(args.max_frames), timeout_s=max(timeout_s, float(args.duration) + 8.0))


if __name__ == "__main__":
    main()
