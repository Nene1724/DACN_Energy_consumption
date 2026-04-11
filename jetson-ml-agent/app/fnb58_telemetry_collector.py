import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import requests


def _as_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_payload(raw, source_name):
    if not isinstance(raw, dict):
        return None

    payload = {
        "energy_mwh": _as_float(raw.get("energy_mwh")),
        "energy_wh": _as_float(raw.get("energy_wh")),
        "energy_mah": _as_float(raw.get("energy_mah")),
        "voltage_v": _as_float(raw.get("voltage_v") or raw.get("voltage")),
        "power_w": _as_float(raw.get("power_w")),
        "power_mw": _as_float(raw.get("power_mw")),
        "duration_s": _as_float(raw.get("duration_s")),
        "duration_ms": _as_float(raw.get("duration_ms")),
        "latency_s": _as_float(raw.get("latency_s")),
        "meter_note": raw.get("note") or raw.get("meter_note"),
        "meter_source": raw.get("meter_source") or source_name,
        "timestamp": raw.get("timestamp") or datetime.now(timezone.utc).isoformat(),
    }

    has_energy = any(
        payload.get(key) is not None
        for key in ("energy_mwh", "energy_wh", "energy_mah", "power_w", "power_mw")
    )
    return payload if has_energy else None


def read_from_command(command):
    completed = subprocess.run(
        command,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Collector command failed: {completed.stderr.strip()}")
    text = completed.stdout.strip()
    if not text:
        raise RuntimeError("Collector command returned empty output")
    return json.loads(text)


def read_from_json_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def post_telemetry(agent_base_url, payload, timeout):
    response = requests.post(f"{agent_base_url.rstrip('/')}/telemetry", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Poll external FNB58 exporter output and forward it to /telemetry")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8000")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--source", default="fnb58")
    parser.add_argument("--once", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json-command")
    group.add_argument("--json-file")

    args = parser.parse_args()
    last_fingerprint = None

    while True:
        try:
            raw = read_from_command(args.json_command) if args.json_command else read_from_json_file(args.json_file)
            payload = _normalize_payload(raw, args.source)
            if payload is None:
                print("[collector] skip: no usable energy fields found")
            else:
                fingerprint = json.dumps(payload, sort_keys=True)
                if fingerprint != last_fingerprint:
                    result = post_telemetry(args.agent_url, payload, timeout=args.timeout)
                    last_fingerprint = fingerprint
                    print(f"[collector] sent ok: {result.get('message', 'Telemetry ingested')}")
                else:
                    print("[collector] skip: payload unchanged")
        except KeyboardInterrupt:
            print("[collector] stopped by user")
            return
        except Exception as exc:
            print(f"[collector] error: {exc}")

        if args.once:
            return
        time.sleep(max(0.3, args.interval))


if __name__ == "__main__":
    main()
