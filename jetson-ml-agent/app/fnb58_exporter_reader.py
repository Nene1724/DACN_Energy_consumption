"""
Fallback FNB58 reader that shells out to the standalone exporter.

This is useful when the long-lived Flask process has trouble re-detecting a
replugged USB device, while a fresh subprocess can still communicate with it.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


MAX_BUFFERED_SAMPLES = 512


class FNB58ExporterReader:
    transport = "usb_exporter"

    def __init__(self, port: Optional[str] = None, stream_seconds: float = 2.5):
        self.port = port
        self.stream_seconds = max(1.0, float(stream_seconds))
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.samples: List[Dict[str, Any]] = []
        self.sample_count = 0
        self.sum_power_w = 0.0
        self.total_energy_wh = 0.0
        self.connection_error: Optional[str] = None
        self.last_values = {
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "energy_wh": None,
            "dp_v": None,
            "dn_v": None,
            "temperature_c": None,
        }

    def _append_sample(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.samples.append(payload)
            if len(self.samples) > MAX_BUFFERED_SAMPLES:
                self.samples = self.samples[-MAX_BUFFERED_SAMPLES:]
            self.sample_count += 1
            power_w = payload.get("power_w")
            if power_w is not None:
                self.sum_power_w += float(power_w)

    def _run_exporter_once(self) -> Dict[str, Any]:
        command = [
            "python3",
            "/usr/src/app/app/fnb58_usb_exporter.py",
            "--stream-seconds",
            str(self.stream_seconds),
        ]
        if self.port:
            command.extend(["--port", str(self.port)])
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if not stdout:
            raise RuntimeError(stderr or "Exporter produced no output")

        last_line = stdout.splitlines()[-1]
        payload = json.loads(last_line)
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or stderr or "Exporter failed")
        return payload

    def _loop(self) -> None:
        while self.running:
            try:
                payload = self._run_exporter_once()
                energy_wh = float(payload.get("energy_wh") or 0.0)
                self.total_energy_wh += energy_wh

                sample = {
                    "timestamp": payload.get("timestamp"),
                    "voltage_v": payload.get("voltage_v"),
                    "current_a": payload.get("current_a"),
                    "power_w": payload.get("power_w"),
                    "energy_wh": self.total_energy_wh,
                    "dp_v": payload.get("dp_v"),
                    "dn_v": payload.get("dn_v"),
                    "temperature_c": payload.get("temperature_c"),
                }
                self.last_values.update(sample)
                self.connection_error = None
                self._append_sample(sample)
            except Exception as exc:
                self.connection_error = str(exc)
                time.sleep(1.0)

    def start(self) -> bool:
        self.running = True
        self.samples = []
        self.sample_count = 0
        self.sum_power_w = 0.0
        self.total_energy_wh = 0.0
        self.connection_error = None
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        deadline = time.time() + self.stream_seconds + 2.0
        while time.time() < deadline:
            if self.sample_count > 0:
                return True
            if self.connection_error:
                break
            time.sleep(0.2)

        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return False

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        with self.lock:
            avg_power_w = (self.sum_power_w / self.sample_count) if self.sample_count > 0 else None
            return {
                "success": self.sample_count > 0,
                "samples_count": self.sample_count,
                "error": self.connection_error,
                "samples": list(self.samples),
                "total_energy_wh": self.total_energy_wh if self.sample_count > 0 else None,
                "total_energy_mwh": (self.total_energy_wh * 1000.0) if self.sample_count > 0 else None,
                "avg_power_w": avg_power_w,
                "avg_power_mw": (avg_power_w * 1000.0) if avg_power_w is not None else None,
                "last_values": dict(self.last_values),
                "meter_source": "fnb58_usb_exporter",
                "transport": self.transport,
            }
