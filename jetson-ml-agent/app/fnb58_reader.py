"""
Helpers for reading FNIRSI FNB58 measurements over a serial port.

The device is expected to emit lines similar to:
    U:5.10V I:2.85A P:14.54W E:123.45Wh
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, Optional

import serial
import serial.tools.list_ports


LINE_PATTERNS = {
    "voltage_v": re.compile(r"U[:\s]+(\d+\.?\d*)\s*V", re.IGNORECASE),
    "current_a": re.compile(r"I[:\s]+(\d+\.?\d*)\s*A", re.IGNORECASE),
    "power_w": re.compile(r"P[:\s]+(\d+\.?\d*)\s*W(?!h)", re.IGNORECASE),
    "energy_wh": re.compile(r"E[:\s]+(\d+\.?\d*)\s*Wh", re.IGNORECASE),
}


def parse_fnb58_line(line: str) -> Optional[Dict[str, float]]:
    text = (line or "").strip()
    if not text:
        return None

    payload: Dict[str, float] = {}
    for key, pattern in LINE_PATTERNS.items():
        match = pattern.search(text)
        if match:
            payload[key] = float(match.group(1))
    return payload or None


def detect_fnb58_port() -> Optional[str]:
    for port in serial.tools.list_ports.comports():
        hint = " ".join(filter(None, [port.device, port.description, port.manufacturer, port.product])).lower()
        if any(token in hint for token in ("fnirsi", "fnb", "usb serial", "cp210", "ch340")):
            return port.device

    for candidate in ("/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0", "/dev/ttyACM1"):
        if os.path.exists(candidate):
            return candidate
    return None


class FNB58Reader:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.samples = []
        self.last_values = {
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "energy_wh": None,
        }
        self.connection_error: Optional[str] = None

    def connect(self) -> bool:
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            time.sleep(0.5)
            return True
        except Exception as exc:
            self.connection_error = str(exc)
            return False

    def disconnect(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.serial_conn = None

    def _read_loop(self) -> None:
        if self.serial_conn is None:
            return

        while self.running:
            try:
                line = self.serial_conn.readline().decode("utf-8", errors="ignore")
                parsed = parse_fnb58_line(line)
                if parsed:
                    with self.lock:
                        self.samples.append({
                            "timestamp": time.time(),
                            "data": parsed,
                            "raw": line.strip(),
                        })
                        self.last_values.update(parsed)
            except Exception as exc:
                self.connection_error = str(exc)
                break

        self.disconnect()

    def start(self) -> bool:
        if not self.connect():
            return False

        self.running = True
        self.samples = []
        self.connection_error = None
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.disconnect()

        with self.lock:
            powers = [
                sample["data"].get("power_w")
                for sample in self.samples
                if sample["data"].get("power_w") is not None
            ]
            total_energy_wh = self.last_values.get("energy_wh")
            avg_power_w = (sum(powers) / len(powers)) if powers else None
            return {
                "success": len(self.samples) > 0,
                "samples_count": len(self.samples),
                "error": self.connection_error,
                "samples": list(self.samples),
                "total_energy_wh": total_energy_wh,
                "total_energy_mwh": (total_energy_wh * 1000.0) if total_energy_wh is not None else None,
                "avg_power_w": avg_power_w,
                "avg_power_mw": (avg_power_w * 1000.0) if avg_power_w is not None else None,
                "last_values": dict(self.last_values),
            }
