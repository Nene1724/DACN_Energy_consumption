"""
Read FNIRSI FNB58 measurements from Linux hidraw without reconfiguring USB.

This transport is safer on Jetson devices where the FNB58 also exposes a
mass-storage interface and aggressive libusb reconfiguration can make the
device fall off the bus.
"""

from __future__ import annotations

import glob
import os
import select
import threading
import time
from typing import Any, Dict, List, Optional


VENDOR_ID = "2E3C"
PRODUCT_ID = "5558"
MAX_BUFFERED_SAMPLES = 512
SAMPLE_INTERVAL_S = 0.01
KEEPALIVE_INTERVAL_S = 1.0


def _read_uevent(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                data[key] = value
    except Exception:
        return {}
    return data


def detect_fnb58_hidraw() -> Optional[str]:
    for path in sorted(glob.glob("/dev/hidraw*")):
        sysfs_name = os.path.basename(path)
        meta = _read_uevent(f"/sys/class/hidraw/{sysfs_name}/device/uevent")
        hid_id = meta.get("HID_ID", "").upper()
        if VENDOR_ID in hid_id and PRODUCT_ID in hid_id:
            return path
    return None


def _request_start_packets():
    return [
        b"\x00" + (b"\xaa\x81" + b"\x00" * 61 + b"\x8e"),
        b"\x00" + (b"\xaa\x82" + b"\x00" * 61 + b"\x96"),
        b"\x00" + (b"\xaa\x82" + b"\x00" * 61 + b"\x96"),
    ]


def _request_continue_packet():
    return b"\x00" + (b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


class FNB58HIDRawReader:
    transport = "hidraw"

    def __init__(self, port: Optional[str] = None):
        self.port = port or detect_fnb58_hidraw()
        self.fd: Optional[int] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.samples: List[Dict[str, Any]] = []
        self.sample_count = 0
        self.sum_power_w = 0.0
        self.total_energy_ws = 0.0
        self.total_capacity_as = 0.0
        self.last_values = {
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "energy_wh": None,
            "dp_v": None,
            "dn_v": None,
            "temperature_c": None,
        }
        self.connection_error: Optional[str] = None

    def connect(self) -> bool:
        if not self.port:
            self.connection_error = "FNB58 hidraw device not found"
            return False
        try:
            self.fd = os.open(self.port, os.O_RDWR | os.O_NONBLOCK)
            self.connection_error = None
            return True
        except Exception as exc:
            self.connection_error = str(exc)
            self.fd = None
            return False

    def disconnect(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
        self.fd = None

    def _append_sample(self, payload: Dict[str, Any]) -> None:
        with self.lock:
            self.samples.append(payload)
            if len(self.samples) > MAX_BUFFERED_SAMPLES:
                self.samples = self.samples[-MAX_BUFFERED_SAMPLES:]
            self.sample_count += 1
            power_w = payload.get("power_w")
            if power_w is not None:
                self.sum_power_w += float(power_w)

    def _decode_packet(self, raw: bytes) -> None:
        if not raw:
            return
        if len(raw) == 65 and raw[0] == 0:
            raw = raw[1:]
        if len(raw) < 63 or raw[1] != 0x04:
            return

        now_ts = time.time()
        for idx in range(4):
            offset = 2 + (15 * idx)
            voltage_v = int.from_bytes(raw[offset:offset + 4], "little") / 100000.0
            current_a = int.from_bytes(raw[offset + 4:offset + 8], "little") / 100000.0
            dp_v = int.from_bytes(raw[offset + 8:offset + 10], "little") / 1000.0
            dn_v = int.from_bytes(raw[offset + 10:offset + 12], "little") / 1000.0
            temperature_c = int.from_bytes(raw[offset + 13:offset + 15], "little") / 10.0
            power_w = voltage_v * current_a

            self.total_energy_ws += power_w * SAMPLE_INTERVAL_S
            self.total_capacity_as += current_a * SAMPLE_INTERVAL_S
            energy_wh = self.total_energy_ws / 3600.0

            payload = {
                "timestamp": now_ts - ((3 - idx) * SAMPLE_INTERVAL_S),
                "voltage_v": voltage_v,
                "current_a": current_a,
                "power_w": power_w,
                "energy_wh": energy_wh,
                "dp_v": dp_v,
                "dn_v": dn_v,
                "temperature_c": temperature_c,
            }
            self.last_values.update(payload)
            self._append_sample(payload)

    def _write_packet(self, payload: bytes) -> None:
        if self.fd is None:
            raise RuntimeError("hidraw is not open")
        os.write(self.fd, payload)

    def _read_loop(self) -> None:
        if self.fd is None:
            return

        next_keepalive = time.time() + KEEPALIVE_INTERVAL_S
        try:
            for packet in _request_start_packets():
                self._write_packet(packet)
            while self.running:
                if time.time() >= next_keepalive:
                    self._write_packet(_request_continue_packet())
                    next_keepalive = time.time() + KEEPALIVE_INTERVAL_S

                readable, _, _ = select.select([self.fd], [], [], 1.0)
                if not readable:
                    continue
                data = os.read(self.fd, 65)
                if not data:
                    continue
                self._decode_packet(bytes(data))
        except Exception as exc:
            self.connection_error = str(exc)
        finally:
            self.running = False
            self.disconnect()

    def start(self) -> bool:
        if not self.connect():
            return False

        self.running = True
        self.samples = []
        self.sample_count = 0
        self.sum_power_w = 0.0
        self.total_energy_ws = 0.0
        self.total_capacity_as = 0.0
        self.last_values = {
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "energy_wh": None,
            "dp_v": None,
            "dn_v": None,
            "temperature_c": None,
        }
        self.connection_error = None
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        self.disconnect()

        with self.lock:
            avg_power_w = (self.sum_power_w / self.sample_count) if self.sample_count > 0 else None
            return {
                "success": self.sample_count > 0,
                "samples_count": self.sample_count,
                "error": self.connection_error,
                "samples": list(self.samples),
                "total_energy_wh": self.last_values.get("energy_wh"),
                "total_energy_mwh": (self.last_values.get("energy_wh") or 0.0) * 1000.0 if self.last_values.get("energy_wh") is not None else None,
                "avg_power_w": avg_power_w,
                "avg_power_mw": (avg_power_w * 1000.0) if avg_power_w is not None else None,
                "last_values": dict(self.last_values),
                "meter_source": "fnb58_hidraw",
                "transport": self.transport,
            }
