"""
Helpers for reading FNIRSI FNB58 measurements over USB HID with pyusb.

The device exposes a HID interface on Linux and does not always create a
serial port. This reader keeps a small rolling buffer of decoded samples
and exposes cumulative energy in Wh so the agent can compare measured vs
predicted energy in real time.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional

import usb.core
import usb.util


VENDOR_ID = 0x2E3C
PRODUCT_ID = 0x5558
MAX_BUFFERED_SAMPLES = 512
SAMPLE_INTERVAL_S = 0.01
KEEPALIVE_INTERVAL_S = 1.0


def detect_fnb58_usb() -> Optional[str]:
    try:
        dev = _find_device()
    except Exception:
        return None
    if dev is None:
        return None
    bus = int(getattr(dev, "bus", 0) or 0)
    address = int(getattr(dev, "address", 0) or 0)
    return f"usb:{bus:03d}:{address:03d}"


def _parse_usb_target(port: Optional[str]):
    text = str(port or "").strip().lower()
    match = re.fullmatch(r"usb:(\d{1,3}):(\d{1,3})", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _iter_matching_devices():
    try:
        return list(usb.core.find(find_all=True, idVendor=VENDOR_ID, idProduct=PRODUCT_ID) or [])
    except Exception:
        return []


def _find_device(preferred_port: Optional[str] = None):
    devices = _iter_matching_devices()
    if not devices:
        return None

    target = _parse_usb_target(preferred_port)
    if target is not None:
        target_bus, target_address = target
        for dev in devices:
            bus = int(getattr(dev, "bus", 0) or 0)
            address = int(getattr(dev, "address", 0) or 0)
            if bus == target_bus and address == target_address:
                return dev

    return devices[0]


def _find_hid_interface_number(dev) -> Optional[int]:
    for cfg in dev:
        for interface in cfg:
            if interface.bInterfaceClass == 0x03:
                return int(interface.bInterfaceNumber)
    return None


def _describe_usb_error(exc: Exception) -> str:
    if isinstance(exc, usb.core.USBError):
        errno = getattr(exc, "errno", None)
        if errno == 2:
            return "USB device detected but configuration was rejected by the host"
        if errno == 5:
            return "USB HID transfer failed; check the FNB58 PC mode, data cable, and power"
        if errno == 16:
            return "USB device is busy"
        if errno == 19:
            return "USB device disappeared from the bus"
        if errno == 110:
            return "USB HID transfer timed out"
    return str(exc)


def _detach_kernel_driver_if_needed(dev, interface_number: int) -> None:
    try:
        if dev.is_kernel_driver_active(interface_number):
            dev.detach_kernel_driver(interface_number)
    except (NotImplementedError, usb.core.USBError):
        pass


def _resolve_interface(dev, interface_number: int):
    try:
        cfg = dev.get_active_configuration()
        return cfg[(interface_number, 0)]
    except Exception:
        pass

    try:
        cfg = dev[0]
        return cfg[(interface_number, 0)]
    except Exception:
        return None


def _request_start_stream(ep_out, is_fnb58=True) -> None:
    ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
    ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    if is_fnb58:
        ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    else:
        ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def _request_continue(ep_out) -> None:
    ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


class FNB58USBReader:
    transport = "usb_hid"

    def __init__(self, port: Optional[str] = None):
        self.port = port
        self.device = None
        self.interface_number = None
        self.ep_in = None
        self.ep_out = None
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
        try:
            dev = _find_device(self.port)
            if dev is None:
                self.connection_error = "FNB58 USB device not found"
                return False

            interface_number = _find_hid_interface_number(dev)
            if interface_number is None:
                self.connection_error = "FNB58 HID interface not found"
                return False

            _detach_kernel_driver_if_needed(dev, interface_number)
            try:
                usb.util.claim_interface(dev, interface_number)
            except usb.core.USBError as exc:
                self.connection_error = _describe_usb_error(exc)
                return False

            intf = _resolve_interface(dev, interface_number)
            if intf is None:
                self.connection_error = "Unable to access active FNB58 HID interface"
                try:
                    usb.util.release_interface(dev, interface_number)
                except Exception:
                    pass
                return False

            ep_out = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
            )
            ep_in = usb.util.find_descriptor(
                intf,
                custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN,
            )
            if ep_in is None or ep_out is None:
                self.connection_error = "FNB58 endpoints not found"
                return False

            bus = int(getattr(dev, "bus", 0) or 0)
            address = int(getattr(dev, "address", 0) or 0)
            self.interface_number = interface_number
            self.port = f"usb:{bus:03d}:{address:03d}"
            self.device = dev
            self.ep_in = ep_in
            self.ep_out = ep_out
            self.connection_error = None
            return True
        except Exception as exc:
            self.connection_error = _describe_usb_error(exc)
            return False

    def disconnect(self) -> None:
        dev = self.device
        interface_number = self.interface_number
        self.device = None
        self.ep_in = None
        self.ep_out = None
        if dev is not None:
            try:
                if interface_number is not None:
                    usb.util.release_interface(dev, interface_number)
            except Exception:
                pass
            try:
                usb.util.dispose_resources(dev)
            except Exception:
                pass

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
        if not raw or len(raw) < 63 or raw[1] != 0x04:
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

    def _read_loop(self) -> None:
        if self.ep_in is None or self.ep_out is None:
            return

        next_keepalive = time.time() + KEEPALIVE_INTERVAL_S
        try:
            _request_start_stream(self.ep_out, is_fnb58=True)
            time.sleep(0.1)
            while self.running:
                if time.time() >= next_keepalive:
                    _request_continue(self.ep_out)
                    next_keepalive = time.time() + KEEPALIVE_INTERVAL_S

                try:
                    data = self.ep_in.read(size_or_buffer=64, timeout=5000)
                except usb.core.USBTimeoutError:
                    continue
                self._decode_packet(bytes(data))
        except Exception as exc:
            self.connection_error = _describe_usb_error(exc)
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
                "meter_source": "fnb58_usb_hid",
                "transport": self.transport,
            }
