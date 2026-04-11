import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import usb.core
import usb.util


VENDOR_ID = 0x2E3C
PRODUCT_ID = 0x5558


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _hex(data: bytes) -> str:
    return data.hex()


def _find_interface_number_from_sysfs():
    base = "/sys/bus/usb/devices"
    try:
        for devname in os.listdir(base):
            devdir = os.path.join(base, devname)
            vid_path = os.path.join(devdir, "idVendor")
            pid_path = os.path.join(devdir, "idProduct")
            if not (os.path.exists(vid_path) and os.path.exists(pid_path)):
                continue
            with open(vid_path, "r", encoding="utf-8") as f:
                vid = f.read().strip().lower()
            with open(pid_path, "r", encoding="utf-8") as f:
                pid = f.read().strip().lower()
            if vid != f"{VENDOR_ID:04x}" or pid != f"{PRODUCT_ID:04x}":
                continue
            for child in os.listdir(devdir):
                if ":1." not in child:
                    continue
                parts = child.split(":1.")
                if len(parts) == 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        pass
    except Exception:
        return None
    return None


def _find_hid_interface_num(dev):
    for cfg in dev:
        for interface in cfg:
            if interface.bInterfaceClass == 0x03:
                return int(interface.bInterfaceNumber)
    return None


def _parse_usb_target(port):
    text = str(port or "").strip().lower()
    match = re.fullmatch(r"usb:(\d{1,3}):(\d{1,3})", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _find_device(preferred_port=None):
    devices = list(usb.core.find(find_all=True, idVendor=VENDOR_ID, idProduct=PRODUCT_ID) or [])
    if not devices:
        return None

    target = _parse_usb_target(preferred_port)
    if target is None:
        return devices[0]

    target_bus, target_address = target
    for dev in devices:
        bus = int(getattr(dev, "bus", 0) or 0)
        address = int(getattr(dev, "address", 0) or 0)
        if bus == target_bus and address == target_address:
            return dev

    return devices[0]


def _detach_kernel_driver_if_needed(dev, interface_number):
    try:
        if dev.is_kernel_driver_active(interface_number):
            dev.detach_kernel_driver(interface_number)
    except Exception:
        pass


def _resolve_interface(dev, interface_number):
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


def _request_start_stream(ep_out):
    ep_out.write(b"\xaa\x81" + b"\x00" * 61 + b"\x8e")
    ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")
    ep_out.write(b"\xaa\x82" + b"\x00" * 61 + b"\x96")


def _request_continue(ep_out):
    ep_out.write(b"\xaa\x83" + b"\x00" * 61 + b"\x9e")


def _decode_samples(raw: bytes):
    if not raw or len(raw) < 63 or raw[1] != 0x04:
        return []
    samples = []
    now_ts = time.time()
    for idx in range(4):
        offset = 2 + (15 * idx)
        voltage_v = int.from_bytes(raw[offset:offset + 4], "little") / 100000.0
        current_a = int.from_bytes(raw[offset + 4:offset + 8], "little") / 100000.0
        dp_v = int.from_bytes(raw[offset + 8:offset + 10], "little") / 1000.0
        dn_v = int.from_bytes(raw[offset + 10:offset + 12], "little") / 1000.0
        temperature_c = int.from_bytes(raw[offset + 13:offset + 15], "little") / 10.0
        power_w = voltage_v * current_a
        samples.append({
            "timestamp": now_ts - ((3 - idx) * 0.01),
            "voltage_v": voltage_v,
            "current_a": current_a,
            "power_w": power_w,
            "dp_v": dp_v,
            "dn_v": dn_v,
            "temperature_c": temperature_c,
        })
    return samples


def stream_snapshot(dev, duration_s: float):
    interface_number = _find_hid_interface_num(dev)
    if interface_number is None:
        raise RuntimeError("HID interface not found")

    _detach_kernel_driver_if_needed(dev, interface_number)
    intf = _resolve_interface(dev, interface_number)
    if intf is None:
        raise RuntimeError("Unable to access active HID interface")

    usb.util.claim_interface(dev, interface_number)

    ep_out = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT,
    )
    ep_in = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN,
    )
    if ep_in is None or ep_out is None:
        raise RuntimeError("HID endpoints not found")

    try:
        _request_start_stream(ep_out)
        time.sleep(0.1)
        deadline = time.time() + max(0.2, duration_s)
        next_keepalive = time.time() + 1.0
        total_energy_ws = 0.0
        total_samples = 0
        last_sample = None

        while time.time() < deadline:
            if time.time() >= next_keepalive:
                _request_continue(ep_out)
                next_keepalive = time.time() + 1.0
            try:
                data = bytes(ep_in.read(64, timeout=1500))
            except usb.core.USBTimeoutError:
                continue
            for sample in _decode_samples(data):
                total_samples += 1
                total_energy_ws += sample["power_w"] * 0.01
                last_sample = sample

        if last_sample is None:
            raise RuntimeError("No stream packets received")

        energy_wh = total_energy_ws / 3600.0
        return {
            "ok": True,
            "meter_source": "fnb58_usb_stream",
            "transport": "usb_hid",
            "samples_count": total_samples,
            "duration_s": max(duration_s, 0.2),
            "energy_wh": energy_wh,
            "energy_mwh": energy_wh * 1000.0,
            "voltage_v": last_sample.get("voltage_v"),
            "current_a": last_sample.get("current_a"),
            "power_w": last_sample.get("power_w"),
            "temperature_c": last_sample.get("temperature_c"),
            "dp_v": last_sample.get("dp_v"),
            "dn_v": last_sample.get("dn_v"),
            "timestamp": _now_iso(),
        }
    finally:
        try:
            usb.util.release_interface(dev, interface_number)
        except Exception:
            pass
        try:
            usb.util.dispose_resources(dev)
        except Exception:
            pass


def get_report(dev, interface_number, report_type, report_id, length):
    bm_request_type = 0xA1
    b_request = 0x01
    w_value = ((report_type & 0xFF) << 8) | (report_id & 0xFF)
    w_index = interface_number & 0xFFFF
    data = dev.ctrl_transfer(bm_request_type, b_request, w_value, w_index, length)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Best-effort USB control-transfer exporter for FNIRSI FNB58")
    parser.add_argument("--interface", type=int, default=-1)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--report-id", type=int, default=0)
    parser.add_argument("--report-type", type=int, default=3, help="1=input, 2=output, 3=feature")
    parser.add_argument("--stream-seconds", type=float, default=0.0, help="Read live HID stream for the given duration and emit decoded JSON")
    parser.add_argument("--port", type=str, default="", help="Optional usb:BUS:ADDR target")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    output = {
        "timestamp": _now_iso(),
        "meter_source": "fnb58_usb_get_report",
        "usb": {
            "vendor_id": f"0x{VENDOR_ID:04x}",
            "product_id": f"0x{PRODUCT_ID:04x}",
        },
    }

    dev = _find_device(args.port)
    if dev is None:
        output["ok"] = False
        output["error"] = "USB device not found"
        print(json.dumps(output))
        raise SystemExit(2)

    try:
        if args.stream_seconds and args.stream_seconds > 0:
            stream_output = stream_snapshot(dev, float(args.stream_seconds))
            output.update(stream_output)
            output["usb"]["interface"] = _find_hid_interface_num(dev)
            print(json.dumps(output))
            return

        interface_number = args.interface if args.interface >= 0 else (_find_hid_interface_num(dev) or _find_interface_number_from_sysfs() or 3)
        output["usb"]["interface"] = interface_number
        output["request"] = {
            "report_type": int(args.report_type),
            "report_id": int(args.report_id),
            "length": int(args.length),
        }
        try:
            dev.set_configuration()
        except Exception:
            pass

        data = get_report(dev, interface_number, int(args.report_type), int(args.report_id), int(args.length))
        output["ok"] = True
        output["response_hex"] = _hex(data)
        output["response_len"] = len(data)
        if not args.raw:
            output["note"] = "raw_report_only; needs decoding"
        print(json.dumps(output))
    except usb.core.USBError as exc:
        output["ok"] = False
        output["error"] = f"USBError: {exc}"
        print(json.dumps(output))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
