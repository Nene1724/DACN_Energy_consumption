import argparse
import glob
import json
import os
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def collect_devices():
    results = []
    for path in sorted(glob.glob("/dev/hidraw*")):
        sysfs_name = os.path.basename(path)
        uevent_path = f"/sys/class/hidraw/{sysfs_name}/device/uevent"
        item = {"path": path}
        if os.path.exists(uevent_path):
            try:
                with open(uevent_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            item[key.lower()] = value
            except Exception as exc:
                item["error"] = str(exc)
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description="List hidraw devices visible to the container for FNB58 debugging")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = {
        "timestamp": now_iso(),
        "devices": collect_devices(),
    }
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
