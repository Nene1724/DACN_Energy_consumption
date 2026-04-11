import argparse
import csv
import time

import requests


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_status(agent_url, timeout):
    response = requests.get(f"{agent_url.rstrip('/')}/status", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _run_predict(agent_url, timeout):
    response = requests.post(f"{agent_url.rstrip('/')}/predict", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _extract_predicted_mwh(status_payload, explicit):
    if explicit is not None:
        return explicit
    metrics = status_payload.get("energy_metrics") or {}
    predicted = _safe_float(metrics.get("predicted_mwh"))
    if predicted is not None:
        return predicted
    model_info = status_payload.get("model_info") or {}
    for key in ("predicted_energy_mwh", "energy_avg_mwh", "energy_mwh"):
        predicted = _safe_float(model_info.get(key))
        if predicted is not None:
            return predicted
    return None


def _latest_sample(status_payload):
    history = (status_payload.get("energy_metrics") or {}).get("history") or []
    return history[-1] if history else None


def _compute_row(run_idx, predicted, sample):
    measured = _safe_float(sample.get("energy_mwh")) if sample else None
    abs_error = None
    err_pct = None
    if measured is not None and predicted is not None and predicted > 0:
        abs_error = abs(measured - predicted)
        err_pct = (abs_error / predicted) * 100.0
    return {
        "run": run_idx,
        "timestamp": sample.get("timestamp") if sample else None,
        "predicted_mwh": predicted,
        "measured_mwh": measured,
        "abs_error_mwh": abs_error,
        "error_pct": err_pct,
        "source": sample.get("source") if sample else None,
    }


def _write_csv(path, rows):
    fieldnames = ["run", "timestamp", "predicted_mwh", "measured_mwh", "abs_error_mwh", "error_pct", "source"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run repeated inference and compute real-vs-predicted energy MAPE")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--sample-timeout", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--predicted-mwh", type=float, default=None)
    parser.add_argument("--csv", default="/data/mape_report.csv")
    args = parser.parse_args()

    status0 = _get_status(args.agent_url, args.timeout)
    predicted = _extract_predicted_mwh(status0, args.predicted_mwh)
    if predicted is None or predicted <= 0:
        raise RuntimeError("Predicted energy is missing. Provide --predicted-mwh or deploy with predicted metadata.")

    rows = []
    last_timestamp = (_latest_sample(status0) or {}).get("timestamp")
    for run_idx in range(1, max(1, min(args.runs, 200)) + 1):
        _run_predict(args.agent_url, args.timeout)
        deadline = time.time() + max(0.5, args.sample_timeout)
        sample = None
        while time.time() < deadline:
            status_now = _get_status(args.agent_url, args.timeout)
            current = _latest_sample(status_now)
            ts = current.get("timestamp") if current else None
            if current and ts and ts != last_timestamp:
                sample = current
                last_timestamp = ts
                break
            time.sleep(0.4)

        row = _compute_row(run_idx, predicted, sample)
        rows.append(row)
        print(
            f"[benchmark] run={run_idx:02d} "
            f"measured={row['measured_mwh']} "
            f"abs_err={row['abs_error_mwh']} "
            f"err_pct={row['error_pct']}"
        )
        if run_idx < args.runs:
            time.sleep(max(0.0, args.delay))

    _write_csv(args.csv, rows)
    valid = [row["error_pct"] for row in rows if row.get("error_pct") is not None]
    mape = (sum(valid) / len(valid)) if valid else None
    print(f"[benchmark] csv={args.csv}")
    print(f"[benchmark] valid_samples={len(valid)}/{len(rows)}")
    print(f"[benchmark] MAPE={mape:.4f}%" if mape is not None else "[benchmark] MAPE=N/A")


if __name__ == "__main__":
    main()
