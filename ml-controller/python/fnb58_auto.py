#!/usr/bin/env python3
"""
FNB58 Auto Measurement Script

Tự động:
1. Phát hiện cổng USB của FNB58
2. Cấp quyền truy cập
3. Trigger agent để đo năng lượng
4. Xem kết quả so sánh trên server

Sử dụng:
    python fnb58_auto.py                    # Chạy với config mặc định
    python fnb58_auto.py --agent-ip 192.168.1.50 --duration 60
    python fnb58_auto.py --port /dev/ttyUSB0 --server http://localhost:5000
"""

import argparse
import sys
import time
import requests
import json
from fnb58_reader import FNB58Reader, detect_fnb58_port, grant_port_permission


def parse_args():
    parser = argparse.ArgumentParser(
        description="FNB58 Auto Measurement - Phát hiện cổng, cấp quyền, đo năng lượng"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Cổng serial FNB58 (ví dụ /dev/ttyUSB0). Nếu không có, tự động phát hiện."
    )
    parser.add_argument(
        "--agent-ip",
        type=str,
        default="localhost",
        help="IP hoặc hostname của agent (mặc định: localhost)"
    )
    parser.add_argument(
        "--agent-port",
        type=int,
        default=8000,
        help="Port của agent (mặc định: 8000)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:5000",
        help="URL server controller (mặc định: http://localhost:5000)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Thời gian đo (giây, mặc định: 30)"
    )
    parser.add_argument(
        "--skip-permission",
        action="store_true",
        help="Bỏ qua cấp quyền (nếu đã có quyền)"
    )
    parser.add_argument(
        "--local-measure",
        action="store_true",
        help="Chỉ đo FNB58 cục bộ, không trigger agent"
    )
    parser.add_argument(
        "--post-server",
        action="store_true",
        default=True,
        help="Auto-post kết quả về server (mặc định: True)"
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="jetson_nano",
        help="Loại thiết bị khi post về server (mặc định: jetson_nano)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Tên model khi post về server (mặc định: None)"
    )
    return parser.parse_args()


def detect_and_grant(port: str = None) -> str:
    """Phát hiện cổng FNB58 và cấp quyền."""
    if port:
        print(f"[AUTO] Sử dụng cổng được chỉ định: {port}")
    else:
        print(f"[AUTO] Tìm kiếm cổng FNB58...")
        port = detect_fnb58_port()
        if not port:
            print("[ERROR] Không tìm thấy FNB58. Hãy:")
            print("  1. Kiểm tra FNB58 có kết nối USB không")
            print("  2. Chỉ định cổng thủ công: --port /dev/ttyUSB0")
            sys.exit(1)
    
    print(f"[AUTO] Tìm thấy FNB58 trên: {port}")
    print(f"[AUTO] Cấp quyền truy cập...")
    if grant_port_permission(port):
        print(f"[AUTO] ✓ Cấp quyền thành công")
    else:
        print(f"[AUTO] ⚠️  Cảnh báo cấp quyền, cố gắng tiếp tục...")
    
    return port


def measure_local(port: str, duration: float) -> dict:
    """Đo FNB58 cục bộ."""
    print(f"[MEASURE] Bắt đầu đo FNB58 trong {duration}s...")
    reader = FNB58Reader(port)
    
    if not reader.start():
        print(f"[ERROR] Lỗi kết nối FNB58: {reader.connection_error}")
        sys.exit(1)
    
    try:
        for i in range(int(duration)):
            remaining = duration - i
            print(f"[MEASURE] Đang đo... {remaining:.1f}s còn lại", end="\r")
            time.sleep(1)
        print()
    except KeyboardInterrupt:
        print(f"\n[MEASURE] Dừng sớm (Ctrl+C)")
    
    result = reader.stop()
    
    if not result.get('success'):
        print(f"[ERROR] Đo lỗi: {result.get('error')}")
        sys.exit(1)
    
    print(f"[MEASURE] ✓ Đo xong")
    print(f"  - Số mẫu: {result['samples_count']}")
    print(f"  - Năng lượng: {result['total_energy_wh']:.2f} Wh = {result['total_energy_mwh']:.1f} mWh")
    print(f"  - Công suất TB: {result['avg_power_w']:.2f} W = {result['avg_power_mw']:.1f} mW")
    print(f"  - Giá trị cuối: V={result['last_values'].get('voltage_v')}V, "
          f"I={result['last_values'].get('current_a')}A")
    
    return result


def measure_via_agent(agent_ip: str, agent_port: int, server: str, duration: float, 
                      device_type: str = "jetson_nano", model_name: str = None) -> dict:
    """Trigger agent để đo qua FNB58 bằng endpoint /measure_energy_fnb58."""
    agent_url = f"http://{agent_ip}:{agent_port}"
    endpoint = f"{agent_url}/measure_energy_fnb58"
    
    payload = {
        "duration_s": duration,
        "auto_detect": True,
        "controller_url": server
    }
    
    print(f"[AGENT] Gửi request tới agent: {endpoint}")
    print(f"[AGENT] Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(endpoint, json=payload, timeout=duration + 20)
        response.raise_for_status()
        result = response.json()
        
        if not result.get('success'):
            print(f"[ERROR] Agent lỗi: {result.get('error')}")
            sys.exit(1)
        
        print(f"[AGENT] ✓ Agent đo xong")
        print(f"  - Cổng: {result.get('port')}")
        print(f"  - Số mẫu: {result.get('samples_count')}")
        print(f"  - Năng lượng: {result.get('actual_energy_mwh'):.1f} mWh")
        print(f"  - Công suất TB: {result.get('avg_power_mw'):.1f} mW")
        
        if result.get('posted_to_controller'):
            print(f"[AGENT] ✓ Auto-posted về server")
        elif result.get('post_warning'):
            print(f"[AGENT] ⚠️  {result.get('post_warning')}")
        
        return result
    
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Không kết nối được agent: {agent_url}")
        print(f"  - Kiểm tra agent có đang chạy không")
        print(f"  - Kiểm tra IP/port đúng không")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Lỗi gọi agent: {e}")
        sys.exit(1)


def post_to_server(result: dict, server: str, device_type: str, model_name: str = None) -> bool:
    """Post kết quả về server."""
    if not result.get('total_energy_mwh') and not result.get('actual_energy_mwh'):
        print(f"[SERVER] ⚠️  Không có năng lượng để post")
        return False
    
    energy_mwh = result.get('actual_energy_mwh') or result.get('total_energy_mwh')
    payload = {
        "device_type": device_type,
        "model_name": model_name,
        "actual_energy_mwh": energy_mwh,
        "avg_power_mw": result.get('avg_power_mw'),
        "duration_s": result.get('duration_s', 0),
        "sensor_type": "fnb58"
    }
    
    print(f"[SERVER] Post kết quả về {server}/api/energy/report...")
    
    try:
        response = requests.post(f"{server}/api/energy/report", json=payload, timeout=10)
        response.raise_for_status()
        
        print(f"[SERVER] ✓ Post thành công")
        return True
    
    except Exception as e:
        print(f"[SERVER] ⚠️  Lỗi post: {e}")
        return False


def view_recent_comparisons(server: str, n: int = 5):
    """Xem n bản ghi so sánh gần nhất."""
    print(f"\n[RESULT] Lấy {n} bản ghi so sánh gần nhất từ server...")
    
    try:
        response = requests.get(f"{server}/api/energy/recent?n={n}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('success'):
            print(f"[ERROR] Server lỗi: {data.get('error')}")
            return
        
        items = data.get('items', [])
        if not items:
            print(f"[RESULT] Chưa có bản ghi nào")
            return
        
        print(f"\n{'Timestamp':<30} {'Model':<25} {'Thực (mWh)':<12} {'Dự đoán':<12} {'Sai số %':<10}")
        print("=" * 95)
        
        for item in items:
            ts = item.get('timestamp', '')[:19]  # YYYY-MM-DD HH:MM:SS
            model = item.get('model_name', 'N/A')[:24]
            actual = item.get('actual_energy_mwh')
            predicted = item.get('predicted_mwh')
            pct_error = item.get('pct_error')
            
            actual_str = f"{actual:.1f}" if actual else "N/A"
            predicted_str = f"{predicted:.1f}" if predicted else "N/A"
            error_str = f"{pct_error:.1f}%" if pct_error else "N/A"
            
            print(f"{ts:<30} {model:<25} {actual_str:<12} {predicted_str:<12} {error_str:<10}")
    
    except Exception as e:
        print(f"[ERROR] Lỗi xem kết quả: {e}")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("FNB58 AUTO MEASUREMENT SCRIPT")
    print("=" * 80)
    print(f"Cấu hình:")
    print(f"  - Port: {args.port or 'auto-detect'}")
    print(f"  - Agent: {args.agent_ip}:{args.agent_port}")
    print(f"  - Server: {args.server}")
    print(f"  - Thời gian đo: {args.duration}s")
    print(f"  - Mode: {'Local' if args.local_measure else 'Via Agent'}")
    print("=" * 80)
    print()
    
    # Bước 1: Phát hiện và cấp quyền cổng
    port = detect_and_grant(args.port)
    print()
    
    # Bước 2: Đo
    if args.local_measure:
        # Chỉ đo cục bộ
        result = measure_local(port, args.duration)
    else:
        # Trigger agent
        result = measure_via_agent(
            args.agent_ip, args.agent_port, args.server,
            args.duration, args.device_type, args.model_name
        )
    
    print()
    
    # Bước 3: Post về server (nếu cần)
    if args.post_server and result.get('success'):
        post_to_server(result, args.server, args.device_type, args.model_name)
        print()
    
    # Bước 4: Xem kết quả
    view_recent_comparisons(args.server, n=5)
    
    print("\n" + "=" * 80)
    print("XONG!")
    print("=" * 80)


if __name__ == "__main__":
    main()
