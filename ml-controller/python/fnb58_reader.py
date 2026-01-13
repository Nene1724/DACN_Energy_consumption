"""
FNB58 USB Tester Reader Module

Hỗ trợ đọc dữ liệu từ FNB58 (hoặc FNB48) qua USB serial.
FNB58 gửi dữ liệu V/A/W/Wh theo định kỳ (thường 1-2 lần/giây).

Format điển hình từ FNB58:
U:5.10V I:2.85A P:14.54W E:123.45Wh

Để dùng:
1. Kết nối FNB58 qua USB vào máy/thiết bị
2. Tìm cổng serial: /dev/ttyUSB0 (Linux) hoặc COM3 (Windows)
3. Chạy: reader = FNB58Reader('/dev/ttyUSB0')
         reader.start()
         sleep(60)  # đợi 60 giây
         data = reader.stop()
         print(data['total_energy_wh'])
"""

import serial
import threading
import time
import re
import os
import sys
import subprocess
from typing import Optional, Dict, Any


class FNB58Reader:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        """
        Khởi tạo reader FNB58.
        
        Args:
            port: Cổng serial, ví dụ '/dev/ttyUSB0' (Linux) hoặc 'COM3' (Windows)
            baudrate: Tốc độ baud (mặc định 9600)
            timeout: Timeout đọc serial (giây)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.running = False
        self.thread = None
        
        # Lưu dữ liệu
        self.samples = []
        self.lock = threading.RLock()
        
        # Trạng thái
        self.last_values = {
            "voltage_v": None,
            "current_a": None,
            "power_w": None,
            "energy_wh": None
        }
        self.connection_error = None

    def connect(self) -> bool:
        """Kết nối tới cổng serial FNB58. Trả về True nếu thành công."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(0.5)  # Đợi cài đặt ổn định
            return True
        except Exception as e:
            self.connection_error = str(e)
            print(f"[FNB58] Lỗi kết nối {self.port}: {e}")
            return False

    def disconnect(self):
        """Ngắt kết nối serial."""
        if self.serial and self.serial.is_open:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None

    def _parse_fnb58_line(self, line: str) -> Optional[Dict[str, float]]:
        """
        Parse một dòng dữ liệu từ FNB58.
        
        Format: "U:5.10V I:2.85A P:14.54W E:123.45Wh"
        
        Trả về dict với keys: voltage_v, current_a, power_w, energy_wh
        Trả về None nếu parse lỗi.
        """
        line = line.strip()
        if not line:
            return None
        
        data = {}
        
        # Regex để lấy U (voltage)
        u_match = re.search(r'U[:\s]+(\d+\.?\d*)\s*V', line, re.IGNORECASE)
        if u_match:
            data['voltage_v'] = float(u_match.group(1))
        
        # Regex để lấy I (current)
        i_match = re.search(r'I[:\s]+(\d+\.?\d*)\s*A', line, re.IGNORECASE)
        if i_match:
            data['current_a'] = float(i_match.group(1))
        
        # Regex để lấy P (power)
        p_match = re.search(r'P[:\s]+(\d+\.?\d*)\s*W(?!h)', line, re.IGNORECASE)
        if p_match:
            data['power_w'] = float(p_match.group(1))
        
        # Regex để lấy E (energy)
        e_match = re.search(r'E[:\s]+(\d+\.?\d*)\s*Wh', line, re.IGNORECASE)
        if e_match:
            data['energy_wh'] = float(e_match.group(1))
        
        return data if data else None

    def _read_loop(self):
        """Thread loop để đọc dữ liệu liên tục từ FNB58."""
        if not self.serial:
            return
        
        while self.running:
            try:
                if self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8', errors='ignore')
                    parsed = self._parse_fnb58_line(line)
                    if parsed:
                        with self.lock:
                            self.samples.append({
                                'timestamp': time.time(),
                                'data': parsed
                            })
                            self.last_values.update(parsed)
            except Exception as e:
                print(f"[FNB58] Lỗi đọc: {e}")
                break
        
        self.disconnect()

    def start(self) -> bool:
        """Bắt đầu đọc FNB58. Trả về True nếu thành công."""
        if not self.connect():
            return False
        
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> Dict[str, Any]:
        """Dừng đọc và trả về kết quả tổng hợp."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.disconnect()
        
        with self.lock:
            result = {
                'success': len(self.samples) > 0,
                'samples_count': len(self.samples),
                'error': self.connection_error,
                'samples': self.samples
            }
            
            # Tính năng lượng cuối cùng (Wh)
            if self.last_values.get('energy_wh') is not None:
                result['total_energy_wh'] = self.last_values['energy_wh']
                result['total_energy_mwh'] = self.last_values['energy_wh'] * 1000.0
            else:
                result['total_energy_wh'] = None
                result['total_energy_mwh'] = None
            
            # Công suất trung bình (W)
            if self.samples:
                powers = [s['data'].get('power_w') for s in self.samples if s['data'].get('power_w') is not None]
                if powers:
                    result['avg_power_w'] = sum(powers) / len(powers)
                    result['avg_power_mw'] = result['avg_power_w'] * 1000.0
                else:
                    result['avg_power_w'] = None
                    result['avg_power_mw'] = None
            else:
                result['avg_power_w'] = None
                result['avg_power_mw'] = None
            
            result['last_values'] = dict(self.last_values)
        
        return result


def detect_fnb58_port() -> Optional[str]:
    """
    Tự động phát hiện cổng serial của FNB58.
    Quét tất cả cổng COM và tìm cổng có FNB58.
    
    Trả về tên cổng (ví dụ '/dev/ttyUSB0') hoặc None.
    """
    try:
        import serial.tools.list_ports
    except ImportError:
        print("[FNB58] Cài pyserial: pip install pyserial")
        return None
    
    for port_info in serial.tools.list_ports.comports():
        port = port_info.device
        try:
            ser = serial.Serial(port, baudrate=9600, timeout=0.5)
            time.sleep(0.2)
            
            # Thử đọc 1-2 dòng
            data = b''
            while ser.in_waiting > 0:
                data += ser.read(100)
            
            ser.close()
            
            # Kiểm tra có pattern FNB58 không (U:...V I:...A)
            text = data.decode('utf-8', errors='ignore')
            if 'U:' in text and 'I:' in text and 'V' in text and 'A' in text:
                print(f"[FNB58] Phát hiện trên {port}")
                return port
        except Exception:
            pass
    
    return None


def grant_port_permission(port: str) -> bool:
    """
    Cấp quyền truy cập cổng serial (tự động theo hệ điều hành).
    
    Linux:
    - Thêm user vào group 'dialout'
    - Hoặc chmod 666 cổng
    
    Windows:
    - Không cần (bình thường đã có quyền)
    
    Args:
        port: Tên cổng, ví dụ '/dev/ttyUSB0'
    
    Returns:
        True nếu thành công hoặc không cần, False nếu lỗi.
    """
    import platform
    
    system = platform.system()
    
    if system == "Windows":
        # Windows thường không cần cấp quyền cho COM port
        print(f"[FNB58] Windows - không cần cấp quyền")
        return True
    
    elif system == "Linux":
        # Cách 1: chmod 666 (nhanh, tạm thời, mất hiệu lực sau reboot)
        try:
            os.chmod(port, 0o666)
            print(f"[FNB58] chmod 666 {port} ✓")
            return True
        except Exception as e:
            print(f"[FNB58] chmod lỗi: {e}")
        
        # Cách 2: Thêm user hiện tại vào group 'dialout' (vĩnh viễn, cần reboot/logout-login)
        try:
            username = os.getenv("USER") or "root"
            print(f"[FNB58] Thêm {username} vào group dialout...")
            
            # Kiểm tra nếu đã ở trong group dialout
            result = subprocess.run(["groups", username], capture_output=True, text=True)
            if "dialout" in result.stdout:
                print(f"[FNB58] {username} đã ở trong group dialout")
                return True
            
            # Thêm vào group (cần sudo)
            cmd = f"sudo usermod -a -G dialout {username}"
            print(f"[FNB58] Chạy: {cmd}")
            subprocess.run(cmd, shell=True, check=False)
            print(f"[FNB58] ⚠️  Cần logout/login hoặc reboot để áp dụng")
            return True
        except Exception as e:
            print(f"[FNB58] Lỗi thêm group: {e}")
            return False
    
    elif system == "Darwin":  # macOS
        print(f"[FNB58] macOS - thường không cần cấp quyền")
        return True
    
    print(f"[FNB58] OS không hỗ trợ: {system}")
    return False


if __name__ == "__main__":
    # Test đơn giản
    print("[FNB58] Phát hiện cổng...")
    port = detect_fnb58_port()
    
    if not port:
        print("[FNB58] Không tìm thấy. Thử /dev/ttyUSB0 trên Linux hoặc COM3 trên Windows")
        port = "/dev/ttyUSB0"
    
    print(f"[FNB58] Kết nối {port}...")
    reader = FNB58Reader(port)
    
    if reader.start():
        print("[FNB58] Đọc trong 10 giây...")
        time.sleep(10)
        result = reader.stop()
        
        print(f"\n=== Kết Quả ===")
        print(f"Số mẫu: {result['samples_count']}")
        print(f"Năng lượng: {result['total_energy_wh']:.2f} Wh = {result['total_energy_mwh']:.1f} mWh")
        print(f"Công suất TB: {result['avg_power_w']:.2f} W = {result['avg_power_mw']:.1f} mW")
        print(f"Giá trị cuối: {result['last_values']}")
    else:
        print(f"[FNB58] Lỗi kết nối: {reader.connection_error}")
