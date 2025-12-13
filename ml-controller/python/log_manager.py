"""
Log Manager - Quản lý deployment logs persistent
Lưu logs vào JSON file để giữ lại lịch sử khi reload trang
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional


class LogManager:
    """Quản lý deployment logs với file JSON backend"""
    
    def __init__(self, log_file_path: str, max_logs: int = 500):
        """
        Args:
            log_file_path: Đường dẫn tới file JSON lưu logs
            max_logs: Số lượng logs tối đa giữ lại (FIFO khi vượt quá)
        """
        self.log_file_path = log_file_path
        self.max_logs = max_logs
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Tạo file logs nếu chưa tồn tại"""
        if not os.path.exists(self.log_file_path):
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            initial_data = {
                "logs": [],
                "metadata": {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                    "total_deployments": 0
                }
            }
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    def _read_logs(self) -> Dict:
        """Đọc toàn bộ logs từ file"""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading logs: {e}")
            return {
                "logs": [],
                "metadata": {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "last_updated": datetime.utcnow().isoformat() + "Z",
                    "total_deployments": 0
                }
            }
    
    def _write_logs(self, data: Dict):
        """Ghi logs vào file"""
        try:
            data["metadata"]["last_updated"] = datetime.utcnow().isoformat() + "Z"
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing logs: {e}")
    
    def add_log(self, log_type: str, message: str, metadata: Optional[Dict] = None):
        """
        Thêm log entry mới
        
        Args:
            log_type: Loại log (info, success, error, warning)
            message: Nội dung log
            metadata: Thông tin bổ sung (model_name, device_ip, energy, etc.)
        """
        data = self._read_logs()
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": log_type,
            "message": message,
            "metadata": metadata or {}
        }
        
        data["logs"].append(log_entry)
        
        # Giới hạn số lượng logs (FIFO)
        if len(data["logs"]) > self.max_logs:
            data["logs"] = data["logs"][-self.max_logs:]
        
        # Update metadata
        if log_type == "success" and metadata and "model_name" in metadata:
            data["metadata"]["total_deployments"] += 1
        
        self._write_logs(data)
        return log_entry
    
    def get_logs(self, limit: Optional[int] = None, log_type: Optional[str] = None) -> List[Dict]:
        """
        Lấy danh sách logs
        
        Args:
            limit: Số lượng logs tối đa trả về (mới nhất)
            log_type: Lọc theo loại log (info, success, error, warning)
        
        Returns:
            List of log entries
        """
        data = self._read_logs()
        logs = data["logs"]
        
        # Filter by type
        if log_type:
            logs = [log for log in logs if log.get("type") == log_type]
        
        # Reverse để mới nhất lên đầu
        logs = list(reversed(logs))
        
        # Limit
        if limit:
            logs = logs[:limit]
        
        return logs
    
    def get_recent_logs(self, minutes: int = 60) -> List[Dict]:
        """Lấy logs trong N phút gần đây"""
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        cutoff_iso = cutoff_time.isoformat() + "Z"
        
        data = self._read_logs()
        recent_logs = [
            log for log in data["logs"]
            if log.get("timestamp", "") >= cutoff_iso
        ]
        
        return list(reversed(recent_logs))
    
    def get_deployment_stats(self) -> Dict:
        """Lấy thống kê deployment"""
        data = self._read_logs()
        logs = data["logs"]
        
        total = len(logs)
        success_count = len([l for l in logs if l.get("type") == "success"])
        error_count = len([l for l in logs if l.get("type") == "error"])
        
        # Models deployed
        deployed_models = set()
        for log in logs:
            if log.get("type") == "success" and log.get("metadata", {}).get("model_name"):
                deployed_models.add(log["metadata"]["model_name"])
        
        return {
            "total_logs": total,
            "total_deployments": data["metadata"].get("total_deployments", 0),
            "success_count": success_count,
            "error_count": error_count,
            "unique_models_deployed": len(deployed_models),
            "deployed_models": list(deployed_models),
            "last_updated": data["metadata"].get("last_updated")
        }
    
    def clear_logs(self):
        """Xóa toàn bộ logs (giữ lại metadata)"""
        data = self._read_logs()
        data["logs"] = []
        data["metadata"]["total_deployments"] = 0
        self._write_logs(data)
    
    def export_logs(self, output_path: str, format: str = "json"):
        """
        Export logs ra file khác
        
        Args:
            output_path: Đường dẫn file output
            format: Định dạng (json, csv)
        """
        data = self._read_logs()
        
        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if not data["logs"]:
                    return
                
                fieldnames = ["timestamp", "type", "message"]
                # Add metadata fields
                all_metadata_keys = set()
                for log in data["logs"]:
                    all_metadata_keys.update(log.get("metadata", {}).keys())
                fieldnames.extend(sorted(all_metadata_keys))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for log in data["logs"]:
                    row = {
                        "timestamp": log.get("timestamp"),
                        "type": log.get("type"),
                        "message": log.get("message")
                    }
                    # Add metadata as columns
                    for key in all_metadata_keys:
                        row[key] = log.get("metadata", {}).get(key, "")
                    writer.writerow(row)
