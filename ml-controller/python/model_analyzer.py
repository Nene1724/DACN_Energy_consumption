import pandas as pd
import os
import re
from typing import Optional, Iterable, List


class ModelAnalyzer:
    def __init__(self, csv_path, predictor_service: Optional[object] = None, model_store_dir: str = None, rpi5_csv_path: str = None, extra_model_dirs: Optional[Iterable[str]] = None):
        """
        Load và phân tích benchmark data
        
        Args:
            csv_path: Path to Jetson CSV
            predictor_service: Energy prediction service
            model_store_dir: Directory containing model files
            rpi5_csv_path: Path to Raspberry Pi 5 CSV (optional)
        """
        # Load Jetson data
        self.df_jetson = pd.read_csv(csv_path)
        self.df_jetson['device'] = 'jetson_nano_2gb'
        
        # Load RPi5 data if available
        if rpi5_csv_path and os.path.exists(rpi5_csv_path):
            self.df_rpi5 = pd.read_csv(rpi5_csv_path)
            self.df_rpi5['device'] = 'raspberry_pi5'
            # Combine both datasets
            self.df = pd.concat([self.df_jetson, self.df_rpi5], ignore_index=True)
            print(f"✅ Loaded {len(self.df_jetson)} Jetson models + {len(self.df_rpi5)} RPi5 models")
        else:
            self.df = self.df_jetson.copy()
            self.df_rpi5 = None
            print(f"✅ Loaded {len(self.df_jetson)} Jetson models only")
        
        self.df = self.df.sort_values('energy_avg_mwh')
        self.predictor_service = predictor_service
        self.model_store_dir = model_store_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_store')
        self.extra_model_dirs = [d for d in (extra_model_dirs or []) if d]
        self.local_model_dirs: List[str] = []
        seen_dirs = set()
        for directory in [self.model_store_dir, *self.extra_model_dirs]:
            normalized = os.path.abspath(directory)
            if normalized in seen_dirs:
                continue
            seen_dirs.add(normalized)
            self.local_model_dirs.append(directory)

    def _iter_local_model_dirs(self):
        for directory in self.local_model_dirs:
            if os.path.isdir(directory):
                yield directory
    
    def _normalize_key(self, value: str) -> str:
        """Normalize model name for comparison"""
        return "".join(ch.lower() for ch in value if ch.isalnum())
    
    def _check_model_downloaded(self, model_name: str) -> bool:
        """Check if model artifact exists in any local model directory"""
        normalized = self._normalize_key(model_name)
        if not normalized:
            return False
        
        # Check for exact match (case-insensitive, alphanumeric only)
        for directory in self._iter_local_model_dirs():
            for filename in os.listdir(directory):
                base, _ = os.path.splitext(filename)
                if self._normalize_key(base) == normalized:
                    return True
        
        return False
    
    def get_all_models(self):
        """Trả về danh sách tất cả models bao gồm cả models trong model_store"""
        models = []
        model_names_from_csv = set()
        
        # 1. Get models from CSV benchmark data
        for _, row in self.df.iterrows():
            model_name = row['model']
            model_names_from_csv.add(self._normalize_key(model_name))
            # Use .get() with default values and handle NaN
            models.append({
                'name': model_name,
                'params_m': round(float(row.get('params_m', 0)), 2) if pd.notna(row.get('params_m')) else 0,
                'size_mb': round(float(row.get('size_mb', 0)), 2) if pd.notna(row.get('size_mb')) else 0,
                'energy_mwh': round(float(row.get('energy_avg_mwh', 0)), 2) if pd.notna(row.get('energy_avg_mwh')) else 0,
                'latency_s': round(float(row.get('latency_avg_s', 0)), 4) if pd.notna(row.get('latency_avg_s')) else 0,
                'throughput': round(float(row.get('throughput_iter_per_s', 0)), 2) if pd.notna(row.get('throughput_iter_per_s')) else 0,
                'input_resolution': str(row.get('input_resolution_actual', '')) if pd.notna(row.get('input_resolution_actual')) else '',
                'device': str(row.get('device', 'unknown')) if pd.notna(row.get('device')) else 'unknown',
                'downloaded': self._check_model_downloaded(model_name)
            })
        
        # 2. Scan local model directories and add models not in CSV
        for directory in self._iter_local_model_dirs():
            source_name = 'model_store' if os.path.abspath(directory) == os.path.abspath(self.model_store_dir) else os.path.basename(directory) or 'local'
            for filename in os.listdir(directory):
                if not filename.endswith(('.onnx', '.tflite', '.pth', '.pt', '.bin')):
                    continue
                
                base_name, ext = os.path.splitext(filename)
                normalized = self._normalize_key(base_name)
                
                # Skip if already in CSV
                if normalized in model_names_from_csv:
                    continue
                
                # Get file size
                file_path = os.path.join(directory, filename)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # Add model with minimal metadata
                models.append({
                    'name': base_name,
                    'params_m': 0,  # Unknown
                    'size_mb': round(file_size_mb, 2),
                    'energy_mwh': 0,  # Unknown - needs prediction
                    'latency_s': 0,  # Unknown
                    'throughput': 0,  # Unknown
                    'input_resolution': '',
                    'device': 'unknown',
                    'downloaded': True,
                    'is_library': True,  # Mark as from local artifact directory
                    'source': source_name
                })
        
        return models
    
    def get_recommended_models(self, device_type='BBB', max_energy_mwh=100):
        """
        Gợi ý models phù hợp dựa trên loại thiết bị
        
        BBB (BeagleBone Black):
        - RAM: 512MB
        - Năng lượng: < 100 mWh (low power)
        - Ưu tiên: Efficiency > Performance
        """
        if device_type == 'BBB':
            # Lọc models phù hợp với BBB
            # - Energy < max_energy_mwh (tiết kiệm năng lượng)
            # - Size < 100MB (vừa vặn với RAM 512MB)
            # - Latency < 0.5s (responsive)
            filtered = self.df[
                (self.df['energy_avg_mwh'] < max_energy_mwh) &
                (self.df['size_mb'] < 100) &
                (self.df['latency_avg_s'] < 0.5)
            ]
            
            # Sắp xếp theo energy (ưu tiên tiết kiệm điện nhất)
            filtered = filtered.sort_values('energy_avg_mwh')
            
            recommendations = []
            for _, row in filtered.head(10).iterrows():
                recommendations.append({
                    'name': row['model'],
                    'params_m': round(float(row.get('params_m', 0)), 2) if pd.notna(row.get('params_m')) else 0,
                    'size_mb': round(float(row.get('size_mb', 0)), 2) if pd.notna(row.get('size_mb')) else 0,
                    'energy_mwh': round(float(row.get('energy_avg_mwh', 0)), 2) if pd.notna(row.get('energy_avg_mwh')) else 0,
                    'latency_s': round(float(row.get('latency_avg_s', 0)), 4) if pd.notna(row.get('latency_avg_s')) else 0,
                    'throughput': round(float(row.get('throughput_iter_per_s', 0)), 2) if pd.notna(row.get('throughput_iter_per_s')) else 0,
                    'input_resolution': str(row.get('input_resolution_actual', '')) if pd.notna(row.get('input_resolution_actual')) else '',
                    'recommended': True,
                    'reason': self._get_recommendation_reason(row)
                })
            
            return recommendations
        
        return []
    
    def _get_recommendation_reason(self, row):
        """Giải thích tại sao model này được recommend"""
        energy = row['energy_avg_mwh']
        size = row['size_mb']
        latency = row['latency_avg_s']
        
        reasons = []
        if energy < 30:
            reasons.append("Cực kỳ tiết kiệm năng lượng")
        elif energy < 70:
            reasons.append("Tiết kiệm năng lượng tốt")
        
        if size < 30:
            reasons.append("Model nhẹ")
        elif size < 60:
            reasons.append("Kích thước vừa phải")
        
        if latency < 0.1:
            reasons.append("Rất nhanh")
        elif latency < 0.3:
            reasons.append("Tốc độ tốt")
        
        return " | ".join(reasons) if reasons else "Phù hợp với BBB"
    
    def get_model_details(self, model_name):
        """Lấy thông tin chi tiết của 1 model"""
        row = self.df[self.df['model'] == model_name]
        if row.empty:
            normalized = self._normalize_key(model_name)
            for directory in self._iter_local_model_dirs():
                for filename in os.listdir(directory):
                    base_name, _ = os.path.splitext(filename)
                    if self._normalize_key(base_name) != normalized:
                        continue
                    file_path = os.path.join(directory, filename)
                    return {
                        'name': base_name,
                        'params_m': 0,
                        'gflops': 0,
                        'gmacs': 0,
                        'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2),
                        'energy_avg_mwh': 0,
                        'energy_std_mwh': 0,
                        'latency_avg_s': 0,
                        'latency_std_s': 0,
                        'throughput_iter_per_s': 0,
                        'input_resolution': '',
                        'input_size': '',
                        'source': 'local_artifact'
                    }
            return None
        
        row = row.iloc[0]
        return {
            'name': row['model'],
            'params_m': round(float(row.get('params_m', 0)), 2) if pd.notna(row.get('params_m')) else 0,
            'gflops': round(float(row.get('gflops', 0)), 2) if pd.notna(row.get('gflops')) else 0,
            'size_mb': round(float(row.get('size_mb', 0)), 2) if pd.notna(row.get('size_mb')) else 0,
            'energy_avg_mwh': round(float(row.get('energy_avg_mwh', 0)), 2) if pd.notna(row.get('energy_avg_mwh')) else 0,
            'energy_std_mwh': round(float(row.get('energy_std_mwh', 0)), 2) if pd.notna(row.get('energy_std_mwh')) else 0,
            'latency_avg_s': round(float(row.get('latency_avg_s', 0)), 4) if pd.notna(row.get('latency_avg_s')) else 0,
            'latency_std_s': round(float(row.get('latency_std_s', 0)), 6) if pd.notna(row.get('latency_std_s')) else 0,
            'throughput_iter_per_s': round(float(row.get('throughput_iter_per_s', 0)), 2) if pd.notna(row.get('throughput_iter_per_s')) else 0,
            'input_resolution': str(row.get('input_resolution_actual', '')) if pd.notna(row.get('input_resolution_actual')) else '',
            'input_size': str(row.get('input_size', '')) if pd.notna(row.get('input_size')) else ''
        }

    def predict_energy(self, metadata: dict):
        """
        Dự đoán năng lượng cho model chưa benchmark.
        metadata cần các field: params_m, gflops, gmacs, size_mb, latency_avg_s, throughput_iter_per_s
        """
        if not self.predictor_service:
            raise RuntimeError("Predictor service chưa được khởi tạo")
        predictions = self.predictor_service.predict([metadata])
        return predictions[0] if predictions else None
    
    def get_energy_stats(self):
        """Thống kê về energy consumption"""
        return {
            'min': round(self.df['energy_avg_mwh'].min(), 2),
            'max': round(self.df['energy_avg_mwh'].max(), 2),
            'mean': round(self.df['energy_avg_mwh'].mean(), 2),
            'median': round(self.df['energy_avg_mwh'].median(), 2),
            'models_under_50mwh': len(self.df[self.df['energy_avg_mwh'] < 50]),
            'models_under_100mwh': len(self.df[self.df['energy_avg_mwh'] < 100]),
            'total_models': len(self.df)
        }
