import pandas as pd
import os
from typing import Optional


class ModelAnalyzer:
    def __init__(self, csv_path, predictor_service: Optional[object] = None):
        """Load và phân tích benchmark data"""
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values('energy_avg_mwh')
        self.predictor_service = predictor_service
    
    def get_all_models(self):
        """Trả về danh sách tất cả models với thông tin cơ bản"""
        models = []
        for _, row in self.df.iterrows():
            models.append({
                'name': row['model'],
                'params_m': round(row['params_m'], 2),
                'size_mb': round(row['size_mb'], 2),
                'energy_mwh': round(row['energy_avg_mwh'], 2),
                'latency_s': round(row['latency_avg_s'], 4),
                'throughput': round(row['throughput_iter_per_s'], 2),
                'input_resolution': row['input_resolution_actual']
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
                    'params_m': round(row['params_m'], 2),
                    'size_mb': round(row['size_mb'], 2),
                    'energy_mwh': round(row['energy_avg_mwh'], 2),
                    'latency_s': round(row['latency_avg_s'], 4),
                    'throughput': round(row['throughput_iter_per_s'], 2),
                    'input_resolution': row['input_resolution_actual'],
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
            return None
        
        row = row.iloc[0]
        return {
            'name': row['model'],
            'params_m': round(row['params_m'], 2),
            'gflops': round(row['gflops'], 2),
            'size_mb': round(row['size_mb'], 2),
            'energy_avg_mwh': round(row['energy_avg_mwh'], 2),
            'energy_std_mwh': round(row['energy_std_mwh'], 2),
            'latency_avg_s': round(row['latency_avg_s'], 4),
            'latency_std_s': round(row['latency_std_s'], 6),
            'throughput_iter_per_s': round(row['throughput_iter_per_s'], 2),
            'input_resolution': row['input_resolution_actual'],
            'input_size': row['input_size']
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
