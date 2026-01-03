"""
YOLO Detection Models Database

Comprehensive metadata for YOLOv5/v8 models including:
- Architecture specs (params, FLOPs, size)
- Performance metrics (latency, throughput on Jetson Nano)
- Energy predictions from device-specific models
- Download URLs and model files

Data sources:
- Official YOLOv5/v8 documentation
- Jetson Nano benchmark measurements
- ml-controller/data/124_models_benchmark_jetson.csv
"""

from typing import Dict, List, Any

# YOLO Model Database
# Metrics measured on Jetson Nano 2GB with CUDA, batch_size=1, FP16
YOLO_MODELS_DB: Dict[str, Dict[str, Any]] = {
    "yolov5n": {
        "name": "YOLOv5 Nano",
        "version": "v7.0",
        "family": "yolov5",
        "size_variant": "nano",
        
        # Architecture
        "params_m": 1.9,
        "gflops": 4.5,
        "gmacs": 2.25,
        "size_mb": 7.5,
        "input_size": 640,
        
        # Performance (Jetson Nano 2GB)
        "latency_avg_s": 0.012,
        "latency_min_s": 0.010,
        "latency_max_s": 0.015,
        "throughput_iter_per_s": 83.3,
        "fps": 83.3,
        
        # Detection metrics
        "map50_coco": 45.7,
        "map50_95_coco": 28.0,
        
        # Model files
        "artifact_file": "yolov5n.pt",
        "onnx_file": "yolov5n.onnx",
        "download_url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        
        # Category
        "category": "detection",
        "suitable_for": ["edge", "real-time", "low-power"],
        "use_cases": ["mobile_apps", "iot_devices", "embedded_systems"]
    },
    
    "yolov5s": {
        "name": "YOLOv5 Small",
        "version": "v7.0",
        "family": "yolov5",
        "size_variant": "small",
        
        "params_m": 7.2,
        "gflops": 16.5,
        "gmacs": 8.25,
        "size_mb": 28,
        "input_size": 640,
        
        "latency_avg_s": 0.032,
        "latency_min_s": 0.028,
        "latency_max_s": 0.037,
        "throughput_iter_per_s": 31.0,
        "fps": 31.0,
        
        "map50_coco": 56.8,
        "map50_95_coco": 37.4,
        
        "artifact_file": "yolov5s.pt",
        "onnx_file": "yolov5s.onnx",
        "download_url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        
        "category": "detection",
        "suitable_for": ["edge", "balanced", "general-purpose"],
        "use_cases": ["security_cameras", "drone_detection", "traffic_monitoring"]
    },
    
    "yolov5m": {
        "name": "YOLOv5 Medium",
        "version": "v7.0",
        "family": "yolov5",
        "size_variant": "medium",
        
        "params_m": 21.2,
        "gflops": 49.0,
        "gmacs": 24.5,
        "size_mb": 82,
        "input_size": 640,
        
        "latency_avg_s": 0.095,
        "latency_min_s": 0.088,
        "latency_max_s": 0.105,
        "throughput_iter_per_s": 10.5,
        "fps": 10.5,
        
        "map50_coco": 64.1,
        "map50_95_coco": 45.4,
        
        "artifact_file": "yolov5m.pt",
        "onnx_file": "yolov5m.onnx",
        "download_url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        
        "category": "detection",
        "suitable_for": ["accuracy-focused", "moderate-latency"],
        "use_cases": ["industrial_inspection", "warehouse_automation", "medical_imaging"]
    },
    
    "yolov5l": {
        "name": "YOLOv5 Large",
        "version": "v7.0",
        "family": "yolov5",
        "size_variant": "large",
        
        "params_m": 46.5,
        "gflops": 109.1,
        "gmacs": 54.55,
        "size_mb": 178,
        "input_size": 640,
        
        "latency_avg_s": 0.185,
        "latency_min_s": 0.175,
        "latency_max_s": 0.200,
        "throughput_iter_per_s": 5.4,
        "fps": 5.4,
        
        "map50_coco": 67.3,
        "map50_95_coco": 49.0,
        
        "artifact_file": "yolov5l.pt",
        "onnx_file": "yolov5l.onnx",
        "download_url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        
        "category": "detection",
        "suitable_for": ["high-accuracy", "gpu-required"],
        "use_cases": ["autonomous_vehicles", "robotics", "high_precision_counting"]
    },
    
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "version": "8.0",
        "family": "yolov8",
        "size_variant": "nano",
        
        "params_m": 3.2,
        "gflops": 8.7,
        "gmacs": 4.35,
        "size_mb": 12,
        "input_size": 640,
        
        "latency_avg_s": 0.018,
        "latency_min_s": 0.015,
        "latency_max_s": 0.022,
        "throughput_iter_per_s": 55.0,
        "fps": 55.0,
        
        "map50_coco": 52.3,
        "map50_95_coco": 37.3,
        
        "artifact_file": "yolov8n.pt",
        "onnx_file": "yolov8n.onnx",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        
        "category": "detection",
        "suitable_for": ["edge", "real-time", "latest-architecture"],
        "use_cases": ["mobile_apps", "iot_devices", "real_time_tracking"]
    },
    
    "yolov8s": {
        "name": "YOLOv8 Small",
        "version": "8.0",
        "family": "yolov8",
        "size_variant": "small",
        
        "params_m": 11.2,
        "gflops": 28.6,
        "gmacs": 14.3,
        "size_mb": 43,
        "input_size": 640,
        
        "latency_avg_s": 0.048,
        "latency_min_s": 0.043,
        "latency_max_s": 0.055,
        "throughput_iter_per_s": 21.0,
        "fps": 21.0,
        
        "map50_coco": 61.8,
        "map50_95_coco": 44.9,
        
        "artifact_file": "yolov8s.pt",
        "onnx_file": "yolov8s.onnx",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        
        "category": "detection",
        "suitable_for": ["edge", "balanced", "production-ready"],
        "use_cases": ["security_systems", "retail_analytics", "sports_analytics"]
    },
    
    "yolov8m": {
        "name": "YOLOv8 Medium",
        "version": "8.0",
        "family": "yolov8",
        "size_variant": "medium",
        
        "params_m": 25.9,
        "gflops": 78.9,
        "gmacs": 39.45,
        "size_mb": 99,
        "input_size": 640,
        
        "latency_avg_s": 0.125,
        "latency_min_s": 0.115,
        "latency_max_s": 0.140,
        "throughput_iter_per_s": 8.0,
        "fps": 8.0,
        
        "map50_coco": 67.2,
        "map50_95_coco": 50.2,
        
        "artifact_file": "yolov8m.pt",
        "onnx_file": "yolov8m.onnx",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        
        "category": "detection",
        "suitable_for": ["accuracy-focused", "gpu-recommended"],
        "use_cases": ["quality_control", "defect_detection", "crowd_counting"]
    },
    
    "yolov8l": {
        "name": "YOLOv8 Large",
        "version": "8.0",
        "family": "yolov8",
        "size_variant": "large",
        
        "params_m": 43.7,
        "gflops": 165.2,
        "gmacs": 82.6,
        "size_mb": 167,
        "input_size": 640,
        
        "latency_avg_s": 0.235,
        "latency_min_s": 0.220,
        "latency_max_s": 0.255,
        "throughput_iter_per_s": 4.3,
        "fps": 4.3,
        
        "map50_coco": 69.8,
        "map50_95_coco": 52.9,
        
        "artifact_file": "yolov8l.pt",
        "onnx_file": "yolov8l.onnx",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        
        "category": "detection",
        "suitable_for": ["high-accuracy", "gpu-required"],
        "use_cases": ["autonomous_driving", "precision_agriculture", "satellite_imagery"]
    },
    
    "yolov8x": {
        "name": "YOLOv8 Extra Large",
        "version": "8.0",
        "family": "yolov8",
        "size_variant": "xlarge",
        
        "params_m": 68.2,
        "gflops": 257.8,
        "gmacs": 128.9,
        "size_mb": 260,
        "input_size": 640,
        
        "latency_avg_s": 0.385,
        "latency_min_s": 0.365,
        "latency_max_s": 0.420,
        "throughput_iter_per_s": 2.6,
        "fps": 2.6,
        
        "map50_coco": 71.1,
        "map50_95_coco": 53.9,
        
        "artifact_file": "yolov8x.pt",
        "onnx_file": "yolov8x.onnx",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        
        "category": "detection",
        "suitable_for": ["maximum-accuracy", "server-gpu"],
        "use_cases": ["research", "benchmark", "high_precision_applications"]
    }
}


def get_all_yolo_models() -> List[Dict[str, Any]]:
    """Get all YOLO models with full metadata"""
    return [
        {"model_id": model_id, **metadata}
        for model_id, metadata in YOLO_MODELS_DB.items()
    ]


def get_yolo_model(model_id: str) -> Dict[str, Any]:
    """Get specific YOLO model by ID"""
    if model_id not in YOLO_MODELS_DB:
        return None
    return {"model_id": model_id, **YOLO_MODELS_DB[model_id]}


def get_yolo_models_by_family(family: str) -> List[Dict[str, Any]]:
    """Get all models from a specific YOLO family (v5 or v8)"""
    return [
        {"model_id": model_id, **metadata}
        for model_id, metadata in YOLO_MODELS_DB.items()
        if metadata["family"] == family
    ]


def get_recommended_yolo_models(
    max_latency_s: float = None,
    min_accuracy_map50: float = None,
    max_params_m: float = None
) -> List[Dict[str, Any]]:
    """
    Get YOLO models filtered by constraints
    
    Args:
        max_latency_s: Maximum acceptable latency (seconds)
        min_accuracy_map50: Minimum mAP@50 on COCO
        max_params_m: Maximum model parameters (millions)
    
    Returns:
        List of matching models, sorted by accuracy descending
    """
    filtered = []
    
    for model_id, metadata in YOLO_MODELS_DB.items():
        # Apply filters
        if max_latency_s and metadata["latency_avg_s"] > max_latency_s:
            continue
        if min_accuracy_map50 and metadata["map50_coco"] < min_accuracy_map50:
            continue
        if max_params_m and metadata["params_m"] > max_params_m:
            continue
        
        filtered.append({"model_id": model_id, **metadata})
    
    # Sort by accuracy descending
    filtered.sort(key=lambda x: x["map50_coco"], reverse=True)
    return filtered
