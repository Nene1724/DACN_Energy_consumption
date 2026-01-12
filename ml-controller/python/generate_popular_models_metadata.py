"""
Script to generate popular_models_metadata.json using trained energy prediction models.
Predicts energy consumption for popular edge-friendly models on both Jetson Nano and RPi5.
"""

import json
import os
from energy_predictor_service import EnergyPredictorService

# Define popular models with REAL specs from PyTorch/TensorFlow model zoo
POPULAR_MODELS = [
    # EXCELLENT - Very lightweight (should be < P25)
    {
        "name": "mobilenetv3_small_050",
        "display_name": "MobileNetV3 Small 0.5x",
        "family": "MobileNet",
        "params_m": 1.53,
        "gflops": 0.024,
        "gmacs": 0.012,
        "size_mb": 6.1,
        "latency_avg_s": 0.008,
        "throughput_iter_per_s": 125.0,
        "input_resolution": "224x224",
        "description": "Siêu nhẹ, phù hợp cho thiết bị low-power",
        "recommended_devices": ["beaglebone", "raspberry-pi-zero", "esp32"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["lightweight", "low-power", "real-time"]
    },
    {
        "name": "edgenext_xx_small",
        "display_name": "EdgeNeXt XX-Small",
        "family": "EdgeNeXt",
        "params_m": 1.33,
        "gflops": 0.266,
        "gmacs": 0.133,
        "size_mb": 5.2,
        "latency_avg_s": 0.02,
        "throughput_iter_per_s": 50.0,
        "input_resolution": "256x256",
        "description": "ConvNeXt cho edge, cực kỳ nhẹ",
        "recommended_devices": ["raspberry-pi-5", "jetson-nano"],
        "url": "https://github.com/mmaaz60/EdgeNeXt",
        "tags": ["modern", "lightweight", "efficient"]
    },
    {
        "name": "squeezenet1_0",
        "display_name": "SqueezeNet 1.0",
        "family": "SqueezeNet",
        "params_m": 1.25,
        "gflops": 0.83,
        "gmacs": 0.415,
        "size_mb": 5.0,
        "latency_avg_s": 0.018,
        "throughput_iter_per_s": 55.6,
        "input_resolution": "224x224",
        "description": "SqueezeNet gốc, rất nhỏ gọn",
        "recommended_devices": ["raspberry-pi-4", "raspberry-pi-5", "jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "compact"]
    },
    
    # GOOD - Lightweight (should be P25-P50)
    {
        "name": "mobilenetv3_small_100",
        "display_name": "MobileNetV3 Small 1.0x",
        "family": "MobileNet",
        "params_m": 2.54,
        "gflops": 0.057,
        "gmacs": 0.028,
        "size_mb": 10.1,
        "latency_avg_s": 0.028,
        "throughput_iter_per_s": 35.7,
        "input_resolution": "224x224",
        "description": "MobileNetV3 Small đầy đủ",
        "recommended_devices": ["raspberry-pi-4", "raspberry-pi-5", "jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["standard", "efficient"]
    },
    {
        "name": "mobilenetv2_100",
        "display_name": "MobileNetV2 1.0x",
        "family": "MobileNet",
        "params_m": 3.5,
        "gflops": 0.3,
        "gmacs": 0.15,
        "size_mb": 14.0,
        "latency_avg_s": 0.035,
        "throughput_iter_per_s": 28.6,
        "input_resolution": "224x224",
        "description": "MobileNetV2 chuẩn",
        "recommended_devices": ["raspberry-pi-5", "jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["standard", "balanced"]
    },
    {
        "name": "shufflenet_v2_x0_5",
        "display_name": "ShuffleNetV2 0.5x",
        "family": "ShuffleNet",
        "params_m": 1.37,
        "gflops": 0.041,
        "gmacs": 0.02,
        "size_mb": 5.5,
        "latency_avg_s": 0.032,
        "throughput_iter_per_s": 31.25,
        "input_resolution": "224x224",
        "description": "ShuffleNet nhẹ, latency trung bình",
        "recommended_devices": ["raspberry-pi-5", "jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["lightweight", "fast"]
    },
    
    # ACCEPTABLE - Medium (should be P50-P75)
    {
        "name": "mobilenetv3_large_100",
        "display_name": "MobileNetV3 Large 1.0x",
        "family": "MobileNet",
        "params_m": 5.48,
        "gflops": 0.219,
        "gmacs": 0.109,
        "size_mb": 21.9,
        "latency_avg_s": 0.055,
        "throughput_iter_per_s": 18.2,
        "input_resolution": "224x224",
        "description": "MobileNetV3 Large đầy đủ",
        "recommended_devices": ["jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["standard", "accurate"]
    },
    {
        "name": "efficientnet_lite0",
        "display_name": "EfficientNet-Lite0",
        "family": "EfficientNet",
        "params_m": 4.65,
        "gflops": 0.39,
        "gmacs": 0.195,
        "size_mb": 18.5,
        "latency_avg_s": 0.065,
        "throughput_iter_per_s": 15.4,
        "input_resolution": "224x224",
        "description": "EfficientNet cho edge",
        "recommended_devices": ["jetson-nano"],
        "url": "https://github.com/tensorflow/tpu",
        "tags": ["efficient", "accurate"]
    },
    {
        "name": "resnet18",
        "display_name": "ResNet-18",
        "family": "ResNet",
        "params_m": 11.69,
        "gflops": 1.82,
        "gmacs": 0.91,
        "size_mb": 46.8,
        "latency_avg_s": 0.085,
        "throughput_iter_per_s": 11.8,
        "input_resolution": "224x224",
        "description": "ResNet nhỏ nhất",
        "recommended_devices": ["jetson-nano"],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "accurate"]
    },
    
    # HIGH - Heavy (should be > P75) - NOT RECOMMENDED
    {
        "name": "efficientnet_b0",
        "display_name": "EfficientNet-B0",
        "family": "EfficientNet",
        "params_m": 5.29,
        "gflops": 0.39,
        "gmacs": 0.195,
        "size_mb": 21.2,
        "latency_avg_s": 0.095,
        "throughput_iter_per_s": 10.5,
        "input_resolution": "224x224",
        "description": "EfficientNet baseline - latency cao",
        "recommended_devices": ["jetson-nano"],
        "url": "https://github.com/tensorflow/tpu",
        "tags": ["efficient", "high-latency"]
    },
    {
        "name": "resnet34",
        "display_name": "ResNet-34",
        "family": "ResNet",
        "params_m": 21.8,
        "gflops": 3.68,
        "gmacs": 1.84,
        "size_mb": 87.3,
        "latency_avg_s": 0.145,
        "throughput_iter_per_s": 6.9,
        "input_resolution": "224x224",
        "description": "ResNet vừa - năng lượng CAO",
        "recommended_devices": [],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "high-energy", "not-recommended"]
    },
    {
        "name": "resnet50",
        "display_name": "ResNet-50",
        "family": "ResNet",
        "params_m": 25.56,
        "gflops": 4.12,
        "gmacs": 2.06,
        "size_mb": 102.5,
        "latency_avg_s": 0.185,
        "throughput_iter_per_s": 5.4,
        "input_resolution": "224x224",
        "description": "ResNet chuẩn - RẤT NẶNG cho edge",
        "recommended_devices": [],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "heavy", "not-recommended"]
    },
    {
        "name": "densenet121",
        "display_name": "DenseNet-121",
        "family": "DenseNet",
        "params_m": 7.98,
        "gflops": 2.87,
        "gmacs": 1.44,
        "size_mb": 32.0,
        "latency_avg_s": 0.155,
        "throughput_iter_per_s": 6.45,
        "input_resolution": "224x224",
        "description": "DenseNet - không phù hợp edge",
        "recommended_devices": [],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "heavy", "not-recommended"]
    },
    {
        "name": "vgg16",
        "display_name": "VGG-16",
        "family": "VGG",
        "params_m": 138.36,
        "gflops": 15.5,
        "gmacs": 7.75,
        "size_mb": 553.5,
        "latency_avg_s": 0.45,
        "throughput_iter_per_s": 2.2,
        "input_resolution": "224x224",
        "description": "VGG-16 - CỰC NẶNG, KHÔNG DÙNG cho edge",
        "recommended_devices": [],
        "url": "https://github.com/pytorch/vision",
        "tags": ["classic", "very-heavy", "not-recommended"]
    },
    {
        "name": "efficientnet_b3",
        "display_name": "EfficientNet-B3",
        "family": "EfficientNet",
        "params_m": 12.23,
        "gflops": 1.86,
        "gmacs": 0.93,
        "size_mb": 49.0,
        "latency_avg_s": 0.25,
        "throughput_iter_per_s": 4.0,
        "input_resolution": "300x300",
        "description": "EfficientNet lớn - năng lượng RẤT CAO",
        "recommended_devices": [],
        "url": "https://github.com/tensorflow/tpu",
        "tags": ["efficient", "very-high-energy", "not-recommended"]
    }
]


def predict_and_categorize():
    """Use trained models to predict energy for popular models"""
    
    # Initialize predictor service
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    predictor = EnergyPredictorService(artifacts_dir)
    
    # Load thresholds
    thresholds_path = os.path.join(artifacts_dir, "energy_thresholds.json")
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)
    
    jetson_p25 = thresholds.get("jetson_nano", {}).get("p25", 50)
    jetson_p50 = thresholds.get("jetson_nano", {}).get("p50", 85)
    jetson_p75 = thresholds.get("jetson_nano", {}).get("p75", 150)
    
    rpi5_p25 = thresholds.get("raspberry_pi5", {}).get("p25", 30)
    rpi5_p50 = thresholds.get("raspberry_pi5", {}).get("p50", 50)
    rpi5_p75 = thresholds.get("raspberry_pi5", {}).get("p75", 80)
    
    print("=" * 80)
    print("ENERGY PREDICTION FOR POPULAR MODELS (Using Trained Models)")
    print("=" * 80)
    print(f"\nJetson Nano Thresholds: P25={jetson_p25:.1f} | P50={jetson_p50:.1f} | P75={jetson_p75:.1f} mWh")
    print(f"Raspberry Pi 5 Thresholds: P25={rpi5_p25:.1f} | P50={rpi5_p50:.1f} | P75={rpi5_p75:.1f} mWh")
    print("\n" + "-" * 80)
    
    # Predict for each model on both devices
    results = []
    for model in POPULAR_MODELS:
        # Predict on Jetson Nano
        jetson_payload = {
            "device_type": "jetson_nano",
            "params_m": model["params_m"],
            "gflops": model["gflops"],
            "gmacs": model["gmacs"],
            "size_mb": model["size_mb"],
            "latency_avg_s": model["latency_avg_s"],
            "throughput_iter_per_s": model["throughput_iter_per_s"]
        }
        
        jetson_pred = predictor.predict([jetson_payload])[0]
        jetson_energy = jetson_pred.get("prediction_mwh", 0)
        
        # Categorize for Jetson
        if jetson_energy < jetson_p25:
            jetson_cat = "EXCELLENT"
        elif jetson_energy < jetson_p50:
            jetson_cat = "GOOD"
        elif jetson_energy < jetson_p75:
            jetson_cat = "ACCEPTABLE"
        else:
            jetson_cat = "HIGH"
        
        # Predict on RPi5
        rpi5_payload = {
            "device_type": "raspberry_pi5",
            "params_m": model["params_m"],
            "gflops": model["gflops"],
            "gmacs": model["gmacs"],
            "size_mb": model["size_mb"],
            "latency_avg_s": model["latency_avg_s"],
            "throughput_iter_per_s": model["throughput_iter_per_s"]
        }
        
        rpi5_pred = predictor.predict([rpi5_payload])[0]
        rpi5_energy = rpi5_pred.get("prediction_mwh", 0)
        
        # Categorize for RPi5
        if rpi5_energy < rpi5_p25:
            rpi5_cat = "EXCELLENT"
        elif rpi5_energy < rpi5_p50:
            rpi5_cat = "GOOD"
        elif rpi5_energy < rpi5_p75:
            rpi5_cat = "ACCEPTABLE"
        else:
            rpi5_cat = "HIGH"
        
        print(f"\n{model['display_name']:30} | Params: {model['params_m']:6.2f}M")
        print(f"  Jetson Nano:      {jetson_energy:6.1f} mWh  [{jetson_cat:11}]")
        print(f"  Raspberry Pi 5:   {rpi5_energy:6.1f} mWh  [{rpi5_cat:11}]")
        
        # Add predictions to model metadata
        model_with_predictions = model.copy()
        model_with_predictions["energy_predictions"] = {
            "jetson_nano": {
                "predicted_energy_mwh": round(jetson_energy, 2),
                "category": jetson_cat.lower(),
                "ci_lower_mwh": round(jetson_pred.get("ci_lower_mwh", 0), 2),
                "ci_upper_mwh": round(jetson_pred.get("ci_upper_mwh", 0), 2)
            },
            "raspberry_pi5": {
                "predicted_energy_mwh": round(rpi5_energy, 2),
                "category": rpi5_cat.lower(),
                "ci_lower_mwh": round(rpi5_pred.get("ci_lower_mwh", 0), 2),
                "ci_upper_mwh": round(rpi5_pred.get("ci_upper_mwh", 0), 2)
            }
        }
        
        results.append(model_with_predictions)
    
    # Save to JSON
    output_path = os.path.join(artifacts_dir, "popular_models_metadata.json")
    output_data = {
        "description": "Metadata for popular ML models with PREDICTED energy consumption using trained models",
        "generated_by": "generate_popular_models_metadata.py",
        "thresholds": {
            "jetson_nano": {"p25": jetson_p25, "p50": jetson_p50, "p75": jetson_p75},
            "raspberry_pi5": {"p25": rpi5_p25, "p50": rpi5_p50, "p75": rpi5_p75}
        },
        "models": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"✅ Generated metadata saved to: {output_path}")
    print(f"   Total models: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    predict_and_categorize()
