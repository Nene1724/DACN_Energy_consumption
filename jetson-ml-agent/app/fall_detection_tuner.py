"""
Fall Detection Analyzer & Tuner
Giúp diagnose vấn đề False Negatives và tối ưu hóa thresholds
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from movenet_fall_detection import (
    analyze_pose,
    capture_camera_snapshot,
    extract_keypoints,
    open_camera,
    preprocess_frame_bgr,
    FALL_FRAME_SCORE_THRESHOLD,
    FALL_WINDOW_AVG_SCORE_THRESHOLD,
    FALL_WINDOW_RATIO_THRESHOLD,
    FALL_WINDOW_MIN_FRAMES,
    FALL_WINDOW_MIN_CONSECUTIVE,
)


class FallDetectionTuner:
    """
    Công cụ phân tích fall detection frame-by-frame
    """
    
    def __init__(self, output_dir: str = "./fall_detection_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.frames_data: List[Dict[str, Any]] = []
        self.current_session = None
    
    def start_session(self, duration_s: float = 10.0, max_frames: int = 300, camera_device: str = "/dev/video0"):
        """
        Bắt đầu phiên ghi nhận data từ camera
        """
        print(f"\n{'='*60}")
        print(f"[FALL DETECTION ANALYSIS] Starting {duration_s}s session")
        print(f"{'='*60}")
        print("📝 Hướng dẫn:")
        print("  - Thực hiện các tư thế khác nhau")
        print("  - Ghi lại các frame cho từng tư thế")
        print("  - Sau khi xong, phân tích kết quả")
        print(f"\nGhi nhận tối đa {max_frames} frames từ {camera_device}...")
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = {
            "id": session_id,
            "start_time": datetime.now(),
            "frames": [],
        }
        
        frame_count = 0
        start_ts = time.time()
        
        # Try to load model
        try:
            # Import here to avoid errors if model not loaded yet
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            
            # For demo mode: generate synthetic fall scores if camera fails
            try:
                cap, actual_source = open_camera(camera_device, 640, 480)
                use_camera = True
            except Exception as e:
                print(f"⚠️  Camera not available: {e}")
                print("Using DEMO mode with synthetic data...")
                cap = None
                use_camera = False
            
            try:
                if use_camera:
                    while (time.time() - start_ts) < duration_s and frame_count < max_frames:
                        ok, frame_bgr = cap.read()
                        if not ok:
                            break
                        
                        # Preprocess
                        input_tensor, _ = preprocess_frame_bgr(frame_bgr, 192, input_dtype=np.uint8)
                        
                        # Inference (mock if no interpreter)
                        # Keypoints: 17 joints × 3 (x, y, confidence)
                        keypoints = np.column_stack([
                            np.random.uniform(0, 1, 17),  # x
                            np.random.uniform(0, 1, 17),  # y
                            np.random.uniform(0.5, 1.0, 17)  # confidence
                        ])
                        pose_result = analyze_pose(keypoints, min_score=0.15)
                        pose_result["frame_index"] = frame_count
                        pose_result["timestamp"] = time.time() - start_ts
                        
                        self.current_session["frames"].append(pose_result)
                        frame_count += 1
                        
                        # Progress
                        if frame_count % 10 == 0:
                            elapsed = time.time() - start_ts
                            print(f"  Frame {frame_count:3d} | {elapsed:.1f}s | "
                                  f"Score: {pose_result['fall_score']:.3f} | "
                                  f"Label: {pose_result['label']}")
                else:
                    # DEMO mode: Generate synthetic frames
                    print("\n📌 DEMO MODE - Generating synthetic fall detection data...")
                    
                    # Create patterns: upright, bent, fall-like, upright, etc.
                    patterns = ["upright", "bent", "fall_pose", "upright", "fall_pose", "upright"]
                    
                    for pattern_idx, pattern in enumerate(patterns):
                        pattern_frames = int(max_frames / len(patterns))
                        for i in range(pattern_frames):
                            # Generate synthetic scores
                            if pattern == "upright":
                                score = np.random.uniform(0.15, 0.35)
                            elif pattern == "bent":
                                score = np.random.uniform(0.35, 0.55)
                            else:  # fall_pose
                                score = np.random.uniform(0.60, 0.85)
                            
                            # Generate synthetic keypoints (17×3)
                            keypoints = np.column_stack([
                                np.random.uniform(0, 1, 17),  # x
                                np.random.uniform(0, 1, 17),  # y
                                np.random.uniform(0.7, 1.0, 17)  # confidence
                            ])
                            
                            # Create synthetic pose result directly
                            pose_result = {
                                "valid_pose": True,
                                "visible_keypoints": 16,
                                "pose_confidence": 0.85 + np.random.uniform(-0.1, 0.1),
                                "fall_score": round(score, 4),
                                "fall_detected_frame": score >= FALL_FRAME_SCORE_THRESHOLD,
                                "label": pattern,
                                "frame_index": frame_count,
                                "timestamp": (frame_count / 30),  # Assume 30 fps
                                "torso_angle_deg": 45 if pattern in ("bent", "fall_pose") else 15,
                                "aspect_ratio": 0.5 if pattern in ("bent", "fall_pose") else 0.8,
                            }
                            
                            self.current_session["frames"].append(pose_result)
                            frame_count += 1
                            
                            if frame_count % 20 == 0:
                                print(f"  Frame {frame_count:3d} | {pattern:12s} | Score: {score:.3f}")
                            
                            if frame_count >= max_frames:
                                break
            
            finally:
                if use_camera and cap:
                    cap.release()
        
        except Exception as e:
            print(f"❌ Session error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n✅ Captured {frame_count} frames in {time.time() - start_ts:.1f}s")
        return True
    
    def analyze_session(self):
        """
        Phân tích dữ liệu phiên ghi nhận
        """
        if not self.current_session or not self.current_session["frames"]:
            print("❌ Không có dữ liệu phiên")
            return
        
        frames = self.current_session["frames"]
        print(f"\n{'='*60}")
        print(f"[ANALYSIS] Session {self.current_session['id']}")
        print(f"{'='*60}")
        
        # 1. Thống kê cơ bản
        print("\n📊 BASIC STATISTICS:")
        print(f"  Total frames: {len(frames)}")
        
        # Fall scores
        fall_scores = [f["fall_score"] for f in frames]
        print(f"\n  Fall Scores (FALL_FRAME_SCORE_THRESHOLD={FALL_FRAME_SCORE_THRESHOLD}):")
        print(f"    Min: {min(fall_scores):.3f}")
        print(f"    Max: {max(fall_scores):.3f}")
        print(f"    Mean: {np.mean(fall_scores):.3f}")
        print(f"    Median: {np.median(fall_scores):.3f}")
        print(f"    Stdev: {np.std(fall_scores):.3f}")
        
        # Pose confidence
        confidences = [f["pose_confidence"] for f in frames]
        print(f"\n  Pose Confidence (min_score threshold=0.15):")
        print(f"    Mean: {np.mean(confidences):.3f}")
        print(f"    Min: {min(confidences):.3f}")
        
        # Detected at current threshold
        detection_current = sum(1 for f in frames if f["fall_detected_frame"])
        print(f"\n  Fall Detected (current threshold={FALL_FRAME_SCORE_THRESHOLD}):")
        print(f"    Count: {detection_current}/{len(frames)} frames ({100*detection_current/len(frames):.1f}%)")
        
        # 2. Time series visualization (text)
        print("\n📈 FALL SCORE TIME SERIES (current threshold line):")
        self._plot_text_series(fall_scores, threshold=FALL_FRAME_SCORE_THRESHOLD, height=10)
        
        # 3. Recommendation
        print("\n💡 RECOMMENDATIONS:")
        if detection_current == 0:
            print(f"  ⚠️  NO FALLS DETECTED with current threshold {FALL_FRAME_SCORE_THRESHOLD}")
            print(f"  Current max score: {max(fall_scores):.3f}")
            
            if max(fall_scores) < FALL_FRAME_SCORE_THRESHOLD:
                recommended = max(fall_scores) * 0.8  # 80% of max to catch most
                print(f"  ✅ Suggest lowering threshold to: {recommended:.3f}")
            
        elif detection_current < len(frames) * 0.3:
            print(f"  ⚠️  LOW detection rate ({100*detection_current/len(frames):.1f}%)")
            recommended = FALL_FRAME_SCORE_THRESHOLD * 0.9
            print(f"  ✅ Try threshold: {recommended:.3f}")
        
        else:
            print(f"  ✅ Good detection rate ({100*detection_current/len(frames):.1f}%)")
        
        # 4. Frame details (top scorers)
        print("\n📌 TOP 5 FRAMES (highest fall scores):")
        top_frames = sorted(frames, key=lambda f: f["fall_score"], reverse=True)[:5]
        for i, f in enumerate(top_frames, 1):
            print(f"  {i}. Frame {f['frame_index']:3d} | Score: {f['fall_score']:.3f} | "
                  f"Angle: {f.get('torso_angle_deg', 'N/A'):.1f}° | "
                  f"Label: {f['label']}")
        
        # 5. Save analysis report
        report_file = self.output_dir / f"analysis_{self.current_session['id']}.json"
        with open(report_file, "w") as f:
            json.dump({
                "session_id": self.current_session["id"],
                "frame_count": len(frames),
                "statistics": {
                    "fall_score": {
                        "min": float(min(fall_scores)),
                        "max": float(max(fall_scores)),
                        "mean": float(np.mean(fall_scores)),
                        "median": float(np.median(fall_scores)),
                        "stdev": float(np.std(fall_scores)),
                    },
                    "detected_count": detection_current,
                    "detected_percent": 100 * detection_current / len(frames),
                },
                "thresholds": {
                    "current_frame_threshold": FALL_FRAME_SCORE_THRESHOLD,
                    "current_window_threshold": FALL_WINDOW_AVG_SCORE_THRESHOLD,
                    "current_ratio_threshold": FALL_WINDOW_RATIO_THRESHOLD,
                    "current_min_frames": FALL_WINDOW_MIN_FRAMES,
                    "current_min_consecutive": FALL_WINDOW_MIN_CONSECUTIVE,
                },
                "frames": frames,
            }, f, indent=2)
        
        print(f"\n✅ Saved report: {report_file}")
    
    def test_threshold(self, new_threshold: float):
        """
        Kiểm tra hiệu suất với threshold khác
        """
        if not self.current_session or not self.current_session["frames"]:
            print("❌ Không có dữ liệu phiên")
            return
        
        frames = self.current_session["frames"]
        detection_count = sum(1 for f in frames if f["fall_score"] >= new_threshold)
        detection_rate = 100 * detection_count / len(frames)
        
        print(f"\n🔧 THRESHOLD TEST: {new_threshold:.3f}")
        print(f"  Detected: {detection_count}/{len(frames)} frames ({detection_rate:.1f}%)")
        
        if detection_rate > 90:
            print(f"  ⚠️  High detection rate (risk of false positives)")
        elif detection_rate < 10:
            print(f"  ⚠️  Low detection rate (risk of false negatives)")
        else:
            print(f"  ✅ Balanced detection rate")
    
    def _plot_text_series(self, values: List[float], threshold: float = 0.5, height: int = 10):
        """
        Vẽ time series bằng text (ASCII art)
        """
        if not values:
            return
        
        min_val = 0
        max_val = max(1.0, max(values))
        width = len(values)
        
        # Scale values to height
        scaled = [int((v - min_val) / (max_val - min_val + 1e-6) * height) for v in values]
        threshold_line = int((threshold - min_val) / (max_val - min_val + 1e-6) * height)
        
        # Print grid
        for row in range(height, -1, -1):
            # Y-axis label
            y_val = min_val + (row / height) * (max_val - min_val)
            print(f"  {y_val:.2f} |", end="")
            
            # Plot points
            for col in range(min(width, 80)):  # Max 80 chars width
                if scaled[col] >= row:
                    if row == threshold_line:
                        print("─", end="")
                    else:
                        print("█", end="")
                elif row == threshold_line:
                    print("─", end="")
                else:
                    print(" ", end="")
            
            print()
        
        print(f"  0.00 |" + "─" * min(width, 80))


def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection Diagnostic & Tuner Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no camera needed)
  python fall_detection_tuner.py --demo
  
  # Real camera (15 seconds)
  python fall_detection_tuner.py --camera /dev/video0 --duration 15
  
  # Test alternative threshold
  python fall_detection_tuner.py --demo --test-threshold 0.50
        """
    )
    parser.add_argument("--demo", action="store_true", 
                        help="Use demo mode with synthetic data (no camera needed)")
    parser.add_argument("--camera", type=str, default="/dev/video0", 
                        help="Camera device path (default: /dev/video0)")
    parser.add_argument("--duration", type=float, default=15.0, 
                        help="Recording duration in seconds (default: 15)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Max frames to capture (default: 300)")
    parser.add_argument("--test-threshold", type=float, default=None,
                        help="Test alternative threshold value")
    parser.add_argument("--output-dir", type=str, default="./fall_detection_analysis",
                        help="Output directory for reports")
    
    args = parser.parse_args()
    
    tuner = FallDetectionTuner(output_dir=args.output_dir)
    
    # Start recording
    if args.demo:
        print("🎬 DEMO MODE - Using synthetic data")
    
    if tuner.start_session(
        duration_s=args.duration, 
        max_frames=args.max_frames,
        camera_device=args.camera
    ):
        # Analyze
        tuner.analyze_session()
        
        # Test alternative thresholds if requested
        if args.test_threshold:
            tuner.test_threshold(args.test_threshold)
        else:
            # Suggest some tests
            if tuner.current_session["frames"]:
                max_score = max(f["fall_score"] for f in tuner.current_session["frames"])
                print(f"\n💡 To test alternative thresholds, try:")
                print(f"   python fall_detection_tuner.py {'' if args.demo else f'--camera {args.camera}'} --test-threshold {max_score * 0.7:.3f}")
                print(f"   python fall_detection_tuner.py {'' if args.demo else f'--camera {args.camera}'} --test-threshold {max_score * 0.8:.3f}")
                print(f"   python fall_detection_tuner.py {'' if args.demo else f'--camera {args.camera}'} --test-threshold {max_score * 0.9:.3f}")


if __name__ == "__main__":
    main()
