#!/usr/bin/env python3
"""
High-Performance Evaluation & Benchmark Suite for NVIDIA Jetson Orin Nano
Target: A* / Q1 Conference & Journal Edge Segmentation Papers

Features:
- ONNX Runtime GPU (CUDA / TensorRT Execution Provider) and native TensorRT Engine (.engine)
- Microsecond latency measurement via time.perf_counter_ns
- Comprehensive Edge & Semantic Metrics: mIoU, Dice (F1), Boundary-IoU, Accuracy
- Detailed Edge Efficiency Metrics: Latency (mean, p50, p95, std), FPS, Throughput
- Automatic generation of publication-ready JSON results
"""

import os
import sys
import time
import glob
import json
import argparse
import numpy as np
import cv2

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def compute_metrics(pred_mask, gt_mask):
    """
    Compute comprehensive lightweight segmentation metrics:
    - mIoU (Mean Intersection over Union)
    - IoU_flood & IoU_background
    - Dice / F1 Score
    - Precision (PPV - Positive Predictive Value)
    - Recall (TPR / Sensitivity)
    - Specificity (TNR - True Negative Rate)
    - Pixel Accuracy (Overall Accuracy)
    - Frequency-Weighted IoU (FWIoU)
    - Boundary-IoU (Contour Quality - CVPR 2021)
    """
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)

    # Basic Confusion Matrix Elements
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    # IoU metrics
    union_flood = tp + fp + fn
    iou_flood = float(tp) / float(max(union_flood, 1))

    union_bg = tn + fp + fn
    iou_bg = float(tn) / float(max(union_bg, 1))

    miou = (iou_flood + iou_bg) / 2.0
    dice = float(2.0 * tp) / float(max(2 * tp + fp + fn, 1))

    # Precision, Recall, Specificity
    precision = float(tp) / float(max(tp + fp, 1))
    recall = float(tp) / float(max(tp + fn, 1))
    specificity = float(tn) / float(max(tn + fp, 1))

    # Pixel Accuracy
    pixel_acc = float(tp + tn) / float(max(pred.size, 1))

    # Frequency-Weighted IoU
    total_pix = float(pred.size)
    n_flood = float(gt.sum())
    n_bg = total_pix - n_flood
    fwiou = (n_flood * iou_flood + n_bg * iou_bg) / total_pix

    # Boundary-IoU with 3-pixel morphological contour band (Cheng et al., CVPR 2021)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    p_bound = cv2.dilate(pred, kernel) - cv2.erode(pred, kernel)
    g_bound = cv2.dilate(gt, kernel) - cv2.erode(gt, kernel)

    b_inter = np.logical_and(p_bound > 0, g_bound > 0).sum()
    b_union = np.logical_or(p_bound > 0, g_bound > 0).sum()
    boundary_iou = float(b_inter) / float(max(b_union, 1)) if b_union > 0 else 1.0

    return {
        "miou": miou,
        "iou_flood": iou_flood,
        "iou_bg": iou_bg,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_accuracy": pixel_acc,
        "fwiou": fwiou,
        "boundary_iou": boundary_iou
    }


class ONNXModelWrapper:
    """Wrapper for ONNX Runtime inference on NVIDIA Jetson."""
    def __init__(self, onnx_path, device="cuda"):
        if not HAS_ORT:
            raise ImportError("onnxruntime is required for ONNX model inference.")
        self.onnx_path = onnx_path
        
        providers = []
        if device == "cuda":
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        # Parse height & width (fallback to 128 if dynamic)
        h = self.input_shape[2] if len(self.input_shape) > 2 and isinstance(self.input_shape[2], int) else 128
        w = self.input_shape[3] if len(self.input_shape) > 3 and isinstance(self.input_shape[3], int) else 128
        self.h = h
        self.w = w

    def preprocess(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, input_tensor):
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw = outputs[0]
        # Sigmoid for binary probability map
        probs = 1.0 / (1.0 + np.exp(-raw))
        if probs.ndim == 4:
            probs = probs[0, 0]
        elif probs.ndim == 3:
            probs = probs[0]
        return probs


def evaluate_dataset(model, images_dir, masks_dir, warm_up=30):
    img_files = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.*")))

    if not img_files or not mask_files:
        print(f"Warning: No evaluation pairs found in {images_dir} & {masks_dir}")
        return {}

    # Warm-up phase
    dummy = np.zeros((1, 3, model.h, model.w), dtype=np.float32)
    for _ in range(warm_up):
        _ = model.predict(dummy)

    latencies_ms = []
    all_metrics = []

    for img_p, mask_p in zip(img_files, mask_files):
        img_bgr = cv2.imread(img_p)
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        
        if img_bgr is None or gt_mask is None:
            continue
            
        orig_h, orig_w = gt_mask.shape
        tensor = model.preprocess(img_bgr)
        
        # Microsecond precision benchmark
        t0 = time.perf_counter_ns()
        probs = model.predict(tensor)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)
        
        # Resize prediction back to original mask size for evaluation
        pred_full = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        m = compute_metrics(pred_full, (gt_mask > 127).astype(np.uint8))
        all_metrics.append(m)

    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    fps = 1000.0 / max(mean_lat, 1e-4)

    mean_miou = float(np.mean([m["miou"] for m in all_metrics]))
    mean_iou_flood = float(np.mean([m["iou_flood"] for m in all_metrics]))
    mean_iou_bg = float(np.mean([m["iou_bg"] for m in all_metrics]))
    mean_dice = float(np.mean([m["dice"] for m in all_metrics]))
    mean_prec = float(np.mean([m["precision"] for m in all_metrics]))
    mean_rec = float(np.mean([m["recall"] for m in all_metrics]))
    mean_spec = float(np.mean([m["specificity"] for m in all_metrics]))
    mean_fwiou = float(np.mean([m["fwiou"] for m in all_metrics]))
    mean_b_iou = float(np.mean([m["boundary_iou"] for m in all_metrics]))
    mean_acc = float(np.mean([m["pixel_accuracy"] for m in all_metrics]))

    report = {
        "samples_evaluated": len(all_metrics),
        "mean_latency_ms": round(mean_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "p50_latency_ms": round(p50_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "fps": round(fps, 1),
        "miou": round(mean_miou * 100, 2),
        "iou_flood": round(mean_iou_flood * 100, 2),
        "iou_bg": round(mean_iou_bg * 100, 2),
        "dice": round(mean_dice * 100, 2),
        "precision": round(mean_prec * 100, 2),
        "recall": round(mean_rec * 100, 2),
        "specificity": round(mean_spec * 100, 2),
        "fwiou": round(mean_fwiou * 100, 2),
        "boundary_iou": round(mean_b_iou * 100, 2),
        "pixel_accuracy": round(mean_acc * 100, 2)
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Jetson Orin Nano Edge Benchmark Suite")
    parser.add_argument("--model_path", type=str, default="", help="Path to ONNX model file")
    parser.add_argument("--images_dir", type=str, default="", help="Path to test images folder")
    parser.add_argument("--masks_dir", type=str, default="", help="Path to test masks folder")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--warm_up", type=int, default=30)
    parser.add_argument("--out_json", type=str, default="jetson_evaluation_results.json")
    args = parser.parse_args()

    if not args.model_path:
        print("[ERROR] Please provide --model_path (e.g., outputs/onnx/ours_128x128.onnx)")
        sys.exit(1)

    print("=" * 90)
    print("NVIDIA Jetson Orin Nano Edge Inference & Accuracy Evaluation")
    print(f"Model       : {args.model_path}")
    print(f"Device      : {args.device.upper()}")
    print("=" * 90)

    model = ONNXModelWrapper(args.model_path, device=args.device)

    if args.images_dir and args.masks_dir:
        report = evaluate_dataset(model, args.images_dir, args.masks_dir, warm_up=args.warm_up)
        print("\nEvaluation Results:")
        for k, v in report.items():
            print(f"  {k:20s}: {v}")

        with open(args.out_json, "w") as f:
            json.dump({os.path.basename(args.model_path): report}, f, indent=2)
        print(f"\nSaved results to {args.out_json}")
    else:
        # Benchmark latency profiling with dummy inputs
        print("\n[INFO] Running latency benchmark with dummy inputs (warmup=50, runs=300)...")
        dummy = np.zeros((1, 3, model.h, model.w), dtype=np.float32)
        for _ in range(50):
            _ = model.predict(dummy)

        latencies = []
        for _ in range(300):
            t0 = time.perf_counter_ns()
            _ = model.predict(dummy)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)

        lat = np.array(latencies)
        mean_lat = float(np.mean(lat))
        p50 = float(np.percentile(lat, 50))
        p95 = float(np.percentile(lat, 95))
        fps = 1000.0 / mean_lat

        print(f"  Mean Latency : {mean_lat:.3f} ms ± {np.std(lat):.3f} ms")
        print(f"  P50 Latency  : {p50:.3f} ms")
        print(f"  P95 Latency  : {p95:.3f} ms")
        print(f"  FPS          : {fps:.1f}")

        res = {
            "model": os.path.basename(args.model_path),
            "resolution": f"{model.h}x{model.w}",
            "mean_latency_ms": round(mean_lat, 3),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "fps": round(fps, 1)
        }
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved latency profile to {args.out_json}")


if __name__ == "__main__":
    main()
