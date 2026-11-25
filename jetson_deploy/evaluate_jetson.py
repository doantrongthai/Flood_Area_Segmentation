#!/usr/bin/env python3
"""
Accuracy and latency of an exported flood-segmentation ONNX model on NVIDIA Jetson.

Preprocessing, mask handling and metrics follow training exactly
(utils/dataloader.py and utils/metrics.py), so the numbers can be compared with the
test-set results obtained on the training machine:
  - input: RGB, resized (bilinear) to the model size, divided by 255. No ImageNet normalisation.
  - mask : floodscene -> flood is value 255; floodkaggle -> value > 127; resized (nearest).
  - IoU / Dice of the flood class, pooled over all test pixels at the model resolution.

Needs numpy, opencv and an onnxruntime build with GPU support. PyTorch is not needed.

Examples:
  python3 jetson_deploy/evaluate_jetson.py --model_path outputs/onnx/ours_floodkaggle_s*_256x256.onnx \\
      --images_dir floodkaggle/test/images --masks_dir floodkaggle/test/masks --dataset_type floodkaggle
  python3 jetson_deploy/evaluate_jetson.py --model_path outputs/onnx/ours_floodkaggle_s42_256x256.onnx
      (no images given: latency only)
"""

import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[ERROR] onnxruntime is not installed. See jetson_deploy/requirements-jetson.txt")

TRAIN_SIZE = 256
IMAGE_EXT = (".png", ".jpg", ".jpeg")
PROVIDERS = {
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


# ---------------------------------------------------------------- data
def preprocess(img_bgr, size):
    """Same as the validation/test transform used in training."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])


def load_mask(path, dataset_type, image_shape, size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    if mask.shape[:2] != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    if dataset_type == "floodscene":
        mask = (mask == 255).astype(np.uint8)    # 0=background, 85=sky, 170=building, 255=flood
    else:
        mask = (mask > 127).astype(np.uint8)     # binary mask with JPEG artefacts
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def list_pairs(images_dir, masks_dir):
    names = sorted(f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXT)
    if not names:
        sys.exit(f"[ERROR] No images in {images_dir}")
    return [(os.path.join(images_dir, f), os.path.join(masks_dir, os.path.splitext(f)[0] + ".png"))
            for f in names]


# ---------------------------------------------------------------- model
def make_session(onnx_path, provider, precision):
    wanted = PROVIDERS[provider]
    available = ort.get_available_providers()
    if wanted not in available:
        sys.exit(f"[ERROR] {wanted} is not available (found: {available}). "
                 f"Install a GPU build of onnxruntime for your JetPack, or choose --provider.")
    if provider == "tensorrt":
        providers = [(wanted, {"trt_fp16_enable": precision == "fp16"}),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [wanted, "CPUExecutionProvider"]
    else:
        providers = [wanted]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, options, providers=providers)
    if session.get_providers()[0] != wanted:
        sys.exit(f"[ERROR] {wanted} could not be initialised; ONNX Runtime fell back to "
                 f"{session.get_providers()}. Refusing to report these timings.")
    return session


def input_size(session, requested):
    shape = session.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int):
        if h != w or (requested and requested != h):
            sys.exit(f"[ERROR] Model input is {h}x{w}, which does not match --size {requested}.")
        size = h
    else:
        size = requested or TRAIN_SIZE
    if size != TRAIN_SIZE:
        print(f"[WARN] Input size {size} differs from the training size {TRAIN_SIZE}.")
    return size


def forward(session, x):
    return session.run(None, {session.get_inputs()[0].name: x})[0]


def to_mask(logits):
    # sigmoid(logit) > 0.5, as in utils/metrics.py
    return (1.0 / (1.0 + np.exp(-logits[0, 0])) > 0.5).astype(np.uint8)


# ---------------------------------------------------------------- statistics
def latency_stats(ms):
    ms = np.asarray(ms)
    return {"mean": float(ms.mean()), "std": float(ms.std()), "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
            "min": float(ms.min()), "max": float(ms.max()), "n": int(ms.size)}


def boundary_band(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return (cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)) > 0


def evaluate(session, size, pairs, dataset_type):
    tp = fp = fn = tn = 0
    infer_ms, total_ms, band_iou = [], [], []
    for image_path, mask_path in pairs:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image unreadable: {image_path}")
        gt = load_mask(mask_path, dataset_type, img.shape, size)

        t0 = time.perf_counter()
        x = preprocess(img, size)
        t1 = time.perf_counter()
        logits = forward(session, x)
        t2 = time.perf_counter()
        pred = to_mask(logits)
        t3 = time.perf_counter()
        infer_ms.append((t2 - t1) * 1e3)
        total_ms.append((t3 - t0) * 1e3)

        tp += int(np.sum((pred == 1) & (gt == 1)))
        fp += int(np.sum((pred == 1) & (gt == 0)))
        fn += int(np.sum((pred == 0) & (gt == 1)))
        tn += int(np.sum((pred == 0) & (gt == 0)))
        p, g = boundary_band(pred), boundary_band(gt)
        union = np.sum(p | g)
        band_iou.append(float(np.sum(p & g) / union) if union else 1.0)

    metrics = {
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn),
        # supplementary, not used in training: per-image overlap of 3x3 morphological boundary bands
        "boundary_band_iou_per_image_mean": float(np.mean(band_iou)),
    }
    return {"images": len(pairs), "metrics": metrics,
            "inference_ms": latency_stats(infer_ms), "end_to_end_ms": latency_stats(total_ms)}


def latency_only(session, size, iterations):
    x = np.random.RandomState(0).rand(1, 3, size, size).astype(np.float32)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        forward(session, x)
        times.append((time.perf_counter() - t0) * 1e3)
    return {"inference_ms": latency_stats(times)}


def device_name():
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def main():
    ap = argparse.ArgumentParser(description="Jetson evaluation of exported ONNX segmentation models")
    ap.add_argument("--model_path", nargs="+", required=True, help="one ONNX file, or one per seed")
    ap.add_argument("--images_dir", help="test images; omit for latency only")
    ap.add_argument("--masks_dir", help="test masks (<image name>.png)")
    ap.add_argument("--dataset_type", choices=["floodkaggle", "floodscene"],
                    help="required with --images_dir: decides how masks are read")
    ap.add_argument("--size", type=int, default=None, help=f"only for dynamic-shape models (default {TRAIN_SIZE})")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="tensorrt")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                    help="fp16 applies to the TensorRT provider")
    ap.add_argument("--warm_up", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=1000, help="latency-only mode")
    ap.add_argument("--out_json", default="jetson_evaluation_results.json")
    args = ap.parse_args()

    if bool(args.images_dir) != bool(args.masks_dir):
        ap.error("--images_dir and --masks_dir go together")
    if args.images_dir and not args.dataset_type:
        ap.error("--dataset_type is required for accuracy evaluation")
    if args.precision == "fp16" and args.provider != "tensorrt":
        ap.error("--precision fp16 needs --provider tensorrt")

    pairs = list_pairs(args.images_dir, args.masks_dir) if args.images_dir else None
    report = {"device": device_name(), "onnxruntime": ort.__version__, "provider": PROVIDERS[args.provider],
              "precision": args.precision, "dataset_type": args.dataset_type, "models": {}}

    for path in args.model_path:
        session = make_session(path, args.provider, args.precision)
        size = input_size(session, args.size)
        dummy = np.zeros((1, 3, size, size), dtype=np.float32)
        for _ in range(args.warm_up):
            forward(session, dummy)
        result = evaluate(session, size, pairs, args.dataset_type) if pairs else \
            latency_only(session, size, args.iterations)
        result.update({"input_size": size, "active_providers": session.get_providers()})
        report["models"][os.path.basename(path)] = result

        print(f"\n{os.path.basename(path)}  ({size}x{size}, {args.provider}, {args.precision})")
        if pairs:
            m = result["metrics"]
            print(f"  images {result['images']} | IoU {m['iou']:.4f} | Dice {m['dice']:.4f} | "
                  f"P {m['precision']:.4f} | R {m['recall']:.4f}")
            e = result["end_to_end_ms"]
            print(f"  pre+inference+threshold : {e['mean']:.2f} +- {e['std']:.2f} ms")
        lat = result["inference_ms"]
        print(f"  inference only          : {lat['mean']:.2f} +- {lat['std']:.2f} ms "
              f"(p95 {lat['p95']:.2f}, max {lat['max']:.2f}, n={lat['n']})")

    if pairs and len(args.model_path) > 1:        # e.g. the five seeds: mean +- std as in the paper
        summary = {}
        for key in ("iou", "dice"):
            values = [r["metrics"][key] for r in report["models"].values()]
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
            print(f"\nacross {len(values)} models: {key} {np.mean(values):.4f} +- {np.std(values):.4f}")
        report["across_models"] = summary

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
