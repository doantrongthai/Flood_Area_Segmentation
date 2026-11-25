# NVIDIA Jetson Orin Nano Edge Deployment & Benchmark Suite

Official edge inference and hardware benchmark protocol for lightweight real-time flood area segmentation models (**DWSeg_Lite / ours** and **MambaFloodLite**), formatted for **A* / Q1 Conference & Journal Papers** (IEEE TGRS, ISPRS P&RS, CVPR, ECCV, ICCV, IEEE T-CSVT).

---

## 1. Directory Structure

```text
Flood_Area_Segmentation/
├── export.py                        # Universal ONNX exporter with DeployableModel wrapper
├── models/
│   ├── mambafloodlite.py            # MambaFloodLite architecture (285K params, pure PyTorch fallback)
│   ├── ours.py                      # DWSeg_Lite proposed architecture
│   └── ...
├── outputs/
│   └── onnx/                        # Exported ONNX models ready for TensorRT compilation
│       ├── ours_128x128.onnx
│       └── mambafloodlite_128x128.onnx
└── jetson_deploy/
    ├── benchmark_trtexec.py         # Official trtexec isolated GPU kernel latency suite
    ├── evaluate_jetson.py           # End-to-end evaluation (mIoU, Dice, Boundary-IoU, Latency, FPS)
    ├── run_official_benchmark.sh    # Hardware clock locking and benchmark automation script
    └── README_JETSON.md             # This deployment protocol documentation
```

---

## 2. Hardware Environment Locking (A* / Q1 Standard)

To prevent frequency scaling jitter and thermal throttling during latency measurement, lock the Jetson Orin Nano clocks to maximum deterministic frequencies before profiling:

```bash
# 1. Select MAXN (15W Max Performance) power mode
sudo nvpmodel -m 0

# 2. Lock GPU, CPU, and EMC memory bus clocks to max frequency
sudo jetson_clocks

# 3. Verify hardware clock status
sudo jetson_clocks --show
```

---

## 3. ONNX Model Export

Export models from the repository root:

```bash
# Export proposed architecture (DWSeg_Lite)
python export.py --model ours --size 128 --output outputs/onnx/ours_128x128.onnx

# Export MambaFloodLite architecture
python export.py --model mambafloodlite --size 128 --output outputs/onnx/mambafloodlite_128x128.onnx

# Batch export all core architectures in models/
python export.py --export_all --size 128
```

---

## 4. Benchmark Execution on NVIDIA Jetson

### Method A: Pure GPU Kernel Benchmark via `trtexec` (Isolated Compute Time)

`trtexec` is the gold standard used by NVIDIA to benchmark hardware latency without host CPU overhead:

```bash
# Run isolated GPU kernel benchmark (FP16, 500 warmups, 1000 iterations)
python3 jetson_deploy/benchmark_trtexec.py \
    --onnx_dir outputs/onnx \
    --precision fp16 \
    --warm_up 500 \
    --iterations 1000 \
    --out_json jetson_trtexec_hardware_results.json
```

**Metrics extracted:**
- `gpu_compute_mean_ms`: Mean GPU execution time (ms)
- `gpu_compute_median_ms`: Median execution time (P50) (ms)
- `gpu_compute_p90_ms`, `gpu_compute_p95_ms`, `gpu_compute_p99_ms`: Tail latency percentiles
- `fps_throughput`: Throughput in frames per second (FPS)

### Method B: End-to-End Accuracy & Edge Latency Evaluation

Evaluates both segmentation accuracy and edge inference speed on test images:

```bash
python3 jetson_deploy/evaluate_jetson.py \
    --model_path outputs/onnx/ours_128x128.onnx \
    --images_dir path/to/test/images \
    --masks_dir path/to/test/masks \
    --device cuda \
    --out_json jetson_evaluation_results.json
```

**Metrics reported:**
- **Accuracy**: mIoU, Flood-IoU, Background-IoU, Dice/F1, Boundary-IoU (3-pixel morphological band, CVPR 2021), Precision, Recall, Specificity, Pixel Accuracy, FWIoU.
- **Latency & Throughput**: Mean latency (ms) $\pm$ Std, P50, P95, P99, FPS.

### Method C: One-Click Automated Run

```bash
bash jetson_deploy/run_official_benchmark.sh
```

---

## 5. Summary of Architecture Specifications

| Model | Parameters | Target Resolution | TensorRT Precision | Pure PyTorch Fallback |
| :--- | :---: | :---: | :---: | :---: |
| **DWSeg_Lite (`ours`)** | ~100K | 128×128 | FP16 / INT8 | Yes |
| **MambaFloodLite** | 285,465 | 128×128 | FP16 / CUDA kernel | Yes |
