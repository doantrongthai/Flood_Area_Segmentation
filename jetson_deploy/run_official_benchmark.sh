#!/usr/bin/env bash
# ==============================================================================
# NVIDIA Jetson Orin Nano Official Benchmark Runner
# Standard Protocol for Top-Tier A* / Q1 Edge AI Papers
# ==============================================================================

set -e

echo "=============================================================================="
echo "STEP 1: Hardware Environment Locking (MAXN + Maximum Clock Frequency)"
echo "=============================================================================="
# On JetPack 5.x / 6.x, lock 15W/MAXN power mode and fix GPU/CPU/EMC clock frequencies
sudo nvpmodel -m 0 2>/dev/null || sudo nvpmodel -q || echo "[INFO] Proceeding with hardware clocks..."
sudo jetson_clocks
sudo jetson_clocks --show | head -n 15
echo "[OK] Hardware clocks locked at maximum frequency."

echo ""
echo "=============================================================================="
echo "STEP 2: Official trtexec Isolated GPU Hardware Benchmark (FP16, 128x128, Batch=1)"
echo "=============================================================================="
# Measures pure GPU kernel compute time with 500 warmups + 1000 iterations
python3 jetson_deploy/benchmark_trtexec.py --onnx_dir outputs/onnx --precision fp16 --warm_up 500 --iterations 1000

echo ""
echo "=============================================================================="
echo "STEP 3: Accuracy & Metric Evaluation (mIoU, Dice, Boundary-IoU, FPS)"
echo "=============================================================================="
if [ -d "data/floodkaggle/test/images" ]; then
    python3 jetson_deploy/evaluate_jetson.py \
        --model_path outputs/onnx/ours_128x128.onnx \
        --images_dir data/floodkaggle/test/images \
        --masks_dir data/floodkaggle/test/masks \
        --out_json jetson_evaluation_results.json
else
    echo "[INFO] Test dataset directory not found. Running latency profile mode..."
    python3 jetson_deploy/evaluate_jetson.py \
        --model_path outputs/onnx/ours_128x128.onnx \
        --out_json jetson_evaluation_results.json
fi

echo ""
echo "=============================================================================="
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Outputs generated:"
echo "  1. jetson_trtexec_hardware_results.json (Hardware GPU Compute Time & FPS)"
echo "  2. jetson_evaluation_results.json (Accuracy & Boundary-IoU Metrics)"
echo "=============================================================================="
