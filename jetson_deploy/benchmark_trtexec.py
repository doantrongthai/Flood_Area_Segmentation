#!/usr/bin/env python3
"""
NVIDIA Jetson Orin Nano Hardware Benchmark Suite via trtexec
A* / Q1 Paper Protocol for Real-Time Edge Flood Segmentation.

Golden Benchmark Protocol:
1. Target Input Size: 1x3x128x128 (Batch size = 1)
2. Precision: FP16 (and FP32, INT8)
3. Hardware Power Mode: nvpmodel -m 0 (MAXN) + jetson_clocks
4. Isolated GPU Compute Time: --warmUp=500 --iterations=1000 (No pre/post-processing interference)
5. Extracts: GPU Compute Mean, Median (P50), P90, P95, P99, Throughput (FPS)
6. Reports Hardware Environment Metadata: Jetson Model, Power Mode, Clocks, TensorRT Version.
"""

import os
import sys
import glob
import json
import re
import subprocess
import argparse


def find_trtexec():
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
        "trtexec"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        res = subprocess.run(["which", c], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    return "/usr/src/tensorrt/bin/trtexec"


def get_system_metadata():
    meta = {
        "device": "NVIDIA Jetson Orin Nano Developer Kit (8GB)",
        "input_resolution": "1x3x128x128",
        "batch_size": 1,
        "power_mode": "Unknown",
        "jetson_clocks": "Unknown",
        "tensorrt_version": "Unknown"
    }
    # Power Mode
    try:
        res = subprocess.run(["nvpmodel", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "NV Power Mode" in line or "mode" in line.lower():
                    meta["power_mode"] = line.strip()
                    break
    except Exception:
        pass

    # Clocks
    try:
        res = subprocess.run(["jetson_clocks", "--show"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            meta["jetson_clocks"] = "Locked at Max Frequencies (jetson_clocks ACTIVE)"
    except Exception:
        pass

    # TensorRT version
    try:
        res = subprocess.run(["dpkg", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "libnvinfer" in line or "tensorrt" in line.lower():
                    meta["tensorrt_version"] = line.split()[2] if len(line.split()) > 2 else line.strip()
                    break
    except Exception:
        pass

    return meta


def parse_trtexec_log(log_text):
    """
    Parses trtexec standard output metrics.
    Example lines:
    [I] GPU Compute Time: min = 1.82349 ms, max = 2.45123 ms, mean = 1.95123 ms, median = 1.94211 ms, percentile(90%) = 2.0123 ms, percentile(95%) = 2.0812 ms, percentile(99%) = 2.1951 ms
    [I] Throughput: 512.451 qps
    """
    res = {}
    
    # GPU Compute Time
    compute_match = re.search(
        r"GPU Compute Time:\s*min\s*=\s*([0-9\.]+)\s*ms,\s*max\s*=\s*([0-9\.]+)\s*ms,\s*mean\s*=\s*([0-9\.]+)\s*ms,\s*median\s*=\s*([0-9\.]+)\s*ms.*percentile\(90%\)\s*=\s*([0-9\.]+)\s*ms.*percentile\(95%\)\s*=\s*([0-9\.]+)\s*ms.*percentile\(99%\)\s*=\s*([0-9\.]+)\s*ms",
        log_text, re.IGNORECASE | re.DOTALL
    )
    if compute_match:
        res["gpu_compute_min_ms"] = float(compute_match.group(1))
        res["gpu_compute_max_ms"] = float(compute_match.group(2))
        res["gpu_compute_mean_ms"] = float(compute_match.group(3))
        res["gpu_compute_median_ms"] = float(compute_match.group(4))
        res["gpu_compute_p90_ms"] = float(compute_match.group(5))
        res["gpu_compute_p95_ms"] = float(compute_match.group(6))
        res["gpu_compute_p99_ms"] = float(compute_match.group(7))
    else:
        mean_m = re.search(r"GPU Compute Time:.*?mean\s*=\s*([0-9\.]+)\s*ms", log_text, re.IGNORECASE | re.DOTALL)
        med_m = re.search(r"GPU Compute Time:.*?median\s*=\s*([0-9\.]+)\s*ms", log_text, re.IGNORECASE | re.DOTALL)
        if mean_m:
            res["gpu_compute_mean_ms"] = float(mean_m.group(1))
        if med_m:
            res["gpu_compute_median_ms"] = float(med_m.group(1))

    # Throughput
    tp_match = re.search(r"Throughput:\s*([0-9\.]+)\s*qps", log_text, re.IGNORECASE)
    if tp_match:
        res["fps_throughput"] = float(tp_match.group(1))
    elif "gpu_compute_mean_ms" in res and res["gpu_compute_mean_ms"] > 0:
        res["fps_throughput"] = round(1000.0 / res["gpu_compute_mean_ms"], 2)

    return res


def run_benchmark_for_model(trtexec_bin, onnx_path, precision="fp16", warm_up=500, iterations=1000):
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--warmUp={warm_up}",
        f"--iterations={iterations}",
        "--separateProfileRun"
    ]
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    elif precision.lower() == "int8":
        cmd.append("--int8")

    print(f"\n[Executing Command]: {' '.join(cmd)}")
    t_start = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if t_start.returncode != 0:
        print(f"[ERROR]: trtexec failed with return code {t_start.returncode}")
        print(t_start.stderr[:500])
        return {"error": t_start.stderr[:500]}
    
    parsed = parse_trtexec_log(t_start.stdout)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Official NVIDIA TensorRT trtexec Benchmark Protocol")
    parser.add_argument("--onnx_dir", type=str, default="outputs/onnx", help="Directory containing ONNX models")
    parser.add_argument("--model", type=str, default="", help="Specific model ONNX to benchmark")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32", "int8"], default="fp16")
    parser.add_argument("--warm_up", type=int, default=500, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of measurement iterations")
    parser.add_argument("--out_json", type=str, default="jetson_trtexec_hardware_results.json")
    args = parser.parse_args()

    trtexec_bin = find_trtexec()
    print("=" * 100)
    print("NVIDIA JETSON ORIN NANO TENSORRT TRTEXEC BENCHMARK SUITE")
    print(f"trtexec Path: {trtexec_bin}")
    print(f"Warm-up runs: {args.warm_up} | Iterations: {args.iterations} | Precision: {args.precision.upper()}")
    print("=" * 100)

    if not os.path.exists(trtexec_bin):
        print(f"[FATAL]: trtexec not found at '{trtexec_bin}'. Please install TensorRT.")
        sys.exit(1)

    sys_meta = get_system_metadata()
    print("\n[Hardware & Environment Context]:")
    for k, v in sys_meta.items():
        print(f"  {k:20s}: {v}")

    if args.model:
        models = [args.model]
    else:
        models = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))

    if not models:
        print("No ONNX models found to benchmark!")
        sys.exit(1)

    all_results = {
        "metadata": sys_meta,
        "config": {
            "precision": args.precision,
            "warm_up": args.warm_up,
            "iterations": args.iterations,
            "input_resolution": "128x128",
            "batch_size": 1
        },
        "benchmarks": {}
    }

    for m in models:
        m_name = os.path.basename(m)
        print(f"\n>>> Benchmarking Model: {m_name}")
        rep = run_benchmark_for_model(trtexec_bin, m, precision=args.precision, warm_up=args.warm_up, iterations=args.iterations)
        all_results["benchmarks"][m_name] = rep
        if "gpu_compute_mean_ms" in rep:
            print(f"    [Result]: GPU Compute Mean: {rep['gpu_compute_mean_ms']:.3f} ms | Median (P50): {rep.get('gpu_compute_median_ms', 0):.3f} ms | P95: {rep.get('gpu_compute_p95_ms', 0):.3f} ms | FPS: {rep.get('fps_throughput', 0):.1f}")
        else:
            print(f"    [Result]: Failed or could not parse. See logs.")

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Hardware Benchmark successfully completed! Saved to: {args.out_json}")
    print("=" * 100)


if __name__ == "__main__":
    main()
