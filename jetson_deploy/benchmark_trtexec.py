#!/usr/bin/env python3
"""
GPU compute time of ONNX models on NVIDIA Jetson, measured with TensorRT's trtexec.

The reported time covers GPU execution only: image loading, preprocessing and
postprocessing are NOT included (use evaluate_jetson.py for those).

Only models named *_<size>x<size>.onnx for the requested size are benchmarked, so files of
different resolutions are never mixed. The raw trtexec log of every model is kept.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys


def find_trtexec():
    for candidate in ("/usr/src/tensorrt/bin/trtexec", shutil.which("trtexec")):
        if candidate and os.path.exists(candidate):
            return candidate
    sys.exit("[ERROR] trtexec not found. It ships with JetPack (/usr/src/tensorrt/bin/trtexec).")


def run(cmd):
    """Output of a system query, or None. Nothing is assumed if the command fails."""
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return res.stdout.strip() if res.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return None


def system_metadata():
    """Recorded as reported by the board; run with sudo if jetson_clocks shows 'unavailable'."""
    return {
        "device": read("/proc/device-tree/model") or "unavailable",
        "l4t_release": read("/etc/nv_tegra_release") or "unavailable",
        "tensorrt_packages": run(["dpkg-query", "-W", "-f=${Package} ${Version}\n", "libnvinfer*"]) or "unavailable",
        "nvpmodel": run(["nvpmodel", "-q"]) or "unavailable",
        "jetson_clocks": run(["jetson_clocks", "--show"]) or "unavailable",
    }


def parse_log(text):
    res = {}
    line = re.search(r"GPU Compute Time:(.*)", text)
    if line:
        body = line.group(1)
        for key in ("min", "max", "mean", "median"):
            m = re.search(rf"\b{key}\s*=\s*([0-9.]+)\s*ms", body)
            if m:
                res[f"gpu_compute_{key}_ms"] = float(m.group(1))
        for pct, value in re.findall(r"percentile\((\d+)%\)\s*=\s*([0-9.]+)\s*ms", body):
            res[f"gpu_compute_p{pct}_ms"] = float(value)
    m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", text)
    if m:
        res["throughput_qps"] = float(m.group(1))
    return res


def benchmark(trtexec, onnx_path, precision, warm_up_ms, iterations, log_dir):
    cmd = [trtexec, f"--onnx={onnx_path}", f"--warmUp={warm_up_ms}", f"--iterations={iterations}",
           "--separateProfileRun"]
    if precision == "fp16":
        cmd.append("--fp16")
    print(f"[Command] {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    name = os.path.splitext(os.path.basename(onnx_path))[0]
    log_path = os.path.join(log_dir, f"{name}_{precision}.log")
    with open(log_path, "w") as f:
        f.write(res.stdout)
    parsed = parse_log(res.stdout)
    if res.returncode != 0 or "gpu_compute_mean_ms" not in parsed:
        return {"error": f"trtexec failed (code {res.returncode}), see {log_path}"}
    parsed["log"] = log_path
    return parsed


def main():
    ap = argparse.ArgumentParser(description="trtexec benchmark of exported ONNX models")
    ap.add_argument("--onnx_dir", default="outputs/onnx")
    ap.add_argument("--model", nargs="+", help="specific ONNX files instead of --onnx_dir")
    ap.add_argument("--size", type=int, default=256, help="input resolution (default: 256, the training size)")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                    help="INT8 is not offered: it needs a calibrated, quantised model")
    ap.add_argument("--warm_up_ms", type=int, default=2000, help="trtexec --warmUp is a time in milliseconds")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--out_json", default="jetson_trtexec_results.json")
    args = ap.parse_args()

    suffix = f"_{args.size}x{args.size}.onnx"
    found = args.model or sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
    models = [m for m in found if m.endswith(suffix)]
    for skipped in sorted(set(found) - set(models)):
        print(f"[SKIP] {skipped}: name does not end in {suffix}")
    if not models:
        sys.exit(f"[ERROR] No *{suffix} models. Export with: python export.py --size {args.size} ...")

    trtexec = find_trtexec()
    log_dir = os.path.splitext(args.out_json)[0] + "_logs"
    os.makedirs(log_dir, exist_ok=True)
    results = {
        "environment": system_metadata(),
        "config": {"input": f"1x3x{args.size}x{args.size}", "precision": args.precision,
                   "warm_up_ms": args.warm_up_ms, "iterations": args.iterations,
                   "measured": "GPU compute time only (no pre/post-processing)"},
        "benchmarks": {},
    }
    failed = 0
    for m in models:
        r = benchmark(trtexec, m, args.precision, args.warm_up_ms, args.iterations, log_dir)
        results["benchmarks"][os.path.basename(m)] = r
        if "error" in r:
            failed += 1
            print(f"  [FAILED] {r['error']}")
        else:
            print(f"  mean {r['gpu_compute_mean_ms']:.3f} ms | median {r.get('gpu_compute_median_ms', float('nan')):.3f} ms")

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.out_json} ({len(models) - failed} ok, {failed} failed)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
