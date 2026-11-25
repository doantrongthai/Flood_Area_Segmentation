#!/usr/bin/env bash
# ==============================================================================
# Jetson benchmark runner. Run from the repository root on the Jetson.
#
# Expects ONNX files exported on the training machine, named
#   outputs/onnx/<model>_<dataset>_s<seed>_256x256.onnx
# and the test sets in floodkaggle/test and floodscene/test.
#
# Settings (environment variables):
#   MODEL=ours  PRECISION=fp16|fp32  SIZE=256  ONNX_DIR=outputs/onnx  OUT=jetson_results
#   POWER_MODE=<id>  optional nvpmodel mode. IDs differ between boards and JetPack
#                    versions (see /etc/nvpmodel.conf), so none is forced by default.
# ==============================================================================
set -euo pipefail

MODEL=${MODEL:-ours}
PRECISION=${PRECISION:-fp16}
SIZE=${SIZE:-256}
ONNX_DIR=${ONNX_DIR:-outputs/onnx}
OUT=${OUT:-jetson_results}
mkdir -p "$OUT"

echo "== STEP 1: power mode and clocks"
if [ -n "${POWER_MODE:-}" ]; then
    sudo nvpmodel -m "$POWER_MODE"
fi
sudo jetson_clocks
sudo nvpmodel -q | tee "$OUT/power_mode.txt"
sudo jetson_clocks --show > "$OUT/clocks.txt"

echo "== STEP 2: trtexec GPU compute time (${PRECISION}, ${SIZE}x${SIZE}, batch 1)"
python3 jetson_deploy/benchmark_trtexec.py --onnx_dir "$ONNX_DIR" --size "$SIZE" \
    --precision "$PRECISION" --out_json "$OUT/trtexec_${PRECISION}.json"

echo "== STEP 3: test-set accuracy and latency with ONNX Runtime + TensorRT"
shopt -s nullglob
for DS in floodkaggle floodscene; do
    FILES=("$ONNX_DIR/${MODEL}_${DS}_"*"${SIZE}x${SIZE}.onnx")
    if [ ${#FILES[@]} -eq 0 ]; then
        echo "[SKIP] ${DS}: no ${MODEL}_${DS}_*${SIZE}x${SIZE}.onnx in ${ONNX_DIR}"
        continue
    fi
    if [ ! -d "${DS}/test/images" ]; then
        echo "[SKIP] ${DS}: ${DS}/test/images not found"
        continue
    fi
    python3 jetson_deploy/evaluate_jetson.py --model_path "${FILES[@]}" \
        --images_dir "${DS}/test/images" --masks_dir "${DS}/test/masks" --dataset_type "$DS" \
        --provider tensorrt --precision "$PRECISION" --out_json "$OUT/eval_${DS}_${PRECISION}.json"
done

echo "== Done. Results are in ${OUT}/"
