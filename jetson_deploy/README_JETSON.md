# Jetson deployment and benchmark

Measurements use the training setup, so they can be compared with the training-machine results:

| Item | Setting |
|---|---|
| Input | 1 x 3 x 256 x 256 |
| Preprocessing | RGB, bilinear resize, divide by 255 (no ImageNet normalisation) |
| Output | logits, sigmoid, threshold 0.5 |
| Masks | `floodscene`: value 255 is flood; `floodkaggle`: value > 127 |
| Accuracy | flood-class IoU and Dice pooled over all test pixels (as `utils/metrics.py`) |
| Precision | FP16 or FP32 (state which one in any report) |

## 1. Export on the training machine

Export one file per dataset and seed, from the trained checkpoints:

```bash
python export.py --model ours \
    --weights outputs/ours/seed_42/ours_hard_afm_floodkaggle_s42.pth \
    --output outputs/onnx/ours_floodkaggle_s42_256x256.onnx
```

Always pass `--weights`. Without it the file has random weights and is only good for latency.
The exporter checks that ONNX Runtime reproduces the PyTorch output.

For a latency comparison, export the baselines the same way (for example `enet`, `fast-scnn`,
`fasegnet`, `edanet`) and keep the `_256x256.onnx` suffix. ENet uses max-unpooling; check
that `trtexec` accepts it.

## 2. Libraries on the Jetson

See `requirements-jetson.txt`. Copy `outputs/onnx/`, `floodkaggle/test/` and `floodscene/test/`
to the board.

## 3. Run

```bash
bash jetson_deploy/run_official_benchmark.sh                  # FP16
PRECISION=fp32 bash jetson_deploy/run_official_benchmark.sh   # FP32
```

Or step by step:

```bash
sudo jetson_clocks
python3 jetson_deploy/benchmark_trtexec.py --onnx_dir outputs/onnx --precision fp16

python3 jetson_deploy/evaluate_jetson.py \
    --model_path outputs/onnx/ours_floodkaggle_s*_256x256.onnx \
    --images_dir floodkaggle/test/images --masks_dir floodkaggle/test/masks \
    --dataset_type floodkaggle --precision fp16
```

With several `--model_path` files (the five seeds) the script also prints mean and standard
deviation of IoU and Dice.

## 4. What each number means

- `trtexec` (`benchmark_trtexec.py`): GPU compute time only, without pre- or post-processing.
- `inference_ms` (`evaluate_jetson.py`): one ONNX Runtime call per image.
- `end_to_end_ms`: preprocessing + inference + threshold. Reading the image from disk is excluded.

Report with every latency: board and memory size, JetPack / TensorRT version, power mode
(`nvpmodel -q`), precision and input size. The result files record these as the board reports them.
