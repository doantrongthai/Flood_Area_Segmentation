import os
import sys
import argparse
import torch
import torch.nn as nn
from models import get_model


class DeployableModel(nn.Module):
    """
    Strips auxiliary heads and ensures clean single-tensor output for 
    clean ONNX / TensorRT / MCU deployment.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        elif isinstance(out, dict):
            # return standard output key if dict
            for key in ['out', 'output', 'logits', 'pred']:
                if key in out:
                    return out[key]
            return list(out.values())[0]
        return out


def export_single_onnx(model_name, weights_path=None, output_path=None, input_size=128, num_classes=1, opset=13, seed=42):
    """
    Exports a single architecture to ONNX format optimized for TensorRT / Edge compilation.
    """
    if output_path is None:
        os.makedirs('outputs/onnx', exist_ok=True)
        output_path = f'outputs/onnx/{model_name}_{input_size}x{input_size}.onnx'
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    print("=" * 70)
    print("EDGE DEPLOYMENT EXPORTER (ONNX) - JETSON ORIN NANO")
    print("=" * 70)
    print(f"Model Architecture : {model_name}")
    print(f"Weights Checkpoint : {weights_path if weights_path else 'None (Profiling / Architecture mode)'}")
    print(f"Input Resolution   : 1x3x{input_size}x{input_size}")
    print(f"Output ONNX Path   : {output_path}")
    print(f"ONNX Opset         : {opset}")
    print("=" * 70)

    # 1. Initialize Model
    model = get_model(model_name, num_classes=num_classes, seed=seed, input_size=input_size)

    # 2. Load Checkpoint if provided
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Strip module. prefix if trained with DataParallel/DDP
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith('module.') else k
            clean_state_dict[clean_k] = v

        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print(f"[OK] Checkpoint weights loaded successfully (strict match).")
        except Exception as e:
            print(f"[WARN] Strict loading failed: {e}. Attempting non-strict loading...")
            model.load_state_dict(clean_state_dict, strict=False)
            print(f"[OK] Checkpoint weights loaded with strict=False.")
    else:
        print(f"[INFO] Initializing model with seed {seed} for latency / FLOPs edge profiling.")

    model.eval()
    deploy_model = DeployableModel(model)
    deploy_model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    # 3. Export to ONNX
    try:
        export_kwargs = {
            'export_params': True,
            'opset_version': opset,
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output']
        }
        # Avoid Dynamo unrolling hang on recursive/loop SSM architectures
        try:
            torch.onnx.export(deploy_model, dummy_input, output_path, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(deploy_model, dummy_input, output_path, **export_kwargs)
        print(f"[OK] ONNX export succeeded: {output_path}")

        # 4. Optional onnx.checker verification
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("[OK] ONNX model verified with onnx.checker.")
        except ImportError:
            pass
        except Exception as check_err:
            print(f"[WARN] onnx.checker notice: {check_err}")

        file_size_kb = os.path.getsize(output_path) / 1024.0
        print(f"ONNX File Size: {file_size_kb:.2f} KB ({file_size_kb / 1024.0:.2f} MB)")
        return output_path

    except Exception as e:
        print(f"[ERROR] ONNX export failed for {model_name}: {e}")
        return None


def export_all_models(models_dir='models', output_dir='outputs/onnx', input_size=128, opset=13):
    """
    Batch exports all baseline and proposed models to ONNX for edge benchmark comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    available_models = [
        f.replace('.py', '')
        for f in os.listdir(models_dir)
        if f.endswith('.py') and f != '__init__.py' and not f.startswith('abl_')
    ]

    print(f"\n[INFO] Found {len(available_models)} core architectures to export: {available_models}\n")
    results = {}
    for name in sorted(available_models):
        out_path = os.path.join(output_dir, f"{name}_{input_size}x{input_size}.onnx")
        try:
            p = export_single_onnx(
                model_name=name,
                weights_path=None,
                output_path=out_path,
                input_size=input_size,
                num_classes=1,
                opset=opset
            )
            results[name] = "SUCCESS" if p else "FAILED"
        except Exception as ex:
            print(f"[ERROR] Export failed for {name}: {ex}")
            results[name] = f"ERROR: {ex}"

    print("\n" + "=" * 70)
    print("BATCH EXPORT SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Export Flood Segmentation Models to ONNX for Jetson Orin Nano Benchmark")
    parser.add_argument('--model', type=str, default='mambafloodlite', help="Architecture name (e.g., mambafloodlite, ours)")
    parser.add_argument('--weights', type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument('--size', type=int, default=128, help="Input resolution (default: 128 for real-time edge)")
    parser.add_argument('--num_classes', type=int, default=1, help="Output channels (default: 1 for binary flood mask)")
    parser.add_argument('--output', type=str, default=None, help="Target path for ONNX file")
    parser.add_argument('--opset', type=int, default=13, help="ONNX Opset version (13 or 17 recommended for TensorRT)")
    parser.add_argument('--export_all', action='store_true', help="Batch export all core models in models/")
    args = parser.parse_args()

    if args.export_all:
        export_all_models(input_size=args.size, opset=args.opset)
    else:
        export_single_onnx(
            model_name=args.model,
            weights_path=args.weights,
            output_path=args.output,
            input_size=args.size,
            num_classes=args.num_classes,
            opset=args.opset
        )


if __name__ == '__main__':
    main()
