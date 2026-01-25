#!/usr/bin/env python3
"""
Export Lyrebird models to ONNX format for optimized inference.

ONNX Runtime can provide better performance on CPU, especially
on ARM devices like Raspberry Pi.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyrebird import FiniteImpulseResponseModel


def export_to_onnx(
    model_path: str,
    output_path: str,
    buffer_length: int,
    hidden_size: int,
    num_layers: int,
    channels: int = 1,
    batch_size: int = 1,
    opset_version: int = 17,
) -> None:
    """
    Export a trained model to ONNX format.

    Args:
        model_path: Path to trained .pth model
        output_path: Path for output .onnx file
        buffer_length: Model's buffer length
        hidden_size: Model's hidden size
        num_layers: Model's number of layers
        channels: Number of audio channels
        batch_size: Batch size for export (use 1 for real-time, larger for batch processing)
        opset_version: ONNX opset version
    """
    print(f"Loading model: {model_path}")

    # Create model
    input_size = buffer_length * channels
    output_size = channels

    model = FiniteImpulseResponseModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    )

    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, buffer_length)

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Opset version: {opset_version}")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        } if batch_size == 1 else None,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print("Export complete!")

    # Verify the export
    verify_onnx(output_path, dummy_input, model)


def verify_onnx(onnx_path: str, dummy_input: torch.Tensor,
                pytorch_model: torch.nn.Module) -> None:
    """Verify ONNX model produces same output as PyTorch."""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        print("\nVerifying ONNX model...")

        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model validation: PASSED")

        # Run inference with ONNX Runtime
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Compare outputs
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()

        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = session.run(None, ort_inputs)[0]

        # Check numerical accuracy
        max_diff = np.abs(pytorch_output - onnx_output).max()
        print(f"  Max output difference: {max_diff:.2e}")

        if max_diff < 1e-5:
            print("  Numerical accuracy: PASSED")
        else:
            print("  Numerical accuracy: WARNING - difference > 1e-5")

        # Benchmark
        print("\nBenchmarking ONNX inference...")
        import time

        # Warmup
        for _ in range(100):
            session.run(None, ort_inputs)

        # Benchmark
        num_iterations = 1000
        start = time.perf_counter()
        for _ in range(num_iterations):
            session.run(None, ort_inputs)
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / num_iterations) * 1e6
        print(f"  Average inference time: {avg_us:.2f} μs")

        # Real-time budget
        per_sample_budget = 1e6 / 44100  # ~22.7 μs at 44.1kHz
        batch_size = dummy_input.shape[0]
        per_sample_us = avg_us / batch_size

        if per_sample_us < per_sample_budget:
            headroom = per_sample_budget - per_sample_us
            print(f"  Real-time capable: YES (headroom: {headroom:.2f} μs/sample)")
        else:
            overhead = per_sample_us - per_sample_budget
            print(f"  Real-time capable: NO (over budget by {overhead:.2f} μs/sample)")

    except ImportError as e:
        print(f"\nSkipping verification: {e}")
        print("Install onnx and onnxruntime for verification")


def optimize_onnx(input_path: str, output_path: str) -> None:
    """
    Apply additional ONNX optimizations.

    Args:
        input_path: Path to input ONNX model
        output_path: Path for optimized output
    """
    try:
        from onnxruntime.transformers import optimizer
        import onnx

        print(f"\nApplying ONNX optimizations...")

        # Load model
        model = onnx.load(input_path)

        # Apply optimizations
        optimized = optimizer.optimize_model(
            input_path,
            model_type='bert',  # Generic optimization
            num_heads=0,
            hidden_size=0,
        )

        optimized.save_model_to_file(output_path)
        print(f"Optimized model saved to: {output_path}")

    except ImportError:
        print("onnxruntime-tools not installed, skipping optimization")


def main():
    parser = argparse.ArgumentParser(
        description='Export Lyrebird models to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a trained model
  python export_onnx.py --model model.pth --buffer-length 128 --hidden-size 64

  # Export with custom output path
  python export_onnx.py --model model.pth -o optimized.onnx --buffer-length 128 --hidden-size 64

  # Export for batched inference
  python export_onnx.py --model model.pth --batch-size 64 --buffer-length 128 --hidden-size 64
        """
    )

    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to trained model (.pth)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for ONNX model (default: same as input with .onnx)')
    parser.add_argument('--buffer-length', type=int, required=True,
                        help='Model buffer length')
    parser.add_argument('--hidden-size', type=int, required=True,
                        help='Model hidden size')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Model number of layers')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply additional ONNX optimizations')

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = args.model.replace('.pth', '.onnx')

    # Export
    export_to_onnx(
        model_path=args.model,
        output_path=output_path,
        buffer_length=args.buffer_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        channels=args.channels,
        batch_size=args.batch_size,
        opset_version=args.opset,
    )

    # Optional optimization
    if args.optimize:
        opt_path = output_path.replace('.onnx', '_optimized.onnx')
        optimize_onnx(output_path, opt_path)


if __name__ == '__main__':
    main()
