#!/usr/bin/env python3
"""
Raspberry Pi-specific benchmarking tools for Lyrebird models.

Measures inference latency, CPU usage, and thermal behavior
to validate real-time performance on target hardware.
"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from lyrebird import FiniteImpulseResponseModel
from pi.model_sizes import add_size_arguments, resolve_size_arguments, print_size_info


def get_cpu_temp() -> Optional[float]:
    """Get CPU temperature on Raspberry Pi."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError):
        return None


def get_cpu_freq() -> Optional[float]:
    """Get current CPU frequency in MHz."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r") as f:
            return float(f.read().strip()) / 1000.0
    except (FileNotFoundError, PermissionError):
        return None


def get_memory_usage() -> Dict[str, float]:
    """Get memory usage in MB."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total / 1024 / 1024,
            "used_mb": mem.used / 1024 / 1024,
            "percent": mem.percent,
        }
    except ImportError:
        return {}


def benchmark_model_pytorch(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_iterations: int = 1000,
    warmup: int = 100,
) -> Dict[str, float]:
    """Benchmark PyTorch model inference."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)

    return {
        "mean_us": np.mean(latencies),
        "std_us": np.std(latencies),
        "min_us": np.min(latencies),
        "max_us": np.max(latencies),
        "p50_us": np.percentile(latencies, 50),
        "p95_us": np.percentile(latencies, 95),
        "p99_us": np.percentile(latencies, 99),
    }


def benchmark_model_onnx(
    onnx_path: str,
    input_array: np.ndarray,
    num_iterations: int = 1000,
    warmup: int = 100,
) -> Dict[str, float]:
    """Benchmark ONNX Runtime inference."""
    try:
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        session = ort.InferenceSession(
            onnx_path, sess_options, providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: input_array}

        # Warmup
        for _ in range(warmup):
            _ = session.run(None, ort_inputs)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = session.run(None, ort_inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)

        return {
            "mean_us": np.mean(latencies),
            "std_us": np.std(latencies),
            "min_us": np.min(latencies),
            "max_us": np.max(latencies),
            "p50_us": np.percentile(latencies, 50),
            "p95_us": np.percentile(latencies, 95),
            "p99_us": np.percentile(latencies, 99),
        }

    except ImportError:
        return {"error": "ONNX Runtime not installed"}


def stress_test(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    duration_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Run continuous inference for a duration to test thermal throttling.

    Returns statistics including temperature over time.
    """
    model.eval()

    samples: List[Dict[str, Any]] = []
    start_time = time.time()

    print(f"Running stress test for {duration_seconds} seconds...")

    with torch.no_grad():
        while time.time() - start_time < duration_seconds:
            # Run batch of inferences
            batch_start = time.perf_counter()
            for _ in range(100):
                _ = model(input_tensor)
            batch_end = time.perf_counter()

            # Record sample
            samples.append(
                {
                    "elapsed": time.time() - start_time,
                    "batch_time_ms": (batch_end - batch_start) * 1000,
                    "temp_c": get_cpu_temp(),
                    "freq_mhz": get_cpu_freq(),
                }
            )

            # Brief pause between batches
            time.sleep(0.01)

    # Analyze results
    batch_times = [s["batch_time_ms"] for s in samples]
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]

    return {
        "duration_s": duration_seconds,
        "num_samples": len(samples),
        "batch_time_mean_ms": np.mean(batch_times),
        "batch_time_max_ms": np.max(batch_times),
        "temp_start_c": temps[0] if len(temps) > 0 else None,
        "temp_end_c": temps[-1] if len(temps) > 0 else None,
        "temp_max_c": max(temps) if len(temps) > 0 else None,
        "throttled": max(temps) >= 80 if len(temps) > 0 else None,
        "samples": samples,
    }


def print_system_info() -> None:
    """Print system information relevant to performance."""
    print("System Information")
    print("=" * 60)

    # Platform
    import platform

    print(f"Platform: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CPU
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"CPU: {line.split(':')[1].strip()}")
                    break
    except FileNotFoundError:
        # /proc/cpuinfo may not exist on non-Linux systems; skip CPU model info
        pass

    # Temperature
    temp = get_cpu_temp()
    if temp:
        print(f"CPU Temperature: {temp:.1f}°C")

    # Frequency
    freq = get_cpu_freq()
    if freq:
        print(f"CPU Frequency: {freq:.0f} MHz")

    # Memory
    mem = get_memory_usage()
    if mem:
        print(
            f"Memory: {mem['used_mb']:.0f} / {mem['total_mb']:.0f} MB ({mem['percent']:.1f}%)"
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi benchmark for Lyrebird models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Path to trained model (.pth)"
    )
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model")

    # Add model size arguments (--size or manual --buffer-length etc.)
    add_size_arguments(parser, default_size="small")

    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,64,128",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--stress-test", type=int, default=0, help="Run stress test for N seconds"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for budget calculations",
    )

    args = parser.parse_args()

    # Print system info
    print_system_info()

    # Resolve model size arguments
    buffer_length, hidden_size, num_layers = resolve_size_arguments(args)

    # Print model size info
    if args.size:
        print_size_info(size=args.size)
    else:
        print_size_info(
            buffer_length=buffer_length, hidden_size=hidden_size, num_layers=num_layers
        )
    print()

    # Calculate budgets
    per_sample_budget = 1e6 / args.sample_rate
    print(
        f"Real-time budget at {args.sample_rate} Hz: {per_sample_budget:.2f} μs/sample"
    )
    print()

    # Create or load model
    input_size = buffer_length * args.channels
    output_size = args.channels

    model = FiniteImpulseResponseModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    )

    if args.model and os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded model: {args.model}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Parse batch sizes
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    # PyTorch benchmarks
    print("PyTorch Inference Benchmark")
    print("-" * 60)
    print(f"{'Batch':>8} {'Mean':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'RT':>6}")
    print(f"{'':>8} {'(μs)':>10} {'(μs)':>10} {'(μs)':>10} {'(μs)':>10} {'':>6}")

    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, args.channels, buffer_length)
        stats = benchmark_model_pytorch(model, input_tensor, args.iterations)

        per_sample = stats["mean_us"] / batch_size
        rt_ok = "✓" if per_sample < per_sample_budget else "✗"

        print(
            f"{batch_size:>8} {stats['mean_us']:>10.2f} {stats['p95_us']:>10.2f} "
            f"{stats['p99_us']:>10.2f} {stats['max_us']:>10.2f} {rt_ok:>6}"
        )

    print()

    # ONNX benchmarks
    if args.onnx and os.path.exists(args.onnx):
        print("ONNX Runtime Inference Benchmark")
        print("-" * 60)
        print(
            f"{'Batch':>8} {'Mean':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'RT':>6}"
        )
        print(f"{'':>8} {'(μs)':>10} {'(μs)':>10} {'(μs)':>10} {'(μs)':>10} {'':>6}")

        for batch_size in batch_sizes:
            input_array = np.random.randn(
                batch_size, args.channels, buffer_length
            ).astype(np.float32)
            stats = benchmark_model_onnx(args.onnx, input_array, args.iterations)

            if "error" in stats:
                print(f"{batch_size:>8} {stats['error']}")
                continue

            per_sample = stats["mean_us"] / batch_size
            rt_ok = "✓" if per_sample < per_sample_budget else "✗"

            print(
                f"{batch_size:>8} {stats['mean_us']:>10.2f} {stats['p95_us']:>10.2f} "
                f"{stats['p99_us']:>10.2f} {stats['max_us']:>10.2f} {rt_ok:>6}"
            )

        print()

    # Stress test
    if args.stress_test > 0:
        print(f"Thermal Stress Test ({args.stress_test}s)")
        print("-" * 60)

        input_tensor = torch.randn(1, args.channels, buffer_length)
        results = stress_test(model, input_tensor, args.stress_test)

        print(f"Duration: {results['duration_s']}s")
        print(
            f"Batch time: {results['batch_time_mean_ms']:.2f}ms avg, "
            f"{results['batch_time_max_ms']:.2f}ms max"
        )

        if results["temp_start_c"]:
            print(
                f"Temperature: {results['temp_start_c']:.1f}°C → "
                f"{results['temp_end_c']:.1f}°C (max: {results['temp_max_c']:.1f}°C)"
            )
            if results["throttled"]:
                print("WARNING: CPU may have throttled (temp >= 80°C)")
            else:
                print("No thermal throttling detected")


if __name__ == "__main__":
    main()
