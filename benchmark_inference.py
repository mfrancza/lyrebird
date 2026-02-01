"""
Inference benchmark script for Lyrebird models.

Measures forward pass latency for different model configurations to help
determine feasibility for real-time audio processing on embedded hardware.

At 44.1kHz sample rate:
- Per sample budget: ~22.7μs
- 64 sample block: ~1.45ms
- 128 sample block: ~2.9ms
- 256 sample block: ~5.8ms
"""

import torch
import time
import argparse
from lyrebird import FiniteImpulseResponseModel


def benchmark_model(model, input_tensor, num_iterations=1000, warmup=100):
    """
    Benchmark forward pass latency.

    Returns:
        dict with mean, min, max, std latency in microseconds
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Synchronize if CUDA
    if input_tensor.is_cuda:
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)  # Convert to microseconds

    return {
        'mean_us': sum(latencies) / len(latencies),
        'min_us': min(latencies),
        'max_us': max(latencies),
        'std_us': (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies)) ** 0.5
    }


def calculate_realtime_budget(sample_rate=44100):
    """Calculate time budgets for real-time processing."""
    per_sample_us = 1e6 / sample_rate
    return {
        'per_sample_us': per_sample_us,
        'block_64_ms': 64 * per_sample_us / 1000,
        'block_128_ms': 128 * per_sample_us / 1000,
        'block_256_ms': 256 * per_sample_us / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Lyrebird model inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to benchmark on: cpu, cuda, or auto')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of benchmark iterations')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Sample rate for budget calculations')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Lyrebird Inference Benchmark")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Channels: {args.channels}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print()

    # Real-time budgets
    budgets = calculate_realtime_budget(args.sample_rate)
    print(f"Real-time budgets at {args.sample_rate} Hz:")
    print(f"  Per sample: {budgets['per_sample_us']:.2f} μs")
    print(f"  64-sample block: {budgets['block_64_ms']:.2f} ms")
    print(f"  128-sample block: {budgets['block_128_ms']:.2f} ms")
    print(f"  256-sample block: {budgets['block_256_ms']:.2f} ms")
    print()

    # Model configurations to test for exploration/benchmarking.
    # Note: These are exploration configs and differ from the deployment presets:
    #   - Deployment presets (pi/model_sizes.py, VST plugin):
    #     small=128/32/2, medium=256/64/2, large=512/128/3
    #   - These exploration configs test a wider range for performance analysis
    configs = [
        # (buffer_length, hidden_size, num_layers, description)
        (128, 32, 1, "Tiny"),
        (128, 32, 2, "Tiny-deep"),      # Same as deployment "small"
        (128, 64, 1, "Small"),
        (128, 64, 2, "Small-deep"),
        (128, 128, 1, "Medium"),
        (128, 128, 2, "Medium-deep"),
        (256, 64, 1, "Small-wide"),
        (256, 64, 2, "Small-wide-deep"),  # Same as deployment "medium"
        (256, 128, 1, "Medium-wide"),
        (256, 256, 2, "Large"),
        (512, 128, 2, "Large-wide"),
        (512, 128, 3, "Large-wide-deep"),  # Same as deployment "large"
        (1024, 256, 4, "XLarge"),
    ]

    # Batch sizes to test
    batch_sizes = [1, 64, 128, 256]

    print(f"{'Config':<15} {'Params':>10} {'Batch':>6} {'Mean':>10} {'Min':>10} {'Max':>10} {'Realtime':>10}")
    print(f"{'':15} {'':>10} {'':>6} {'(μs)':>10} {'(μs)':>10} {'(μs)':>10} {'':>10}")
    print("-" * 83)

    results = []

    for buffer_length, hidden_size, num_layers, desc in configs:
        input_size = buffer_length * args.channels
        output_size = args.channels

        model = FiniteImpulseResponseModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, args.channels, buffer_length).to(device)

            stats = benchmark_model(model, input_tensor, num_iterations=args.iterations)

            # Calculate per-sample latency for batched inference
            per_sample_us = stats['mean_us'] / batch_size

            # Check if real-time capable (per-sample must be < budget)
            realtime = "✓" if per_sample_us < budgets['per_sample_us'] else "✗"

            print(f"{desc:<15} {num_params:>10,} {batch_size:>6} {stats['mean_us']:>10.2f} "
                  f"{stats['min_us']:>10.2f} {stats['max_us']:>10.2f} {realtime:>10}")

            results.append({
                'config': desc,
                'buffer_length': buffer_length,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'params': num_params,
                'batch_size': batch_size,
                'mean_us': stats['mean_us'],
                'per_sample_us': per_sample_us,
                'realtime_capable': per_sample_us < budgets['per_sample_us']
            })

        print()  # Blank line between configs

    # Summary
    print("=" * 83)
    print("\nSummary: Real-time capable configurations (batch=1, per-sample < budget)")
    print("-" * 60)

    realtime_configs = [r for r in results if r['batch_size'] == 1 and r['realtime_capable']]
    if realtime_configs:
        for r in realtime_configs:
            headroom = budgets['per_sample_us'] - r['per_sample_us']
            print(f"  {r['config']:<15} {r['params']:>8,} params, {r['mean_us']:>8.2f} μs "
                  f"(headroom: {headroom:.2f} μs)")
    else:
        print("  No configurations achieved real-time at batch_size=1 on this device.")
        print("  Consider using batched processing or faster hardware.")

    print("\nNote: Embedded hardware (Raspberry Pi, ESP32, Teensy) will be slower than this benchmark.")
    print("      Use these numbers as a baseline; expect 2-10x slower on target hardware.")


if __name__ == '__main__':
    main()
