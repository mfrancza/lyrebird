#!/usr/bin/env python3
"""
Latency measurement tests for the Lyrebird real-time processor.

Measures round-trip audio latency using loopback connection and
impulse response detection.
"""

import sys
from pathlib import Path
import time
import argparse
from typing import Optional, Tuple
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pi.audio_io import AudioConfig, AudioIO, print_devices


def measure_loopback_latency(
    config: AudioConfig,
    duration: float = 2.0,
    impulse_interval: float = 0.5,
) -> Tuple[Optional[float], dict]:
    """
    Measure round-trip latency using loopback.

    Sends impulses and detects them in the input to measure total
    system latency (ADC + processing + DAC).

    IMPORTANT: Connect audio output to audio input for this test.

    Args:
        config: Audio configuration
        duration: Test duration in seconds
        impulse_interval: Time between impulses in seconds

    Returns:
        Tuple of (latency_ms, details_dict)
    """
    sample_rate = config.sample_rate
    block_size = config.block_size
    channels = config.channels

    # Recording buffer
    max_samples = int(duration * sample_rate * 1.5)
    recorded = np.zeros((max_samples, channels), dtype=np.float32)
    sent = np.zeros((max_samples, channels), dtype=np.float32)
    record_pos = [0]

    # Impulse state
    samples_per_impulse = int(impulse_interval * sample_rate)
    impulse_sample = [0]
    impulse_positions = []

    def callback(indata: np.ndarray) -> np.ndarray:
        """Audio callback that generates impulses and records."""
        pos = record_pos[0]
        block_len = len(indata)

        # Record input
        if pos + block_len <= max_samples:
            recorded[pos:pos + block_len] = indata

        # Generate output with periodic impulses
        outdata = np.zeros_like(indata)

        for i in range(block_len):
            if impulse_sample[0] == 0:
                # Generate impulse (short click)
                outdata[i, :] = 0.8
                impulse_positions.append(pos + i)
            impulse_sample[0] = (impulse_sample[0] + 1) % samples_per_impulse

        # Record what we sent
        if pos + block_len <= max_samples:
            sent[pos:pos + block_len] = outdata

        record_pos[0] = pos + block_len
        return outdata

    # Run test
    print(f"Measuring loopback latency...")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Block size: {block_size}")
    print(f"  Impulse interval: {impulse_interval}s")
    print()
    print("NOTE: Connect audio output to input for this test!")
    print()

    audio = AudioIO(config, callback)
    audio.start()

    time.sleep(duration)

    audio.stop()

    # Trim to actual recorded length
    actual_length = min(record_pos[0], max_samples)
    recorded = recorded[:actual_length]
    sent = sent[:actual_length]

    # Find impulses in recorded signal
    if len(impulse_positions) == 0:
        return None, {'error': 'No impulses generated'}

    # Use cross-correlation to find delay
    # Flatten to mono for analysis
    sent_mono = sent.mean(axis=1) if channels > 1 else sent.flatten()
    recorded_mono = recorded.mean(axis=1) if channels > 1 else recorded.flatten()

    # Normalize
    sent_mono = sent_mono / (np.abs(sent_mono).max() + 1e-10)
    recorded_mono = recorded_mono / (np.abs(recorded_mono).max() + 1e-10)

    # Check if we got any signal
    if np.abs(recorded_mono).max() < 0.01:
        return None, {
            'error': 'No signal detected in recording',
            'hint': 'Check loopback connection'
        }

    # Cross-correlate
    correlation = np.correlate(recorded_mono, sent_mono, mode='full')
    lag_samples = np.argmax(correlation) - len(sent_mono) + 1

    # Sanity check
    if lag_samples < 0 or lag_samples > sample_rate:  # Max 1 second
        return None, {
            'error': f'Invalid lag detected: {lag_samples} samples',
            'correlation_max': float(correlation.max()),
        }

    latency_ms = (lag_samples / sample_rate) * 1000

    details = {
        'lag_samples': lag_samples,
        'latency_ms': latency_ms,
        'sample_rate': sample_rate,
        'block_size': block_size,
        'num_impulses': len(impulse_positions),
        'recorded_samples': actual_length,
        'correlation_max': float(correlation.max()),
        'theoretical_min_ms': (block_size * 2 / sample_rate) * 1000,
    }

    return latency_ms, details


def measure_callback_latency(
    config: AudioConfig,
    duration: float = 5.0,
) -> dict:
    """
    Measure callback timing characteristics.

    Returns statistics about callback execution time and jitter.

    Args:
        config: Audio configuration
        duration: Test duration in seconds

    Returns:
        Dictionary with timing statistics
    """
    callback_times = []
    last_callback_time = [None]

    def callback(indata: np.ndarray) -> np.ndarray:
        current_time = time.perf_counter()
        if last_callback_time[0] is not None:
            delta = current_time - last_callback_time[0]
            callback_times.append(delta * 1000)  # ms
        last_callback_time[0] = current_time
        return indata

    print(f"Measuring callback timing for {duration}s...")

    audio = AudioIO(config, callback)
    audio.start()

    time.sleep(duration)

    audio.stop()

    if len(callback_times) < 10:
        return {'error': 'Not enough callbacks recorded'}

    times = np.array(callback_times)
    expected_interval = (config.block_size / config.sample_rate) * 1000

    return {
        'expected_interval_ms': expected_interval,
        'mean_interval_ms': float(np.mean(times)),
        'std_interval_ms': float(np.std(times)),
        'min_interval_ms': float(np.min(times)),
        'max_interval_ms': float(np.max(times)),
        'jitter_ms': float(np.std(times)),
        'num_callbacks': len(callback_times),
        'audio_stats': audio.get_stats(),
    }


def run_processor_latency_test(
    model_path: str,
    buffer_length: int,
    hidden_size: int,
    num_layers: int,
    config: AudioConfig,
    duration: float = 5.0,
) -> dict:
    """
    Test latency with actual model processing.

    Args:
        model_path: Path to model
        buffer_length: Model buffer length
        hidden_size: Model hidden size
        num_layers: Model layers
        config: Audio configuration
        duration: Test duration

    Returns:
        Processing statistics
    """
    from pi.realtime_processor import RealtimeProcessor

    processor = RealtimeProcessor(
        model_path=model_path,
        buffer_length=buffer_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        sample_rate=config.sample_rate,
        block_size=config.block_size,
        channels=config.channels,
        input_device=config.input_device,
        output_device=config.output_device,
    )

    print(f"Testing processor latency for {duration}s...")
    processor.start()

    time.sleep(duration)

    stats = processor.get_stats()
    processor.stop()

    return {
        'duration_s': duration,
        'total_blocks': stats['total_blocks'],
        'underruns': stats['underruns'],
        'overruns': stats['overruns'],
        'avg_callback_ms': stats['avg_callback_ms'],
        'max_callback_ms': stats['max_callback_ms'],
        'budget_ms': stats['budget_ms'],
        'cpu_utilization_pct': (stats['avg_callback_ms'] / stats['budget_ms']) * 100,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Latency measurement tests for Lyrebird',
    )

    parser.add_argument('--test', choices=['loopback', 'callback', 'processor', 'all'],
                        default='callback',
                        help='Test to run')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--block-size', type=int, default=128,
                        help='Audio block size')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of channels')
    parser.add_argument('--input-device', type=int, default=None,
                        help='Input device index')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Output device index')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Test duration in seconds')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='Model path (for processor test)')
    parser.add_argument('--buffer-length', type=int, default=128,
                        help='Model buffer length')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Model hidden size')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Model layers')
    parser.add_argument('--list-devices', action='store_true',
                        help='List audio devices')

    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    config = AudioConfig(
        sample_rate=args.sample_rate,
        channels=args.channels,
        block_size=args.block_size,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    print("Lyrebird Latency Tests")
    print("=" * 60)
    print(f"Sample rate: {config.sample_rate} Hz")
    print(f"Block size: {config.block_size} samples")
    print(f"Channels: {config.channels}")
    print(f"Theoretical min latency: {(config.block_size * 2 / config.sample_rate) * 1000:.2f} ms")
    print()

    tests_to_run = ['loopback', 'callback', 'processor'] if args.test == 'all' else [args.test]

    for test in tests_to_run:
        print(f"\n{'=' * 60}")
        print(f"Test: {test}")
        print('=' * 60)

        if test == 'callback':
            results = measure_callback_latency(config, args.duration)
            print("\nResults:")
            for key, value in results.items():
                if key == 'audio_stats':
                    continue
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        elif test == 'loopback':
            latency, details = measure_loopback_latency(config, args.duration)
            print("\nResults:")
            if latency is not None:
                print(f"  Round-trip latency: {latency:.2f} ms")
                print(f"  Lag samples: {details['lag_samples']}")
                print(f"  Theoretical minimum: {details['theoretical_min_ms']:.2f} ms")
            else:
                print(f"  Error: {details.get('error', 'Unknown')}")
                if 'hint' in details:
                    print(f"  Hint: {details['hint']}")

        elif test == 'processor':
            results = run_processor_latency_test(
                args.model,
                args.buffer_length,
                args.hidden_size,
                args.num_layers,
                config,
                args.duration,
            )
            print("\nResults:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

            if results['underruns'] > 0:
                print("\nWARNING: Buffer underruns detected!")
                print("Consider increasing block size or using a smaller model.")


if __name__ == '__main__':
    main()
