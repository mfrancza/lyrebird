#!/usr/bin/env python3
"""
Real-time audio processor for Lyrebird neural FIR models.

This is the main entry point for running trained models on a Raspberry Pi
for real-time audio effect processing.
"""

import argparse
import signal
import sys
import time
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyrebird import FiniteImpulseResponseModel
from pi.ring_buffer import RingBuffer, BatchRingBuffer
from pi.audio_io import AudioConfig, AudioIO, print_devices


class RealtimeProcessor:
    """
    Real-time audio processor using Lyrebird neural FIR models.

    Args:
        model_path: Path to trained model (.pth file)
        buffer_length: Model's buffer length (must match training)
        hidden_size: Model's hidden size (must match training)
        num_layers: Model's number of layers (must match training)
        sample_rate: Audio sample rate
        block_size: Audio processing block size
        channels: Number of audio channels
        use_onnx: Use ONNX Runtime for inference (if available)
        device: Torch device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        model_path: str,
        buffer_length: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
        sample_rate: int = 44100,
        block_size: int = 128,
        channels: int = 1,
        use_onnx: bool = False,
        device: str = 'cpu',
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        self.model_path = model_path
        self.buffer_length = buffer_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.use_onnx = use_onnx
        self.device = torch.device(device)

        # Load model
        self._load_model()

        # Initialize ring buffer for sample history
        self.ring_buffer = RingBuffer(buffer_length, channels)

        # Pre-fill buffer with zeros
        self.ring_buffer.push(np.zeros((channels, buffer_length), dtype=np.float32))

        # Audio I/O configuration
        self.audio_config = AudioConfig(
            sample_rate=sample_rate,
            channels=channels,
            block_size=block_size,
            input_device=input_device,
            output_device=output_device,
            latency='low',
        )

        # Audio I/O handler
        self.audio_io = AudioIO(self.audio_config, self._process_block)

        # Bypass mode
        self.bypass = False

        # Pre-allocated output buffer
        self._output_buffer = np.zeros((block_size, channels), dtype=np.float32)

        # ONNX session (if used)
        self._onnx_session = None

    def _load_model(self) -> None:
        """Load the trained model."""
        input_size = self.buffer_length * self.channels
        output_size = self.channels

        self.model = FiniteImpulseResponseModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size
        )

        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from: {self.model_path}")
        else:
            print(f"Warning: Model file not found: {self.model_path}")
            print("Running with randomly initialized model")

        self.model.to(self.device)
        self.model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")

        # Load ONNX if requested
        if self.use_onnx:
            self._load_onnx()

    def _load_onnx(self) -> None:
        """Load ONNX model for optimized inference."""
        try:
            import onnxruntime as ort

            onnx_path = self.model_path.replace('.pth', '.onnx')
            if not os.path.exists(onnx_path):
                print(f"ONNX model not found: {onnx_path}")
                print("Run export_onnx.py to create it")
                return

            # Configure ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._onnx_session = ort.InferenceSession(
                onnx_path,
                sess_options,
                providers=['CPUExecutionProvider']
            )
            print(f"Loaded ONNX model: {onnx_path}")

        except ImportError:
            print("ONNX Runtime not installed, using PyTorch")
            self.use_onnx = False

    def _process_block(self, indata: np.ndarray) -> np.ndarray:
        """
        Process a block of audio samples.

        This is the callback for real-time audio processing.

        Args:
            indata: Input audio block, shape (block_size, channels)

        Returns:
            Processed audio block, shape (block_size, channels)
        """
        if self.bypass:
            return indata

        # Process sample by sample
        # Note: For better efficiency, consider batched processing
        for i in range(self.block_size):
            # Get input sample and add to ring buffer
            sample = indata[i]  # Shape: (channels,)
            self.ring_buffer.push(sample)

            # Get buffer as tensor and run inference
            input_tensor = self.ring_buffer.get_tensor(self.device)

            if self._onnx_session is not None:
                # ONNX inference
                ort_inputs = {
                    self._onnx_session.get_inputs()[0].name:
                    input_tensor.cpu().numpy()
                }
                output = self._onnx_session.run(None, ort_inputs)[0]
                self._output_buffer[i] = output[0]
            else:
                # PyTorch inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                self._output_buffer[i] = output.cpu().numpy()[0]

        return self._output_buffer

    def start(self) -> None:
        """Start real-time processing."""
        print(f"\nStarting real-time processor...")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Block size: {self.block_size} samples")
        print(f"  Buffer length: {self.buffer_length} samples")
        print(f"  Channels: {self.channels}")
        print(f"  Device: {self.device}")
        print(f"  ONNX: {self.use_onnx and self._onnx_session is not None}")

        latency_ms = (self.block_size * 2) / self.sample_rate * 1000
        print(f"  Estimated latency: ~{latency_ms:.1f} ms")
        print()

        self.audio_io.start()

    def stop(self) -> None:
        """Stop real-time processing."""
        self.audio_io.stop()
        print("\nProcessor stopped")

    def toggle_bypass(self) -> bool:
        """Toggle bypass mode."""
        self.bypass = not self.bypass
        return self.bypass

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return self.audio_io.get_stats()

    def print_stats(self) -> None:
        """Print processing statistics."""
        stats = self.get_stats()
        budget = stats['budget_ms']
        avg = stats['avg_callback_ms']
        max_t = stats['max_callback_ms']

        cpu_pct = (avg / budget) * 100 if budget > 0 else 0

        print(f"Blocks: {stats['total_blocks']:,} | "
              f"Underruns: {stats['underruns']} | "
              f"Overruns: {stats['overruns']} | "
              f"Avg: {avg:.3f}ms / {budget:.2f}ms ({cpu_pct:.1f}%) | "
              f"Max: {max_t:.3f}ms")


class BatchedRealtimeProcessor(RealtimeProcessor):
    """
    Batched real-time processor for improved efficiency.

    Processes multiple samples in a single model forward pass,
    which is more efficient on most hardware.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use batched ring buffer
        self.batch_buffer = BatchRingBuffer(
            self.buffer_length,
            self.block_size,
            self.channels
        )

        # Pre-fill with zeros
        zeros = np.zeros((self.channels, self.buffer_length + self.block_size),
                        dtype=np.float32)
        for i in range(self.buffer_length + self.block_size):
            self.batch_buffer.push(zeros[:, i])

    def _process_block(self, indata: np.ndarray) -> np.ndarray:
        """Process a block using batched inference."""
        if self.bypass:
            return indata

        # Push all input samples to batch buffer
        # Note: We process as a batch after all samples are pushed
        for i in range(self.block_size):
            self.batch_buffer.push(indata[i])

        # Get batched input tensor: (block_size, channels, buffer_length)
        input_tensor = self.batch_buffer.get_batch_tensor(self.device)

        if self._onnx_session is not None:
            # ONNX batched inference
            ort_inputs = {
                self._onnx_session.get_inputs()[0].name:
                input_tensor.cpu().numpy()
            }
            output = self._onnx_session.run(None, ort_inputs)[0]
            self._output_buffer[:] = output
        else:
            # PyTorch batched inference
            with torch.no_grad():
                output = self.model(input_tensor)
            self._output_buffer[:] = output.cpu().numpy()

        return self._output_buffer


def main():
    parser = argparse.ArgumentParser(
        description='Real-time audio processor for Lyrebird models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List audio devices
  python realtime_processor.py --list-devices

  # Run with default settings
  python realtime_processor.py --model models/small/model.pth

  # Run with specific audio devices
  python realtime_processor.py --model model.pth --input-device 2 --output-device 3

  # Run with larger buffer for more stability
  python realtime_processor.py --model model.pth --block-size 256
        """
    )

    parser.add_argument('--model', type=str, default='model.pth',
                        help='Path to model file (.pth)')
    parser.add_argument('--buffer-length', type=int, default=128,
                        help='Model buffer length (must match training)')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Model hidden size (must match training)')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Model number of layers (must match training)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--block-size', type=int, default=128,
                        help='Audio block size')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of audio channels')
    parser.add_argument('--input-device', type=int, default=None,
                        help='Input audio device index')
    parser.add_argument('--output-device', type=int, default=None,
                        help='Output audio device index')
    parser.add_argument('--use-onnx', action='store_true',
                        help='Use ONNX Runtime for inference')
    parser.add_argument('--batched', action='store_true',
                        help='Use batched processing')
    parser.add_argument('--list-devices', action='store_true',
                        help='List audio devices and exit')
    parser.add_argument('--stats-interval', type=float, default=5.0,
                        help='Statistics print interval in seconds')

    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    # Select processor class
    ProcessorClass = BatchedRealtimeProcessor if args.batched else RealtimeProcessor

    # Create processor
    processor = ProcessorClass(
        model_path=args.model,
        buffer_length=args.buffer_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        channels=args.channels,
        use_onnx=args.use_onnx,
        input_device=args.input_device,
        output_device=args.output_device,
    )

    # Signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start processing
    processor.start()

    print("Press Ctrl+C to stop")
    print("Press 'b' + Enter to toggle bypass")
    print()

    # Main loop - print stats periodically
    last_stats_time = time.time()

    try:
        while True:
            # Check for user input (non-blocking would be better)
            # For now, just print stats periodically
            time.sleep(0.1)

            current_time = time.time()
            if current_time - last_stats_time >= args.stats_interval:
                processor.print_stats()
                last_stats_time = current_time

    except KeyboardInterrupt:
        pass
    finally:
        processor.stop()


if __name__ == '__main__':
    main()
