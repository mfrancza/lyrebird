"""
Ring buffer implementation for real-time audio processing.

Provides a thread-safe circular buffer that maintains sample history
for FIR model inference.
"""

import numpy as np
import torch
import threading
from typing import Optional


class RingBuffer:
    """
    Thread-safe circular buffer for audio samples.

    Maintains a fixed-size history of samples that can be efficiently
    converted to a PyTorch tensor for model inference.

    Args:
        buffer_length: Number of samples to maintain in history
        num_channels: Number of audio channels (1=mono, 2=stereo)
        dtype: NumPy dtype for internal storage (default: float32)
    """

    def __init__(
        self, buffer_length: int, num_channels: int = 1, dtype: np.dtype = np.float32
    ):
        self.buffer_length = buffer_length
        self.num_channels = num_channels
        self.dtype = dtype

        # Internal storage: (num_channels, buffer_length)
        self._buffer = np.zeros((num_channels, buffer_length), dtype=dtype)
        self._write_pos = 0
        self._samples_written = 0
        self._lock = threading.Lock()

        # Pre-allocated linear buffer for zero-allocation linearization,
        # plus a tensor view of it so get_tensor() doesn't allocate a
        # torch.from_numpy wrapper per sample
        self._linear_buffer = np.zeros((num_channels, buffer_length), dtype=dtype)
        self._linear_tensor = torch.from_numpy(self._linear_buffer)

        # Pre-allocated tensor for inference (avoids allocation in hot path)
        self._tensor_buffer: Optional[torch.Tensor] = None
        self._tensor_device: Optional[torch.device] = None

    def push(self, samples: np.ndarray) -> None:
        """
        Push new samples into the buffer.

        Args:
            samples: Audio samples, shape (num_channels,) for single sample
                    or (num_channels, num_samples) for multiple samples
        """
        with self._lock:
            # Handle single sample vs batch
            if samples.ndim == 1:
                samples = samples.reshape(self.num_channels, 1)

            num_samples = samples.shape[1]

            if num_samples >= self.buffer_length:
                # If input is larger than buffer, just keep the last buffer_length samples
                self._buffer[:] = samples[:, -self.buffer_length :]
                self._write_pos = 0
                self._samples_written += num_samples
            else:
                # Calculate how to wrap around
                end_pos = self._write_pos + num_samples

                if end_pos <= self.buffer_length:
                    # Simple case: no wrap-around
                    self._buffer[:, self._write_pos : end_pos] = samples
                else:
                    # Wrap-around case
                    first_part = self.buffer_length - self._write_pos
                    self._buffer[:, self._write_pos :] = samples[:, :first_part]
                    self._buffer[:, : num_samples - first_part] = samples[
                        :, first_part:
                    ]

                self._write_pos = end_pos % self.buffer_length
                self._samples_written += num_samples

    def get_buffer(self) -> np.ndarray:
        """
        Get the current buffer contents in chronological order.

        Returns:
            NumPy array of shape (num_channels, buffer_length) — always a copy
        """
        with self._lock:
            return self._linearize_buffer().copy()

    def get_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get the current buffer as a PyTorch tensor ready for model inference.

        Uses pre-allocated tensor to avoid allocations in real-time path.

        Args:
            device: Target device for tensor. If None, uses CPU.

        Returns:
            Tensor of shape (1, num_channels, buffer_length) - batched for model input
        """
        if device is None:
            device = torch.device("cpu")

        # Allocate tensor buffer if needed or device changed
        if self._tensor_buffer is None or self._tensor_device != device:
            self._tensor_buffer = torch.zeros(
                1,
                self.num_channels,
                self.buffer_length,
                dtype=torch.float32,
                device=device,
            )
            self._tensor_device = device

        # Linearize buffer in-place and copy to tensor via the cached view
        with self._lock:
            self._linearize_buffer()
            self._tensor_buffer[0].copy_(self._linear_tensor)

        return self._tensor_buffer

    def _linearize_buffer(self) -> np.ndarray:
        """
        Linearize buffer into chronological order using pre-allocated storage.

        Caller must hold self._lock. Returns self._linear_buffer (not a copy).
        """
        if self._write_pos == 0:
            self._linear_buffer[:] = self._buffer
        else:
            tail_len = self.buffer_length - self._write_pos
            self._linear_buffer[:, :tail_len] = self._buffer[:, self._write_pos :]
            self._linear_buffer[:, tail_len:] = self._buffer[:, : self._write_pos]
        return self._linear_buffer

    def linearize_into(self, out: np.ndarray) -> None:
        """
        Copy buffer contents in chronological order into a provided array.

        Thread-safe: acquires the lock internally.

        Args:
            out: Pre-allocated array of shape (num_channels, buffer_length)
                 to receive the linearized data.
        """
        with self._lock:
            # Write wrapped segments directly into out — avoids the extra
            # full-history pass through _linear_buffer on the hot path
            if self._write_pos == 0:
                np.copyto(out, self._buffer)
            else:
                tail_len = self.buffer_length - self._write_pos
                out[:, :tail_len] = self._buffer[:, self._write_pos :]
                out[:, tail_len:] = self._buffer[:, : self._write_pos]

    def is_ready(self) -> bool:
        """Check if buffer has been filled at least once."""
        with self._lock:
            return self._samples_written >= self.buffer_length

    def reset(self) -> None:
        """Clear the buffer and reset state."""
        with self._lock:
            self._buffer.fill(0)
            self._linear_buffer.fill(0)
            self._write_pos = 0
            self._samples_written = 0

    @property
    def samples_written(self) -> int:
        """Total number of samples written to buffer."""
        with self._lock:
            return self._samples_written


class BatchRingBuffer:
    """
    Ring buffer that accumulates samples for batch processing.

    Instead of processing one sample at a time, this buffer accumulates
    samples and provides batched tensor output for more efficient inference.

    Args:
        buffer_length: Number of samples to maintain in history per position
        batch_size: Number of samples to accumulate before processing
        num_channels: Number of audio channels
    """

    def __init__(self, buffer_length: int, batch_size: int, num_channels: int = 1):
        self.buffer_length = buffer_length
        self.batch_size = batch_size
        self.num_channels = num_channels

        # Main history buffer
        self._history = RingBuffer(buffer_length + batch_size, num_channels)

        # Pre-allocated buffer for linearized history output
        self._history_linear = np.zeros(
            (num_channels, buffer_length + batch_size), dtype=np.float32
        )

        # Batch accumulator
        self._batch_buffer = np.zeros((num_channels, batch_size), dtype=np.float32)
        self._batch_pos = 0
        self._lock = threading.Lock()

        # Pre-allocated staging buffer for sliding window output
        # Shape: (batch_size, num_channels, buffer_length) — contiguous for torch
        self._window_staging = np.zeros(
            (batch_size, num_channels, buffer_length), dtype=np.float32
        )

        # Precomputed stable views so get_batch_tensor() allocates nothing:
        # sliding windows over _history_linear (which is updated in place
        # each block) transposed to (batch, channels, buffer_length), and a
        # tensor view sharing _window_staging's memory
        self._windows_view = np.lib.stride_tricks.sliding_window_view(
            self._history_linear, buffer_length, axis=1
        )[:, :batch_size, :].transpose(1, 0, 2)
        self._staging_tensor = torch.from_numpy(self._window_staging)

        # Pre-allocated batch tensor
        self._batch_tensor: Optional[torch.Tensor] = None
        self._tensor_device: Optional[torch.device] = None

    def push(self, samples: np.ndarray) -> int:
        """
        Push samples into the buffer.

        Args:
            samples: Audio samples, shape (num_channels,) or (num_channels, num_samples)

        Returns:
            Number of complete batches ready for processing
        """
        with self._lock:
            if samples.ndim == 1:
                samples = samples.reshape(self.num_channels, 1)

            num_samples = samples.shape[1]
            batches_ready = 0

            # Process samples
            for i in range(num_samples):
                self._batch_buffer[:, self._batch_pos] = samples[:, i]
                self._batch_pos += 1

                if self._batch_pos >= self.batch_size:
                    # Push batch to history and reset
                    self._history.push(self._batch_buffer)
                    self._batch_pos = 0
                    batches_ready += 1

            return batches_ready

    def push_block(self, block: np.ndarray) -> int:
        """
        Push a full block of audio samples.

        Optimized for the common case where block_size == batch_size and the
        batch accumulator is empty — bypasses the per-sample loop entirely.

        Args:
            block: Audio samples, shape (block_size, channels)

        Returns:
            Number of complete batches ready for processing
        """
        with self._lock:
            # Transpose to (channels, block_size) for internal storage
            samples = block.T

            num_samples = samples.shape[1]

            # Fast path: block fills exactly one batch and accumulator is empty
            if num_samples == self.batch_size and self._batch_pos == 0:
                self._batch_buffer[:] = samples
                self._history.push(self._batch_buffer)
                return 1

            # Slow path: process sample by sample (handles partial batches)
            batches_ready = 0
            for i in range(num_samples):
                self._batch_buffer[:, self._batch_pos] = samples[:, i]
                self._batch_pos += 1

                if self._batch_pos >= self.batch_size:
                    self._history.push(self._batch_buffer)
                    self._batch_pos = 0
                    batches_ready += 1

            return batches_ready

    def get_batch_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get batched input tensors for model inference.

        Returns a tensor of shape (batch_size, num_channels, buffer_length)
        containing overlapping windows of history for each sample in the batch.

        Args:
            device: Target device for tensor.

        Returns:
            Tensor ready for batched model inference
        """
        if device is None:
            device = torch.device("cpu")

        # Get full history and create sliding windows under lock for thread safety
        # Tensor allocation is also inside lock to prevent race conditions
        with self._lock:
            # Allocate tensor if needed
            if self._batch_tensor is None or self._tensor_device != device:
                self._batch_tensor = torch.zeros(
                    self.batch_size,
                    self.num_channels,
                    self.buffer_length,
                    dtype=torch.float32,
                    device=device,
                )
                self._tensor_device = device

            # Linearize history via public API (acquires history lock internally)
            self._history.linearize_into(self._history_linear)
            # Copy through the precomputed views — no per-call view/tensor
            # wrapper allocation
            np.copyto(self._window_staging, self._windows_view)
            self._batch_tensor.copy_(self._staging_tensor)

        return self._batch_tensor

    def get_pending_count(self) -> int:
        """Get number of samples pending in partial batch."""
        with self._lock:
            return self._batch_pos

    def is_ready(self) -> bool:
        """Check if buffer has enough history for processing."""
        return self._history.is_ready()

    def reset(self) -> None:
        """Reset buffer state."""
        with self._lock:
            self._history.reset()
            self._batch_buffer.fill(0)
            self._batch_pos = 0
