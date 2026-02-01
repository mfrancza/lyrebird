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

    def __init__(self, buffer_length: int, num_channels: int = 1,
                 dtype: np.dtype = np.float32):
        self.buffer_length = buffer_length
        self.num_channels = num_channels
        self.dtype = dtype

        # Internal storage: (num_channels, buffer_length)
        self._buffer = np.zeros((num_channels, buffer_length), dtype=dtype)
        self._write_pos = 0
        self._samples_written = 0
        self._lock = threading.Lock()

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
                self._buffer[:] = samples[:, -self.buffer_length:]
                self._write_pos = 0
                self._samples_written += num_samples
            else:
                # Calculate how to wrap around
                end_pos = self._write_pos + num_samples

                if end_pos <= self.buffer_length:
                    # Simple case: no wrap-around
                    self._buffer[:, self._write_pos:end_pos] = samples
                else:
                    # Wrap-around case
                    first_part = self.buffer_length - self._write_pos
                    self._buffer[:, self._write_pos:] = samples[:, :first_part]
                    self._buffer[:, :num_samples - first_part] = samples[:, first_part:]

                self._write_pos = end_pos % self.buffer_length
                self._samples_written += num_samples

    def get_buffer(self) -> np.ndarray:
        """
        Get the current buffer contents in chronological order.

        Returns:
            NumPy array of shape (num_channels, buffer_length)
        """
        with self._lock:
            if self._write_pos == 0:
                return self._buffer.copy()
            else:
                # Reorder to chronological: [write_pos:] then [:write_pos]
                return np.concatenate([
                    self._buffer[:, self._write_pos:],
                    self._buffer[:, :self._write_pos]
                ], axis=1)

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
            device = torch.device('cpu')

        # Allocate tensor buffer if needed or device changed
        if self._tensor_buffer is None or self._tensor_device != device:
            self._tensor_buffer = torch.zeros(
                1, self.num_channels, self.buffer_length,
                dtype=torch.float32, device=device
            )
            self._tensor_device = device

        # Get buffer in chronological order and copy to tensor
        # Note: get_buffer() returns a copy, ensuring thread safety
        with self._lock:
            buffer_data = self._get_buffer_unlocked()
            self._tensor_buffer[0].copy_(torch.from_numpy(buffer_data))

        return self._tensor_buffer

    def _get_buffer_unlocked(self) -> np.ndarray:
        """Get buffer contents without acquiring lock (caller must hold lock)."""
        if self._write_pos == 0:
            return self._buffer.copy()
        else:
            return np.concatenate([
                self._buffer[:, self._write_pos:],
                self._buffer[:, :self._write_pos]
            ], axis=1)

    def is_ready(self) -> bool:
        """Check if buffer has been filled at least once."""
        return self._samples_written >= self.buffer_length

    def reset(self) -> None:
        """Clear the buffer and reset state."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._samples_written = 0

    @property
    def samples_written(self) -> int:
        """Total number of samples written to buffer."""
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

        # Batch accumulator
        self._batch_buffer = np.zeros((num_channels, batch_size), dtype=np.float32)
        self._batch_pos = 0
        self._lock = threading.Lock()

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
            device = torch.device('cpu')

        # Allocate tensor if needed
        if self._batch_tensor is None or self._tensor_device != device:
            self._batch_tensor = torch.zeros(
                self.batch_size, self.num_channels, self.buffer_length,
                dtype=torch.float32, device=device
            )
            self._tensor_device = device

        # Get full history and create sliding windows under lock for thread safety
        with self._lock:
            history = self._history.get_buffer()
            for i in range(self.batch_size):
                start = i
                end = start + self.buffer_length
                self._batch_tensor[i].copy_(
                    torch.from_numpy(history[:, start:end])
                )

        return self._batch_tensor

    def get_pending_count(self) -> int:
        """Get number of samples pending in partial batch."""
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
