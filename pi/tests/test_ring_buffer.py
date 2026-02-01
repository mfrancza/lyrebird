"""Unit tests for ring buffer implementations."""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from pi.ring_buffer to match package structure
from pi.ring_buffer import RingBuffer, BatchRingBuffer


class TestRingBuffer:
    """Tests for RingBuffer class."""

    def test_initialization(self):
        """Test buffer initializes correctly."""
        buffer = RingBuffer(buffer_length=128, num_channels=1)
        assert buffer.buffer_length == 128
        assert buffer.num_channels == 1
        assert not buffer.is_ready()
        assert buffer.samples_written == 0

    def test_push_single_sample(self):
        """Test pushing single samples."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        for i in range(4):
            sample = np.array([float(i)])
            buffer.push(sample)

        assert buffer.is_ready()
        assert buffer.samples_written == 4

        result = buffer.get_buffer()
        expected = np.array([[0.0, 1.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_push_multiple_samples(self):
        """Test pushing multiple samples at once."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        samples = np.array([[0.0, 1.0, 2.0, 3.0]])
        buffer.push(samples)

        assert buffer.is_ready()
        result = buffer.get_buffer()
        np.testing.assert_array_almost_equal(result, samples)

    def test_wrap_around(self):
        """Test buffer wraps around correctly."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        # Fill buffer
        for i in range(6):
            sample = np.array([float(i)])
            buffer.push(sample)

        assert buffer.samples_written == 6

        # Should contain [2, 3, 4, 5] in chronological order
        result = buffer.get_buffer()
        expected = np.array([[2.0, 3.0, 4.0, 5.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_stereo_channels(self):
        """Test stereo (2 channel) operation."""
        buffer = RingBuffer(buffer_length=3, num_channels=2)

        for i in range(3):
            sample = np.array([float(i), float(i) * 10])
            buffer.push(sample)

        result = buffer.get_buffer()
        expected = np.array([[0.0, 1.0, 2.0], [0.0, 10.0, 20.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_get_tensor(self):
        """Test tensor conversion."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        for i in range(4):
            buffer.push(np.array([float(i)]))

        tensor = buffer.get_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 1, 4)
        assert tensor.dtype == torch.float32

        expected = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
        torch.testing.assert_close(tensor, expected)

    def test_tensor_device(self):
        """Test tensor is created on correct device."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        for i in range(4):
            buffer.push(np.array([float(i)]))

        device = torch.device("cpu")
        tensor = buffer.get_tensor(device)

        assert tensor.device == device

    def test_reset(self):
        """Test buffer reset."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        for i in range(4):
            buffer.push(np.array([float(i)]))

        buffer.reset()

        assert not buffer.is_ready()
        assert buffer.samples_written == 0

        result = buffer.get_buffer()
        expected = np.zeros((1, 4))
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_push_overwrites(self):
        """Test pushing more samples than buffer size."""
        buffer = RingBuffer(buffer_length=4, num_channels=1)

        # Push 10 samples at once
        samples = np.array([[float(i) for i in range(10)]])
        buffer.push(samples)

        # Should only contain last 4 samples
        result = buffer.get_buffer()
        expected = np.array([[6.0, 7.0, 8.0, 9.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestBatchRingBuffer:
    """Tests for BatchRingBuffer class."""

    def test_initialization(self):
        """Test batch buffer initializes correctly."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=2, num_channels=1)
        assert buffer.buffer_length == 4
        assert buffer.batch_size == 2
        assert buffer.num_channels == 1
        assert buffer.get_pending_count() == 0

    def test_batch_accumulation(self):
        """Test samples accumulate into batches."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=2, num_channels=1)

        # Fill history first
        for i in range(6):
            buffer.push(np.array([float(i)]))

        # Push first sample of batch
        batches = buffer.push(np.array([6.0]))
        assert batches == 0
        assert buffer.get_pending_count() == 1

        # Push second sample - completes batch
        batches = buffer.push(np.array([7.0]))
        assert batches == 1
        assert buffer.get_pending_count() == 0

    def test_batch_tensor_shape(self):
        """Test batch tensor has correct shape."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=3, num_channels=1)

        # Fill history
        for i in range(10):
            buffer.push(np.array([float(i)]))

        tensor = buffer.get_batch_tensor()

        assert tensor.shape == (3, 1, 4)

    def test_batch_tensor_content(self):
        """Test batch tensor contains sliding windows."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=2, num_channels=1)

        # Fill with sequence
        for i in range(8):
            buffer.push(np.array([float(i)]))

        tensor = buffer.get_batch_tensor()

        # Each row should be a sliding window
        # Window 0: samples around position 0
        # Window 1: samples around position 1
        assert tensor.shape == (2, 1, 4)

    def test_reset(self):
        """Test batch buffer reset."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=2, num_channels=1)

        for i in range(6):
            buffer.push(np.array([float(i)]))

        buffer.reset()

        assert buffer.get_pending_count() == 0
        assert not buffer.is_ready()


class TestRingBufferThreadSafety:
    """Thread safety tests for RingBuffer."""

    def test_concurrent_push_and_read(self):
        """Test concurrent push and read operations."""
        import threading
        import time

        buffer = RingBuffer(buffer_length=100, num_channels=1)
        errors = []

        def writer():
            try:
                for i in range(1000):
                    buffer.push(np.array([float(i)]))
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(1000):
                    _ = buffer.get_buffer()
                    _ = buffer.get_tensor()
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
