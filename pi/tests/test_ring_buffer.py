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
        """Test batch tensor contains correct sliding window values."""
        buffer = BatchRingBuffer(buffer_length=4, batch_size=3, num_channels=1)

        # Fill with known sequence: 0,1,2,...,9
        for i in range(10):
            buffer.push(np.array([float(i)]))

        # 10 samples with batch_size=3: batches [0,1,2], [3,4,5], [6,7,8]; sample 9 pending
        # History(size=7) linearized: [2,3,4,5,6,7,8]
        # Sliding windows of length 4:
        #   window 0: [2,3,4,5]
        #   window 1: [3,4,5,6]
        #   window 2: [4,5,6,7]
        tensor = buffer.get_batch_tensor()

        assert tensor.shape == (3, 1, 4)
        expected = torch.tensor(
            [
                [[2.0, 3.0, 4.0, 5.0]],
                [[3.0, 4.0, 5.0, 6.0]],
                [[4.0, 5.0, 6.0, 7.0]],
            ]
        )
        torch.testing.assert_close(tensor, expected)

    def test_batch_tensor_content_stereo(self):
        """Test batch tensor with stereo channels."""
        buffer = BatchRingBuffer(buffer_length=3, batch_size=2, num_channels=2)

        for i in range(7):
            buffer.push(np.array([float(i), float(i) * 10]))

        tensor = buffer.get_batch_tensor()
        assert tensor.shape == (2, 2, 3)

        # 7 samples with batch_size=2: batches [0,1], [2,3], [4,5]; sample 6 pending
        # History(size=5) linearized: [1,2,3,4,5]
        # Sliding windows of length 3:
        #   window 0: [1,2,3], window 1: [2,3,4]
        expected_ch0 = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        expected_ch1 = torch.tensor([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]])
        torch.testing.assert_close(tensor[:, 0, :], expected_ch0)
        torch.testing.assert_close(tensor[:, 1, :], expected_ch1)

    def test_push_block(self):
        """Test push_block produces same results as per-sample push."""
        buffer_length = 4
        batch_size = 3
        num_channels = 1

        # Create two buffers - one uses push, the other uses push_block
        buf_sample = BatchRingBuffer(buffer_length, batch_size, num_channels)
        buf_block = BatchRingBuffer(buffer_length, batch_size, num_channels)

        # Pre-fill both identically
        for i in range(buffer_length + batch_size):
            sample = np.array([float(i)])
            buf_sample.push(sample)
            buf_block.push(sample)

        # Now push a block via both methods
        block = np.array([[10.0], [11.0], [12.0]], dtype=np.float32)

        for i in range(batch_size):
            buf_sample.push(block[i])
        batches = buf_block.push_block(block)

        assert batches == 1

        tensor_sample = buf_sample.get_batch_tensor()
        tensor_block = buf_block.get_batch_tensor()
        torch.testing.assert_close(tensor_sample, tensor_block)

    def test_push_block_partial_batch(self):
        """Test push_block when batch accumulator is partially filled."""
        buffer_length = 4
        batch_size = 4
        num_channels = 1

        buf_sample = BatchRingBuffer(buffer_length, batch_size, num_channels)
        buf_block = BatchRingBuffer(buffer_length, batch_size, num_channels)

        # Pre-fill both
        for i in range(buffer_length + batch_size):
            sample = np.array([float(i)])
            buf_sample.push(sample)
            buf_block.push(sample)

        # Push 1 sample to offset the batch accumulator
        buf_sample.push(np.array([100.0]))
        buf_block.push(np.array([100.0]))

        # Now push a block of 4 — it will split across batch boundaries
        block = np.array([[20.0], [21.0], [22.0], [23.0]], dtype=np.float32)

        for i in range(4):
            buf_sample.push(block[i])
        buf_block.push_block(block)

        tensor_sample = buf_sample.get_batch_tensor()
        tensor_block = buf_block.get_batch_tensor()
        torch.testing.assert_close(tensor_sample, tensor_block)

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
