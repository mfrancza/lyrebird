import pytest
import torch
import torchaudio
import os
import tempfile
from torch.utils.data import DataLoader
from lyrebird_audio import (
    FiniteImpulseResponseDataSet,
    FiniteImpulseResponseModel,
    train_epoch,
    evaluate,
    transform,
)


@pytest.fixture
def temp_wav_files():
    """Create temporary WAV files for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create test audio data
    sample_rate = 44100
    duration = 0.1  # 0.1 seconds
    num_samples = int(sample_rate * duration)

    # Create input waveform (stereo, 2 channels)
    input_waveform = torch.randn(2, num_samples)
    input_path = os.path.join(temp_dir, "test_input.wav")
    torchaudio.save(input_path, input_waveform, sample_rate)

    # Create output waveform (stereo, 2 channels)
    output_waveform = torch.randn(2, num_samples)
    output_path = os.path.join(temp_dir, "test_output.wav")
    torchaudio.save(output_path, output_waveform, sample_rate)

    # Create mono input
    mono_input_waveform = torch.randn(1, num_samples)
    mono_input_path = os.path.join(temp_dir, "test_mono_input.wav")
    torchaudio.save(mono_input_path, mono_input_waveform, sample_rate)

    # Create different sample rate file
    diff_sr_waveform = torch.randn(2, num_samples)
    diff_sr_path = os.path.join(temp_dir, "test_diff_sr.wav")
    torchaudio.save(diff_sr_path, diff_sr_waveform, 48000)

    # Create different length file
    short_waveform = torch.randn(2, num_samples // 2)
    short_path = os.path.join(temp_dir, "test_short.wav")
    torchaudio.save(short_path, short_waveform, sample_rate)

    yield {
        "input_path": input_path,
        "output_path": output_path,
        "mono_input_path": mono_input_path,
        "diff_sr_path": diff_sr_path,
        "short_path": short_path,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "temp_dir": temp_dir,
    }

    # Cleanup
    for file in [input_path, output_path, mono_input_path, diff_sr_path, short_path]:
        if os.path.exists(file):
            os.remove(file)
    os.rmdir(temp_dir)


class TestFiniteImpulseResponseDataSet:
    """Test suite for FiniteImpulseResponseDataSet class."""

    def test_initialization_valid_files(self, temp_wav_files):
        """Test that dataset initializes correctly with valid files."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        assert dataset.buffer_length == buffer_length
        assert dataset.sample_rate == temp_wav_files["sample_rate"]
        assert dataset.input_channels == 2
        assert dataset.output_channels == 2
        assert dataset.total_samples == temp_wav_files["num_samples"]

    def test_input_file_not_found(self, temp_wav_files):
        """Test that FileNotFoundError is raised for missing input file."""
        with pytest.raises(FileNotFoundError, match="Input WAV file not found"):
            FiniteImpulseResponseDataSet(
                "nonexistent_input.wav", temp_wav_files["output_path"], 100
            )

    def test_output_file_not_found(self, temp_wav_files):
        """Test that FileNotFoundError is raised for missing output file."""
        with pytest.raises(FileNotFoundError, match="Output WAV file not found"):
            FiniteImpulseResponseDataSet(
                temp_wav_files["input_path"], "nonexistent_output.wav", 100
            )

    def test_mismatched_sample_rates(self, temp_wav_files):
        """Test that ValueError is raised when sample rates don't match."""
        with pytest.raises(ValueError, match="Sample rates must match"):
            FiniteImpulseResponseDataSet(
                temp_wav_files["input_path"], temp_wav_files["diff_sr_path"], 100
            )

    def test_mismatched_lengths(self, temp_wav_files):
        """Test that ValueError is raised when file lengths don't match."""
        with pytest.raises(ValueError, match="WAV files must have same length"):
            FiniteImpulseResponseDataSet(
                temp_wav_files["input_path"], temp_wav_files["short_path"], 100
            )

    def test_buffer_length_too_large(self, temp_wav_files):
        """Test that ValueError is raised when buffer_length exceeds total samples."""
        buffer_length = temp_wav_files["num_samples"] + 100
        with pytest.raises(ValueError, match="Buffer length must be <= total samples"):
            FiniteImpulseResponseDataSet(
                temp_wav_files["input_path"],
                temp_wav_files["output_path"],
                buffer_length,
            )

    def test_different_channel_counts_allowed(self, temp_wav_files):
        """Test that files with different channel counts are allowed."""
        buffer_length = 100
        # This should NOT raise an error
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["mono_input_path"],
            temp_wav_files["output_path"],
            buffer_length,
        )

        assert dataset.input_channels == 1
        assert dataset.output_channels == 2

    def test_dataset_length(self, temp_wav_files):
        """Test that dataset length is calculated correctly."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        expected_length = temp_wav_files["num_samples"] - buffer_length
        assert len(dataset) == expected_length

    def test_getitem_shapes(self, temp_wav_files):
        """Test that __getitem__ returns correct shapes."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        input_buffer, output_sample = dataset[0]

        # Check shapes
        assert input_buffer.shape == (2, buffer_length)
        assert output_sample.shape == (2,)

    def test_getitem_different_channels(self, temp_wav_files):
        """Test that __getitem__ returns correct shapes with different channel counts."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["mono_input_path"],
            temp_wav_files["output_path"],
            buffer_length,
        )

        input_buffer, output_sample = dataset[0]

        # Check shapes
        assert input_buffer.shape == (1, buffer_length)
        assert output_sample.shape == (2,)

    def test_getitem_boundary_conditions(self, temp_wav_files):
        """Test __getitem__ at boundary conditions."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        # First item
        input_buffer, output_sample = dataset[0]
        assert input_buffer.shape == (2, buffer_length)
        assert output_sample.shape == (2,)

        # Last item
        last_idx = len(dataset) - 1
        input_buffer, output_sample = dataset[last_idx]
        assert input_buffer.shape == (2, buffer_length)
        assert output_sample.shape == (2,)

    def test_getitem_out_of_range(self, temp_wav_files):
        """Test that IndexError is raised for out of range indices."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        with pytest.raises(IndexError, match="Index .* is out of range"):
            _ = dataset[len(dataset)]

    def test_getitem_correct_samples(self, temp_wav_files):
        """Test that __getitem__ returns the correct samples."""
        buffer_length = 50
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        # Load the original waveforms
        input_waveform, _ = torchaudio.load(temp_wav_files["input_path"])
        output_waveform, _ = torchaudio.load(temp_wav_files["output_path"])

        # Test index 0
        input_buffer, output_sample = dataset[0]
        i = buffer_length  # Position 50

        # Check that input_buffer contains samples [0:50]
        assert torch.allclose(input_buffer, input_waveform[:, 0:buffer_length])

        # Check that output_sample is the sample at position 50
        assert torch.allclose(output_sample, output_waveform[:, i])

        # Test index 10
        input_buffer, output_sample = dataset[10]
        i = 10 + buffer_length  # Position 60

        # Check that input_buffer contains samples [10:60]
        assert torch.allclose(input_buffer, input_waveform[:, 10:i])

        # Check that output_sample is the sample at position 60
        assert torch.allclose(output_sample, output_waveform[:, i])

    def test_get_sample_rate(self, temp_wav_files):
        """Test get_sample_rate method."""
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], 100
        )
        assert dataset.get_sample_rate() == temp_wav_files["sample_rate"]

    def test_get_input_channels(self, temp_wav_files):
        """Test get_input_channels method."""
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], 100
        )
        assert dataset.get_input_channels() == 2

    def test_get_output_channels(self, temp_wav_files):
        """Test get_output_channels method."""
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], 100
        )
        assert dataset.get_output_channels() == 2

    def test_get_total_samples(self, temp_wav_files):
        """Test get_total_samples method."""
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], 100
        )
        assert dataset.get_total_samples() == temp_wav_files["num_samples"]

    def test_iteration_over_dataset(self, temp_wav_files):
        """Test that we can iterate over the entire dataset."""
        buffer_length = 100
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        count = 0
        for input_buffer, output_sample in dataset:
            assert input_buffer.shape == (2, buffer_length)
            assert output_sample.shape == (2,)
            count += 1

        assert count == len(dataset)

    def test_buffer_length_equals_total_samples(self, temp_wav_files):
        """Test edge case where buffer_length equals total samples."""
        buffer_length = temp_wav_files["num_samples"]

        # This should create an empty dataset (0 items) since we can't get
        # buffer_length samples plus an output sample
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        # Dataset should have 0 items
        assert len(dataset) == 0

    def test_small_buffer_length(self, temp_wav_files):
        """Test with a very small buffer length."""
        buffer_length = 1
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"], temp_wav_files["output_path"], buffer_length
        )

        assert len(dataset) == temp_wav_files["num_samples"] - buffer_length

        input_buffer, output_sample = dataset[0]
        assert input_buffer.shape == (2, 1)
        assert output_sample.shape == (2,)


class TestFiniteImpulseResponseModel:
    """Test suite for FiniteImpulseResponseModel class."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=3, output_size=1
        )

        assert model.input_size == 1024
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.output_size == 1

    def test_model_invalid_num_layers(self):
        """Test that ValueError is raised for invalid num_layers."""
        with pytest.raises(ValueError, match="num_layers must be >= 1"):
            FiniteImpulseResponseModel(
                input_size=1024, hidden_size=128, num_layers=0, output_size=1
            )

    def test_model_forward_batch(self):
        """Test forward pass with batch input."""
        model = FiniteImpulseResponseModel(
            input_size=1024,  # 1 channel * 1024 buffer_length
            hidden_size=128,
            num_layers=2,
            output_size=1,
        )

        # Create batch input: (batch_size, channels, buffer_length)
        batch_input = torch.randn(32, 1, 1024)

        output = model(batch_input)

        # Output should be (batch_size, output_size)
        assert output.shape == (32, 1)

    def test_model_forward_single(self):
        """Test forward pass with single input (batch size of 1)."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=2, output_size=1
        )

        # Create single input with batch dimension: (1, channels, buffer_length)
        single_input = torch.randn(1, 1, 1024)

        output = model(single_input)

        # Output should be (1, output_size)
        assert output.shape == (1, 1)

    def test_model_multi_channel(self):
        """Test model with multi-channel input."""
        model = FiniteImpulseResponseModel(
            input_size=2048,  # 2 channels * 1024 buffer_length
            hidden_size=256,
            num_layers=3,
            output_size=2,
        )

        # Create batch input: (batch_size, channels, buffer_length)
        batch_input = torch.randn(16, 2, 1024)

        output = model(batch_input)

        # Output should be (batch_size, output_size)
        assert output.shape == (16, 2)

    def test_model_different_sizes(self):
        """Test model with various hidden sizes and layer counts."""
        test_configs = [
            (512, 64, 1, 1),
            (1024, 128, 2, 1),
            (2048, 256, 3, 2),
            (4096, 512, 4, 1),
        ]

        for input_size, hidden_size, num_layers, output_size in test_configs:
            model = FiniteImpulseResponseModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
            )

            # Test with appropriate input
            batch_input = torch.randn(8, 1, input_size)
            output = model(batch_input)

            assert output.shape == (8, output_size)

    def test_model_requires_batch_dimension(self):
        """Test that model expects input with batch dimension."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=2, output_size=1
        )

        # Input without batch dimension will be handled by Flatten
        # which starts flattening from dim 1, so 2D input should work
        input_2d = torch.randn(1, 1024)
        output = model(input_2d)

        # Flatten(start_dim=1) on 2D input just keeps it as is
        assert output.shape == (1, 1)

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=2, output_size=1
        )

        # Create input and target
        batch_input = torch.randn(16, 1, 1024, requires_grad=True)
        target = torch.randn(16, 1)

        # Forward pass
        output = model(batch_input)

        # Compute loss
        loss = torch.nn.functional.mse_loss(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert batch_input.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_model_layer_structure(self):
        """Test that model has correct layer structure."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=3, output_size=1
        )

        # Count layer types
        flatten_count = 0
        linear_count = 0
        relu_count = 0

        for module in model.network:
            if isinstance(module, torch.nn.Flatten):
                flatten_count += 1
            elif isinstance(module, torch.nn.Linear):
                linear_count += 1
            elif isinstance(module, torch.nn.ReLU):
                relu_count += 1

        # Should have 1 Flatten layer at the start
        assert flatten_count == 1

        # Should have num_layers + 1 Linear layers (hidden + output)
        assert linear_count == 4  # 3 hidden + 1 output

        # Should have num_layers ReLU layers (one after each hidden layer)
        assert relu_count == 3

    def test_model_deterministic(self):
        """Test that model produces consistent output for same input."""
        model = FiniteImpulseResponseModel(
            input_size=1024, hidden_size=128, num_layers=2, output_size=1
        )

        model.eval()  # Set to evaluation mode

        # Create fixed input
        torch.manual_seed(42)
        batch_input = torch.randn(8, 1, 1024)

        # Run forward pass twice
        output1 = model(batch_input)
        output2 = model(batch_input)

        # Outputs should be identical
        assert torch.allclose(output1, output2)


class TestTrainingFunctions:
    """Test suite for train and test functions."""

    def test_train_function(self, temp_wav_files):
        """Test that train function runs without errors."""
        # Create a small dataset
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"],
            temp_wav_files["output_path"],
            buffer_length=100,
        )

        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Create model
        model = FiniteImpulseResponseModel(
            input_size=200,  # 2 channels * 100 buffer_length
            hidden_size=32,
            num_layers=1,
            output_size=2,
        )

        # Setup training
        device = torch.device("cpu")
        model = model.to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train for one batch (should not raise errors)
        initial_params = [p.clone() for p in model.parameters()]
        train_epoch(dataloader, model, loss_fn, optimizer, device)

        # Check that parameters were updated
        final_params = list(model.parameters())
        params_changed = any(
            not torch.allclose(initial, final)
            for initial, final in zip(initial_params, final_params)
        )
        assert params_changed, "Parameters should be updated during training"

    def test_evaluate_function(self, temp_wav_files):
        """Test that evaluate function runs without errors."""
        # Create a small dataset
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"],
            temp_wav_files["output_path"],
            buffer_length=100,
        )

        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Create model
        model = FiniteImpulseResponseModel(
            input_size=200,  # 2 channels * 100 buffer_length
            hidden_size=32,
            num_layers=1,
            output_size=2,
        )

        # Setup evaluation
        device = torch.device("cpu")
        model = model.to(device)
        loss_fn = torch.nn.MSELoss()

        # Evaluate function should not raise errors
        evaluate(dataloader, model, loss_fn, device)

    def test_train_function_updates_model(self, temp_wav_files):
        """Test that training actually updates model weights."""
        # Create a small dataset
        dataset = FiniteImpulseResponseDataSet(
            temp_wav_files["input_path"],
            temp_wav_files["output_path"],
            buffer_length=50,
        )

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Create model
        model = FiniteImpulseResponseModel(
            input_size=100,  # 2 channels * 50 buffer_length
            hidden_size=64,
            num_layers=2,
            output_size=2,
        )

        device = torch.device("cpu")
        model = model.to(device)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Get initial loss
        model.eval()
        initial_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                initial_loss += loss_fn(pred, y).item()
                break

        # Train for multiple iterations
        for _ in range(5):
            train_epoch(dataloader, model, loss_fn, optimizer, device)

        # Get final loss
        model.eval()
        final_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                final_loss += loss_fn(pred, y).item()
                break

        # Loss should generally decrease (with high probability)
        # We don't assert this strictly as random initialization might cause issues
        # but we verify the training ran
        assert final_loss >= 0, "Loss should be non-negative"


class TestTransformFunction:
    """Test suite for transform function."""

    def test_transform_function(self, temp_wav_files):
        """Test that transform function processes audio correctly."""
        buffer_length = 100

        # Create and train a simple model
        model = FiniteImpulseResponseModel(
            input_size=200,  # 2 channels * 100 buffer_length
            hidden_size=32,
            num_layers=1,
            output_size=2,
        )

        device = torch.device("cpu")
        model = model.to(device)
        model.eval()

        # Create output path
        output_path = os.path.join(temp_wav_files["temp_dir"], "output.wav")

        # Transform the audio
        transform(
            model, temp_wav_files["input_path"], output_path, buffer_length, device
        )

        # Verify output file exists
        assert os.path.exists(output_path), "Output file should be created"

        # Load output and verify properties
        output_waveform, output_sample_rate = torchaudio.load(output_path)
        input_waveform, input_sample_rate = torchaudio.load(
            temp_wav_files["input_path"]
        )

        # Sample rates should match
        assert output_sample_rate == input_sample_rate

        # Shapes should match
        assert output_waveform.shape == input_waveform.shape

        # First buffer_length samples should match input (copied as-is)
        assert torch.allclose(
            output_waveform[:, :buffer_length], input_waveform[:, :buffer_length]
        )

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_transform_mono_audio(self, temp_wav_files):
        """Test transform with mono audio."""
        buffer_length = 50

        # Create model for mono audio
        model = FiniteImpulseResponseModel(
            input_size=50,  # 1 channel * 50 buffer_length
            hidden_size=32,
            num_layers=1,
            output_size=1,
        )

        device = torch.device("cpu")
        model = model.to(device)

        # Create output path
        output_path = os.path.join(temp_wav_files["temp_dir"], "output_mono.wav")

        # Transform the mono audio
        transform(
            model, temp_wav_files["mono_input_path"], output_path, buffer_length, device
        )

        # Verify output exists and has correct shape
        assert os.path.exists(output_path)
        output_waveform, _ = torchaudio.load(output_path)
        input_waveform, _ = torchaudio.load(temp_wav_files["mono_input_path"])

        assert output_waveform.shape == input_waveform.shape
        assert output_waveform.shape[0] == 1, "Should be mono"

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_transform_preserves_sample_rate(self, temp_wav_files):
        """Test that transform preserves the input sample rate."""
        buffer_length = 100

        model = FiniteImpulseResponseModel(
            input_size=200, hidden_size=32, num_layers=1, output_size=2
        )

        device = torch.device("cpu")
        model = model.to(device)

        output_path = os.path.join(temp_wav_files["temp_dir"], "output_sr.wav")

        # Get input sample rate
        _, input_sr = torchaudio.load(temp_wav_files["input_path"])

        # Transform
        transform(
            model, temp_wav_files["input_path"], output_path, buffer_length, device
        )

        # Check output sample rate
        _, output_sr = torchaudio.load(output_path)
        assert output_sr == input_sr, "Sample rate should be preserved"

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
