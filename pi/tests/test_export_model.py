"""Unit tests for pi/tools/export_model.py."""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyrebird import FiniteImpulseResponseModel
from pi.tools.export_model import export_model, find_size_preset


def _make_checkpoint(tmp_path, **overrides):
    """Create a valid training checkpoint and return its path."""
    model = FiniteImpulseResponseModel(
        input_size=128, hidden_size=32, num_layers=2, output_size=1
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "input_size": 128,
        "hidden_size": 32,
        "num_layers": 2,
        "output_size": 1,
        "buffer_length": 128,
    }
    checkpoint.update(overrides)
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    return path


class TestFindSizePreset:
    """Tests for the find_size_preset helper."""

    def test_matches_small(self):
        assert find_size_preset(128, 32, 2) == "small"

    def test_matches_medium(self):
        assert find_size_preset(256, 64, 2) == "medium"

    def test_matches_large(self):
        assert find_size_preset(512, 128, 3) == "large"

    def test_no_match(self):
        assert find_size_preset(999, 999, 999) is None


class TestExportModel:
    """Tests for the export_model function."""

    def test_successful_export(self, tmp_path):
        """A valid checkpoint should produce a loadable deployment state dict."""
        checkpoint_path = _make_checkpoint(tmp_path)
        output_path = str(tmp_path / "model.deploy.pth")

        export_model(str(checkpoint_path), output_path)

        # Output file should exist and be a raw state dict
        assert Path(output_path).exists()
        state_dict = torch.load(output_path, map_location="cpu", weights_only=False)
        assert isinstance(state_dict, dict)
        # Should load into the matching model without error
        model = FiniteImpulseResponseModel(
            input_size=128, hidden_size=32, num_layers=2, output_size=1
        )
        model.load_state_dict(state_dict)

    def test_creates_output_directory(self, tmp_path):
        """Export should create parent directories for the output path."""
        checkpoint_path = _make_checkpoint(tmp_path)
        output_path = str(tmp_path / "nested" / "dir" / "model.deploy.pth")

        export_model(str(checkpoint_path), output_path)

        assert Path(output_path).exists()

    def test_rejects_non_checkpoint(self, tmp_path):
        """A plain state dict (not a training checkpoint) should be rejected."""
        model = FiniteImpulseResponseModel(
            input_size=128, hidden_size=32, num_layers=2, output_size=1
        )
        path = tmp_path / "plain.pth"
        torch.save(model.state_dict(), path)

        with pytest.raises(SystemExit):
            export_model(str(path), str(tmp_path / "out.pth"))

    def test_rejects_missing_metadata(self, tmp_path):
        """A checkpoint missing required metadata keys should be rejected."""
        model = FiniteImpulseResponseModel(
            input_size=128, hidden_size=32, num_layers=2, output_size=1
        )
        path = tmp_path / "incomplete.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "input_size": 128},
            path,
        )

        with pytest.raises(SystemExit):
            export_model(str(path), str(tmp_path / "out.pth"))

    def test_rejects_mismatched_state_dict(self, tmp_path):
        """A checkpoint whose state dict doesn't match the metadata should fail."""
        # Create checkpoint with small model weights but medium metadata
        small_model = FiniteImpulseResponseModel(
            input_size=128, hidden_size=32, num_layers=2, output_size=1
        )
        checkpoint_path = _make_checkpoint(
            tmp_path,
            model_state_dict=small_model.state_dict(),
            hidden_size=64,  # mismatch: metadata says 64 but weights are for 32
        )

        with pytest.raises(SystemExit):
            export_model(str(checkpoint_path), str(tmp_path / "out.pth"))
