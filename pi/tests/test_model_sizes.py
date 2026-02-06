"""Unit tests for model_sizes.py module."""

import argparse
import sys
from pathlib import Path

import pytest

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pi.model_sizes import (
    MODEL_SIZE_PRESETS,
    get_preset,
    add_size_arguments,
    resolve_size_arguments,
    print_size_info,
)


class TestModelSizePresets:
    """Tests for the MODEL_SIZE_PRESETS constant."""

    def test_presets_have_required_keys(self):
        """All presets should have buffer_length, hidden_size, num_layers."""
        required_keys = ["buffer_length", "hidden_size", "num_layers", "description"]
        for name, preset in MODEL_SIZE_PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset '{name}' missing key '{key}'"

    def test_preset_values_are_positive(self):
        """All numeric preset values should be positive integers."""
        for name, preset in MODEL_SIZE_PRESETS.items():
            assert preset["buffer_length"] > 0
            assert preset["hidden_size"] > 0
            assert preset["num_layers"] > 0

    def test_small_preset_values(self):
        """Small preset should have expected values."""
        small = MODEL_SIZE_PRESETS["small"]
        assert small["buffer_length"] == 128
        assert small["hidden_size"] == 32
        assert small["num_layers"] == 2

    def test_medium_preset_values(self):
        """Medium preset should have expected values."""
        medium = MODEL_SIZE_PRESETS["medium"]
        assert medium["buffer_length"] == 256
        assert medium["hidden_size"] == 64
        assert medium["num_layers"] == 2

    def test_large_preset_values(self):
        """Large preset should have expected values."""
        large = MODEL_SIZE_PRESETS["large"]
        assert large["buffer_length"] == 512
        assert large["hidden_size"] == 128
        assert large["num_layers"] == 3


class TestGetPreset:
    """Tests for the get_preset() function."""

    def test_get_preset_small(self):
        """get_preset('small') should return small preset."""
        preset = get_preset("small")
        assert preset["buffer_length"] == 128
        assert preset["hidden_size"] == 32
        assert preset["num_layers"] == 2

    def test_get_preset_medium(self):
        """get_preset('medium') should return medium preset."""
        preset = get_preset("medium")
        assert preset["buffer_length"] == 256
        assert preset["hidden_size"] == 64
        assert preset["num_layers"] == 2

    def test_get_preset_large(self):
        """get_preset('large') should return large preset."""
        preset = get_preset("large")
        assert preset["buffer_length"] == 512
        assert preset["hidden_size"] == 128
        assert preset["num_layers"] == 3

    def test_get_preset_case_insensitive(self):
        """get_preset() should be case insensitive."""
        assert get_preset("SMALL")["buffer_length"] == 128
        assert get_preset("Small")["buffer_length"] == 128
        assert get_preset("MEDIUM")["hidden_size"] == 64
        assert get_preset("Large")["num_layers"] == 3

    def test_get_preset_invalid_raises(self):
        """get_preset() should raise ValueError for invalid size."""
        with pytest.raises(ValueError) as exc_info:
            get_preset("invalid")
        assert "Invalid size 'invalid'" in str(exc_info.value)
        assert "small" in str(exc_info.value)
        assert "medium" in str(exc_info.value)
        assert "large" in str(exc_info.value)

    def test_get_preset_empty_raises(self):
        """get_preset() should raise ValueError for empty string."""
        with pytest.raises(ValueError):
            get_preset("")


class TestAddSizeArguments:
    """Tests for the add_size_arguments() function."""

    def test_adds_size_argument(self):
        """add_size_arguments() should add --size argument."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser)
        args = parser.parse_args(["--size", "small"])
        assert args.size == "small"

    def test_adds_short_size_argument(self):
        """add_size_arguments() should add -s shorthand."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser)
        args = parser.parse_args(["-s", "medium"])
        assert args.size == "medium"

    def test_adds_manual_arguments(self):
        """add_size_arguments() should add manual override arguments."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser)
        args = parser.parse_args(
            ["--buffer-length", "256", "--hidden-size", "64", "--num-layers", "2"]
        )
        assert args.buffer_length == 256
        assert args.hidden_size == 64
        assert args.num_layers == 2

    def test_default_size_none(self):
        """With no default_size, --size should default to None."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser)
        args = parser.parse_args([])
        assert args.size is None

    def test_default_size_specified(self):
        """With default_size, --size should use that default."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser, default_size="small")
        args = parser.parse_args([])
        assert args.size == "small"

    def test_size_choices_restricted(self):
        """--size should only accept small, medium, large."""
        parser = argparse.ArgumentParser()
        add_size_arguments(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--size", "invalid"])


class TestResolveSizeArguments:
    """Tests for the resolve_size_arguments() function."""

    def _make_args(self, **kwargs):
        """Helper to create args namespace with defaults."""
        defaults = {
            "size": None,
            "buffer_length": None,
            "hidden_size": None,
            "num_layers": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_resolve_size_preset_small(self):
        """resolve_size_arguments() with --size small returns small values."""
        args = self._make_args(size="small")
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 128
        assert hidden_size == 32
        assert num_layers == 2

    def test_resolve_size_preset_medium(self):
        """resolve_size_arguments() with --size medium returns medium values."""
        args = self._make_args(size="medium")
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 256
        assert hidden_size == 64
        assert num_layers == 2

    def test_resolve_size_preset_large(self):
        """resolve_size_arguments() with --size large returns large values."""
        args = self._make_args(size="large")
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 512
        assert hidden_size == 128
        assert num_layers == 3

    def test_resolve_full_manual_args(self):
        """resolve_size_arguments() with full manual args returns those values."""
        args = self._make_args(buffer_length=100, hidden_size=50, num_layers=4)
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 100
        assert hidden_size == 50
        assert num_layers == 4

    def test_resolve_partial_manual_with_preset(self):
        """Manual args should override preset values selectively."""
        args = self._make_args(size="medium", buffer_length=300)
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 300  # overridden
        assert hidden_size == 64  # from preset
        assert num_layers == 2  # from preset

    def test_resolve_partial_manual_hidden_with_preset(self):
        """Manual hidden_size should override preset."""
        args = self._make_args(size="small", hidden_size=48)
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 128  # from preset
        assert hidden_size == 48  # overridden
        assert num_layers == 2  # from preset

    def test_resolve_partial_manual_num_layers_with_preset(self):
        """Manual num_layers should override preset."""
        args = self._make_args(size="large", num_layers=5)
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert buffer_length == 512  # from preset
        assert hidden_size == 128  # from preset
        assert num_layers == 5  # overridden

    def test_resolve_no_args_raises(self):
        """resolve_size_arguments() should raise when no --size and no manual."""
        args = self._make_args()
        with pytest.raises(ValueError) as exc_info:
            resolve_size_arguments(args)
        assert "--size" in str(exc_info.value)

    def test_resolve_partial_manual_no_preset_missing_buffer(self):
        """Without --size, missing --buffer-length should raise."""
        args = self._make_args(hidden_size=64)
        with pytest.raises(ValueError) as exc_info:
            resolve_size_arguments(args)
        assert "--buffer-length required" in str(exc_info.value)

    def test_resolve_partial_manual_no_preset_missing_hidden(self):
        """Without --size, missing --hidden-size should raise."""
        args = self._make_args(buffer_length=128)
        with pytest.raises(ValueError) as exc_info:
            resolve_size_arguments(args)
        assert "--hidden-size required" in str(exc_info.value)

    def test_resolve_num_layers_defaults_to_one(self):
        """Without --size, missing --num-layers should default to 1."""
        args = self._make_args(buffer_length=128, hidden_size=32)
        buffer_length, hidden_size, num_layers = resolve_size_arguments(args)
        assert num_layers == 1


class TestPrintSizeInfo:
    """Tests for the print_size_info() function."""

    def test_print_with_size_preset(self, capsys):
        """print_size_info() with size should print preset info."""
        print_size_info(size="small")
        captured = capsys.readouterr()
        assert "SMALL" in captured.out
        assert "128" in captured.out
        assert "32" in captured.out
        assert "2" in captured.out
        assert "Raspberry Pi" in captured.out

    def test_print_with_manual_values(self, capsys):
        """print_size_info() with manual values should print them."""
        print_size_info(buffer_length=100, hidden_size=50, num_layers=3)
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "50" in captured.out
        assert "3" in captured.out
        assert "Model configuration" in captured.out

    def test_print_medium_preset(self, capsys):
        """print_size_info() with medium should show medium info."""
        print_size_info(size="medium")
        captured = capsys.readouterr()
        assert "MEDIUM" in captured.out
        assert "256" in captured.out
        assert "64" in captured.out

    def test_print_large_preset(self, capsys):
        """print_size_info() with large should show large info."""
        print_size_info(size="large")
        captured = capsys.readouterr()
        assert "LARGE" in captured.out
        assert "512" in captured.out
        assert "128" in captured.out
        assert "3" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
