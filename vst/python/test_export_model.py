#!/usr/bin/env python3
"""
Tests for the model export script.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lyrebird import FiniteImpulseResponseModel
from export_model import export_model


class TestExportModel(unittest.TestCase):
    """Tests for export_model function."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        self.json_path = os.path.join(self.temp_dir, "test_model.json")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_model(self, buffer_length=512, hidden_size=128, num_layers=3):
        """Create and save a test model."""
        model = FiniteImpulseResponseModel(
            input_size=buffer_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
        )
        torch.save(model.state_dict(), self.model_path)
        return model

    def test_export_creates_json_file(self):
        """Test that export creates a JSON file."""
        self._create_test_model()

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=512,
            hidden_size=128,
            num_layers=3,
        )

        self.assertTrue(os.path.exists(self.json_path))

    def test_export_json_structure(self):
        """Test that exported JSON has correct structure."""
        self._create_test_model()

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=512,
            hidden_size=128,
            num_layers=3,
        )

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.assertIn('config', data)
        self.assertIn('layers', data)

        config = data['config']
        self.assertEqual(config['buffer_length'], 512)
        self.assertEqual(config['hidden_size'], 128)
        self.assertEqual(config['num_layers'], 3)
        self.assertEqual(config['input_size'], 512)
        self.assertEqual(config['output_size'], 1)

    def test_export_layer_count(self):
        """Test that exported JSON has correct number of layers."""
        self._create_test_model(num_layers=3)

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=512,
            hidden_size=128,
            num_layers=3,
        )

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # 3 hidden layers + 1 output layer = 4 dense layers total
        self.assertEqual(len(data['layers']), 4)

    def test_export_layer_weights_shape(self):
        """Test that layer weights have correct shapes."""
        self._create_test_model(buffer_length=64, hidden_size=32, num_layers=2)

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=64,
            hidden_size=32,
            num_layers=2,
        )

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        layers = data['layers']

        # Layer 0: input(64) -> hidden(32)
        self.assertEqual(len(layers[0]['weights']), 32)  # 32 outputs
        self.assertEqual(len(layers[0]['weights'][0]), 64)  # 64 inputs
        self.assertEqual(len(layers[0]['bias']), 32)

        # Layer 1: hidden(32) -> hidden(32)
        self.assertEqual(len(layers[1]['weights']), 32)
        self.assertEqual(len(layers[1]['weights'][0]), 32)
        self.assertEqual(len(layers[1]['bias']), 32)

        # Layer 2: hidden(32) -> output(1)
        self.assertEqual(len(layers[2]['weights']), 1)
        self.assertEqual(len(layers[2]['weights'][0]), 32)
        self.assertEqual(len(layers[2]['bias']), 1)

    def test_export_activations(self):
        """Test that layers have correct activation functions."""
        self._create_test_model(num_layers=3)

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=512,
            hidden_size=128,
            num_layers=3,
        )

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        layers = data['layers']

        # Hidden layers should have ReLU
        for i in range(3):
            self.assertEqual(layers[i]['activation'], 'relu')

        # Output layer should have no activation
        self.assertEqual(layers[3]['activation'], 'none')

    def test_export_weights_match_pytorch(self):
        """Test that exported weights match original PyTorch model."""
        model = self._create_test_model(buffer_length=64, hidden_size=32, num_layers=2)

        export_model(
            model_path=self.model_path,
            output_path=self.json_path,
            buffer_length=64,
            hidden_size=32,
            num_layers=2,
        )

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # Get PyTorch weights for first layer
        pytorch_weights = model.network[1].weight.detach().numpy()
        pytorch_bias = model.network[1].bias.detach().numpy()

        # Compare with exported weights
        exported_weights = data['layers'][0]['weights']
        exported_bias = data['layers'][0]['bias']

        for i in range(32):
            self.assertAlmostEqual(exported_bias[i], float(pytorch_bias[i]), places=5)
            for j in range(64):
                self.assertAlmostEqual(
                    exported_weights[i][j],
                    float(pytorch_weights[i][j]),
                    places=5
                )

    def test_export_different_configurations(self):
        """Test export with various model configurations."""
        configs = [
            (128, 64, 1),   # Small model
            (256, 128, 2),  # Medium model
            (512, 256, 4),  # Large model
        ]

        for buffer_length, hidden_size, num_layers in configs:
            with self.subTest(buffer_length=buffer_length, hidden_size=hidden_size, num_layers=num_layers):
                self._create_test_model(buffer_length, hidden_size, num_layers)

                export_model(
                    model_path=self.model_path,
                    output_path=self.json_path,
                    buffer_length=buffer_length,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                )

                with open(self.json_path, 'r') as f:
                    data = json.load(f)

                self.assertEqual(data['config']['buffer_length'], buffer_length)
                self.assertEqual(data['config']['hidden_size'], hidden_size)
                self.assertEqual(data['config']['num_layers'], num_layers)
                self.assertEqual(len(data['layers']), num_layers + 1)


class TestExportModelWithRealModel(unittest.TestCase):
    """Tests using real trained models if available."""

    def test_export_big_muff_model(self):
        """Test exporting the Big Muff model if it exists."""
        model_path = Path(__file__).parent.parent.parent / "models" / "big_muff_fir_model.pth"

        if not model_path.exists():
            self.skipTest("big_muff_fir_model.pth not found")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name

        try:
            # These are the expected parameters for the big muff model
            # Adjust if the actual model has different parameters
            export_model(
                model_path=str(model_path),
                output_path=json_path,
                buffer_length=512,
                hidden_size=128,
                num_layers=3,
            )

            with open(json_path, 'r') as f:
                data = json.load(f)

            self.assertIn('config', data)
            self.assertIn('layers', data)
            self.assertEqual(len(data['layers']), 4)

        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


if __name__ == '__main__':
    unittest.main()
