#!/usr/bin/env python3
"""
Export trained Lyrebird FIR models to RTNeural JSON format.

This script converts PyTorch .pth checkpoint files to the JSON format
that RTNeural can load for real-time inference in the VST plugin.

Usage:
    python export_model.py model.pth output.json --buffer_length 512
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path to import lyrebird module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from lyrebird import FiniteImpulseResponseModel


def export_model(
    model_path: str,
    output_path: str,
    buffer_length: int,
    hidden_size: int,
    num_layers: int,
    num_channels: int = 1,
) -> None:
    """
    Export a trained PyTorch model to RTNeural JSON format.

    Args:
        model_path: Path to the .pth checkpoint file
        output_path: Path to save the JSON file
        buffer_length: Buffer length used during training
        hidden_size: Hidden layer size
        num_layers: Number of hidden layers
        num_channels: Number of audio channels (default 1 for mono)
    """
    input_size = buffer_length * num_channels
    output_size = num_channels

    # Create model with same architecture
    model = FiniteImpulseResponseModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    )

    # Load trained weights
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        # Full checkpoint with optimizer state, etc.
        state_dict = checkpoint["model_state_dict"]
    else:
        # Direct state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Extract weights in RTNeural format
    # RTNeural expects weights in a specific JSON structure
    export_data = {
        "config": {
            "buffer_length": buffer_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_channels": num_channels,
            "input_size": input_size,
            "output_size": output_size,
        },
        "layers": [],
    }

    # The model.network is a Sequential with:
    # [0] Flatten (skip - no weights)
    # [1] Linear(input_size, hidden_size)
    # [2] ReLU (skip - no weights)
    # [3] Linear(hidden_size, hidden_size) - repeated (num_layers - 1) times
    # [4] ReLU (skip - no weights)
    # ... more hidden layers ...
    # [-1] Linear(hidden_size, output_size)

    layer_idx = 0
    for name, param in model.named_parameters():
        # Parse layer info from parameter name (e.g., "network.1.weight")
        parts = name.split(".")
        if len(parts) >= 3:
            seq_idx = int(parts[1])
            param_type = parts[2]  # "weight" or "bias"

            # Convert to list for JSON serialization
            param_data = param.detach().numpy().tolist()

            # Find or create layer entry
            # Sequential indices: 1 = first linear, 3 = second linear, etc.
            # Map to our layer indices: 0, 1, 2, ...
            rtneural_layer_idx = (seq_idx - 1) // 2

            # Ensure we have enough layer entries
            while len(export_data["layers"]) <= rtneural_layer_idx:
                export_data["layers"].append({"type": "dense"})

            layer = export_data["layers"][rtneural_layer_idx]

            if param_type == "weight":
                layer["weights"] = param_data
            elif param_type == "bias":
                layer["bias"] = param_data

            # Determine if this is a hidden layer or output layer
            if rtneural_layer_idx < num_layers:
                layer["activation"] = "relu"
            else:
                layer["activation"] = "none"

    # The last layer should have no activation
    if export_data["layers"]:
        export_data["layers"][-1]["activation"] = "none"

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Model exported to: {output_path}")
    print(f"Config: {export_data['config']}")
    print(f"Layers: {len(export_data['layers'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Lyrebird models to RTNeural JSON format"
    )
    parser.add_argument("model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("output_path", type=str, help="Path for the output JSON file")
    parser.add_argument(
        "--buffer_length",
        type=int,
        required=True,
        help="Buffer length used during training",
    )
    parser.add_argument(
        "--hidden_size", type=int, required=True, help="Hidden layer size"
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of hidden layers"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=1,
        help="Number of audio channels (default: 1)",
    )

    args = parser.parse_args()

    export_model(
        model_path=args.model_path,
        output_path=args.output_path,
        buffer_length=args.buffer_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_channels=args.num_channels,
    )


if __name__ == "__main__":
    main()
