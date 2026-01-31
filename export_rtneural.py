#!/usr/bin/env python3
"""
Export trained Lyrebird models to RTNeural-compatible JSON format.

Usage:
    python export_rtneural.py model.pth output.json

    # Or in Python:
    from export_rtneural import export_to_rtneural
    export_to_rtneural("model.pth", "output.json")
"""

import torch
import json
import argparse
from pathlib import Path


def export_to_rtneural(model_path: str, output_path: str) -> dict:
    """
    Export a trained Lyrebird model to RTNeural-compatible JSON format.

    Args:
        model_path: Path to the .pth model file
        output_path: Path to save the RTNeural JSON file

    Returns:
        The exported model dictionary
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Extract configuration
    buffer_length = checkpoint["buffer_length"]
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    output_size = checkpoint["output_size"]

    print("Model configuration:")
    print(f"  buffer_length: {buffer_length}")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  output_size: {output_size}")

    # Get the state dict
    state_dict = checkpoint["model_state_dict"]

    # Build the RTNeural JSON structure
    rtneural_model = {
        "config": {
            "buffer_length": buffer_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
        "layers": [],
    }

    # The PyTorch model has layers indexed as:
    # network.1 = Linear(input -> hidden)   [layer 0]
    # network.3 = Linear(hidden -> hidden)  [layer 1]
    # network.5 = Linear(hidden -> hidden)  [layer 2]
    # network.7 = Linear(hidden -> output)  [layer 3]
    # (Even indices are Flatten/ReLU)

    # Layer indices in the Sequential: 1, 3, 5, 7, ... (odd numbers for Linear layers)
    # First layer is at index 1 (after Flatten at 0)
    # Then alternating Linear/ReLU, so Linear at 1, 3, 5, ...

    layer_indices = [1]  # First linear layer
    for i in range(num_layers - 1):
        layer_indices.append(3 + i * 2)  # Hidden layers
    layer_indices.append(1 + num_layers * 2)  # Output layer

    # Actually, let's just enumerate the state dict keys to find the linear layers
    linear_layers = []
    for key in state_dict.keys():
        if "weight" in key:
            layer_num = key.split(".")[1]  # e.g., "network.1.weight" -> "1"
            linear_layers.append(int(layer_num))

    linear_layers = sorted(set(linear_layers))
    print(f"\nFound {len(linear_layers)} linear layers at indices: {linear_layers}")

    for i, layer_idx in enumerate(linear_layers):
        weight_key = f"network.{layer_idx}.weight"
        bias_key = f"network.{layer_idx}.bias"

        weights = state_dict[weight_key].numpy()
        bias = state_dict[bias_key].numpy()

        print(
            f"  Layer {i}: {weight_key} shape={weights.shape}, bias shape={bias.shape}"
        )

        # RTNeural expects weights as [out_features][in_features]
        # PyTorch Linear stores weights as [out_features, in_features]
        # So the shape is already correct

        rtneural_model["layers"].append(
            {"weights": weights.tolist(), "bias": bias.tolist()}
        )

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(rtneural_model, f)

    # Print file size
    file_size = output_path.stat().st_size
    print(f"\nExported to: {output_path}")
    print(f"File size: {file_size / 1024:.1f} KB")

    return rtneural_model


def main():
    parser = argparse.ArgumentParser(
        description="Export Lyrebird models to RTNeural JSON format"
    )
    parser.add_argument("model_path", help="Path to the .pth model file")
    parser.add_argument("output_path", help="Path to save the RTNeural JSON file")

    args = parser.parse_args()

    export_to_rtneural(args.model_path, args.output_path)
    print("\nDone! You can now load this model in the Lyrebird VST plugin.")


if __name__ == "__main__":
    main()
