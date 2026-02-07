#!/usr/bin/env python3
"""
Export Lyrebird training checkpoints to deployment-ready state dicts.

Training checkpoints (from train_all_sizes.py) contain model_state_dict,
optimizer_state_dict, and metadata. The Pi deployment tools expect raw
state dicts. This script extracts and validates the state dict for deployment.

Usage:
    python export_model.py --checkpoint models/big_muff_small.pth
    python export_model.py --checkpoint models/big_muff_small.pth --output models/deployed.pth
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyrebird import FiniteImpulseResponseModel
from pi.model_sizes import MODEL_SIZE_PRESETS


def find_size_preset(buffer_length, hidden_size, num_layers):
    """Find matching size preset name, or None if no match."""
    for name, preset in MODEL_SIZE_PRESETS.items():
        if (
            preset["buffer_length"] == buffer_length
            and preset["hidden_size"] == hidden_size
            and preset["num_layers"] == num_layers
        ):
            return name
    return None


def export_model(checkpoint_path: str, output_path: str) -> None:
    """
    Export a training checkpoint to a deployment-ready state dict.

    Args:
        checkpoint_path: Path to training checkpoint (.pth with model_state_dict key)
        output_path: Path for output deployment state dict
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        print(
            f"Error: {checkpoint_path} does not appear to be a training checkpoint.",
            file=sys.stderr,
        )
        print(
            "Expected a dict with 'model_state_dict' key (from train_all_sizes.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract metadata
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    output_size = checkpoint["output_size"]
    buffer_length = checkpoint["buffer_length"]

    # Print model info
    size_preset = find_size_preset(buffer_length, hidden_size, num_layers)
    if size_preset:
        print(f"  Size preset: {size_preset.upper()}")
    print(f"  Buffer length: {buffer_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")

    # Validate by loading into a model
    state_dict = checkpoint["model_state_dict"]

    model = FiniteImpulseResponseModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    )
    model.load_state_dict(state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Save raw state dict
    torch.save(state_dict, output_path)
    print(f"\nExported deployment model: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Lyrebird training checkpoints to deployment-ready state dicts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default naming (<name>.pth -> <name>.deploy.pth)
  python export_model.py --checkpoint models/big_muff_small.pth

  # Export with custom output path
  python export_model.py --checkpoint models/big_muff_small.pth --output models/deployed.pth
        """,
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to training checkpoint (.pth)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for deployment model (default: <name>.deploy.pth)",
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if output_path is None:
        checkpoint_path = Path(args.checkpoint)
        output_path = str(checkpoint_path.with_suffix(".deploy.pth"))

    export_model(args.checkpoint, output_path)


if __name__ == "__main__":
    main()
