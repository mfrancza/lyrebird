#!/usr/bin/env python3
"""
Train Lyrebird models in all size presets (Small, Medium, Large).

Usage:
    python train_all_sizes.py --input clean.wav --output processed.wav --name effect_name
    python train_all_sizes.py --input clean.wav --output processed.wav --name big_muff --epochs 10
    python train_all_sizes.py --input clean.wav --output processed.wav --name big_muff --sizes small medium

Model Size Presets:
    small:  buffer_length=128, hidden_size=32,  num_layers=2  (~5K params, Raspberry Pi)
    medium: buffer_length=256, hidden_size=64,  num_layers=2  (~21K params, Laptop/Desktop)
    large:  buffer_length=512, hidden_size=128, num_layers=3  (~99K params, High-end Desktop)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lyrebird
from export_rtneural import export_to_rtneural, MODEL_SIZE_PRESETS


def count_parameters(model):
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    input_wav: str,
    output_wav: str,
    buffer_length: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    num_workers: int = 4,
) -> tuple:
    """
    Train a single model with the specified configuration.

    Returns:
        tuple: (model, final_loss, training_dataset)
    """
    # Create dataset
    print(f"\nLoading training data...")
    dataset = lyrebird.FiniteImpulseResponseDataSet(
        input_wav_path=input_wav,
        output_wav_path=output_wav,
        buffer_length=buffer_length,
    )

    print(f"  Samples: {len(dataset):,}")
    print(f"  Sample rate: {dataset.get_sample_rate()} Hz")
    print(f"  Channels: {dataset.get_input_channels()}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    num_channels = dataset.get_input_channels()
    input_size = buffer_length * num_channels

    model = lyrebird.FiniteImpulseResponseModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=dataset.get_output_channels(),
    )
    model = model.to(device)

    print(f"\nModel architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Parameters: {count_parameters(model):,}")

    # Setup training
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    final_loss = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            # Progress update every 5000 batches
            if batch_idx % 5000 == 0:
                current = batch_idx * len(X)
                print(f"  loss: {loss.item():.6f}  [{current:>7d}/{len(dataset):>7d}]")

        avg_loss = epoch_loss / num_batches
        print(f"  Average loss: {avg_loss:.6f}")
        final_loss = avg_loss

    return model, final_loss, dataset


def save_model(
    model,
    optimizer,
    dataset,
    buffer_length: int,
    hidden_size: int,
    num_layers: int,
    output_path: str,
):
    """Save the trained model to a .pth file."""
    num_channels = dataset.get_input_channels()
    input_size = buffer_length * num_channels

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "buffer_length": buffer_length,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "output_size": dataset.get_output_channels(),
        },
        output_path,
    )
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Lyrebird models in multiple size presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all sizes for Big Muff effect
  python train_all_sizes.py \\
      --input "Big Muff piano clean.wav" \\
      --output "Big Muff piano distorted.wav" \\
      --name big_muff

  # Train only small and medium sizes with more epochs
  python train_all_sizes.py \\
      --input clean.wav --output processed.wav \\
      --name my_effect --sizes small medium --epochs 10

  # Train with custom batch size and learning rate
  python train_all_sizes.py \\
      --input clean.wav --output processed.wav \\
      --name my_effect --batch-size 64 --lr 0.0005
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Path to clean/dry input WAV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to processed/wet output WAV file"
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Name for the effect (used in output filenames)",
    )
    parser.add_argument(
        "--sizes",
        "-s",
        nargs="+",
        choices=["small", "medium", "large"],
        default=["small", "medium", "large"],
        help="Model sizes to train (default: all)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        default="models",
        help="Output directory for trained models (default: models)",
    )
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting to RTNeural JSON format",
    )

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    if not os.path.exists(args.output):
        print(f"Error: Output file not found: {args.output}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Train each size
    results = {}

    for size_name in args.sizes:
        preset = MODEL_SIZE_PRESETS[size_name]

        print("\n" + "=" * 60)
        print(f"TRAINING {size_name.upper()} MODEL")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Buffer length: {preset['buffer_length']}")
        print(f"  Hidden size: {preset['hidden_size']}")
        print(f"  Num layers: {preset['num_layers']}")

        try:
            model, final_loss, dataset = train_model(
                input_wav=args.input,
                output_wav=args.output,
                buffer_length=preset["buffer_length"],
                hidden_size=preset["hidden_size"],
                num_layers=preset["num_layers"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                num_workers=args.num_workers,
            )

            # Save PyTorch model
            pth_path = output_dir / f"{args.name}_{size_name}.pth"
            save_model(
                model=model,
                optimizer=None,  # Don't save optimizer for inference-only models
                dataset=dataset,
                buffer_length=preset["buffer_length"],
                hidden_size=preset["hidden_size"],
                num_layers=preset["num_layers"],
                output_path=str(pth_path),
            )

            # Export to RTNeural JSON
            if not args.no_export:
                json_path = output_dir / f"{args.name}_{size_name}.json"
                print(f"\nExporting to RTNeural format...")
                export_to_rtneural(str(pth_path), str(json_path), size=size_name)

            results[size_name] = {
                "status": "success",
                "final_loss": final_loss,
                "parameters": count_parameters(model),
                "pth_path": str(pth_path),
            }

        except Exception as e:
            print(f"\nError training {size_name} model: {e}")
            results[size_name] = {"status": "failed", "error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for size_name, result in results.items():
        if result["status"] == "success":
            print(f"\n{size_name.upper()}:")
            print(f"  Status: SUCCESS")
            print(f"  Final loss: {result['final_loss']:.6f}")
            print(f"  Parameters: {result['parameters']:,}")
            print(f"  Model: {result['pth_path']}")
        else:
            print(f"\n{size_name.upper()}:")
            print(f"  Status: FAILED")
            print(f"  Error: {result['error']}")

    # Check for failures
    failed = [name for name, r in results.items() if r["status"] == "failed"]
    if failed:
        print(f"\nWarning: {len(failed)} model(s) failed to train: {', '.join(failed)}")
        return 1

    print(f"\nAll models saved to: {output_dir}/")
    print("\nTo use in the VST plugin:")
    print("  1. Open the plugin in your DAW")
    print("  2. Select the matching model size from the dropdown")
    print("  3. Click 'Load Model...' and select the .json file")

    return 0


if __name__ == "__main__":
    sys.exit(main())
