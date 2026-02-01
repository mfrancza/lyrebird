"""
Model size presets for Lyrebird.

These presets match the VST plugin's compile-time model variants
and provide consistent configurations across Python and C++ implementations.
"""

# Model size presets matching the VST plugin
MODEL_SIZE_PRESETS = {
    "small": {
        "buffer_length": 128,
        "hidden_size": 32,
        "num_layers": 2,
        "description": "Optimized for Raspberry Pi real-time processing",
    },
    "medium": {
        "buffer_length": 256,
        "hidden_size": 64,
        "num_layers": 2,
        "description": "Balanced for laptop/desktop use",
    },
    "large": {
        "buffer_length": 512,
        "hidden_size": 128,
        "num_layers": 3,
        "description": "Maximum quality for high-end systems",
    },
}


def get_preset(size: str) -> dict:
    """
    Get model configuration for a size preset.

    Args:
        size: One of 'small', 'medium', 'large'

    Returns:
        Dictionary with buffer_length, hidden_size, num_layers

    Raises:
        ValueError: If size is not a valid preset name
    """
    size = size.lower()
    if size not in MODEL_SIZE_PRESETS:
        valid = ", ".join(MODEL_SIZE_PRESETS.keys())
        raise ValueError(f"Invalid size '{size}'. Valid options: {valid}")
    return MODEL_SIZE_PRESETS[size]


def add_size_arguments(parser, default_size: str = None) -> None:
    """
    Add model size arguments to an argument parser.

    Adds mutually exclusive --size OR manual --buffer-length/--hidden-size/--num-layers.

    Args:
        parser: argparse.ArgumentParser instance
        default_size: Default size preset (if None, manual args are required)
    """
    group = parser.add_argument_group('model size')

    group.add_argument(
        '--size', '-s',
        type=str,
        choices=['small', 'medium', 'large'],
        default=default_size,
        help=f'Model size preset (default: {default_size}). '
             'Small: 128/32/2 (Pi), Medium: 256/64/2 (Desktop), Large: 512/128/3 (High-end)'
    )
    group.add_argument(
        '--buffer-length',
        type=int,
        default=None,
        help='Model buffer length (overrides --size)'
    )
    group.add_argument(
        '--hidden-size',
        type=int,
        default=None,
        help='Model hidden size (overrides --size)'
    )
    group.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='Model number of layers (overrides --size)'
    )


def resolve_size_arguments(args) -> tuple:
    """
    Resolve size arguments from parsed args.

    Returns (buffer_length, hidden_size, num_layers) based on either
    --size preset or manual arguments.

    Args:
        args: Parsed argparse namespace

    Returns:
        Tuple of (buffer_length, hidden_size, num_layers)

    Raises:
        ValueError: If neither --size nor manual args are provided
    """
    # Check for manual overrides
    has_manual = any([
        args.buffer_length is not None,
        args.hidden_size is not None,
        args.num_layers is not None,
    ])

    if has_manual:
        # Use manual values, with defaults from size preset if specified
        if args.size:
            preset = get_preset(args.size)
            buffer_length = args.buffer_length or preset["buffer_length"]
            hidden_size = args.hidden_size or preset["hidden_size"]
            num_layers = args.num_layers or preset["num_layers"]
        else:
            # Require all manual values if no size preset
            if args.buffer_length is None:
                raise ValueError("--buffer-length required when not using --size")
            if args.hidden_size is None:
                raise ValueError("--hidden-size required when not using --size")
            buffer_length = args.buffer_length
            hidden_size = args.hidden_size
            num_layers = args.num_layers or 1
        return buffer_length, hidden_size, num_layers

    elif args.size:
        preset = get_preset(args.size)
        return preset["buffer_length"], preset["hidden_size"], preset["num_layers"]

    else:
        raise ValueError("Either --size or manual --buffer-length/--hidden-size required")


def print_size_info(size: str = None, buffer_length: int = None,
                    hidden_size: int = None, num_layers: int = None) -> None:
    """Print model size information."""
    if size:
        preset = get_preset(size)
        print(f"Model size: {size.upper()}")
        print(f"  {preset['description']}")
        print(f"  Buffer length: {preset['buffer_length']}")
        print(f"  Hidden size: {preset['hidden_size']}")
        print(f"  Num layers: {preset['num_layers']}")
    else:
        print(f"Model configuration:")
        print(f"  Buffer length: {buffer_length}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
