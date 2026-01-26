#!/usr/bin/env python3
"""Generate a minimal test model JSON for inference testing."""

import json
import numpy as np

BUFFER_LENGTH = 512
HIDDEN_SIZE = 128
NUM_LAYERS = 3

def generate_test_model():
    """Generate a model that produces predictable non-zero output.

    The model sums input values and scales them, producing output that
    differs from input (to verify model is actually running).
    """
    model = {
        "config": {
            "buffer_length": BUFFER_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS
        },
        "layers": []
    }

    # Layer 0: 512 -> 128
    # Use small weights that sum the inputs
    weights_0 = np.ones((HIDDEN_SIZE, BUFFER_LENGTH)) / BUFFER_LENGTH * 0.1
    bias_0 = np.zeros(HIDDEN_SIZE)
    model["layers"].append({
        "weights": weights_0.tolist(),
        "bias": bias_0.tolist()
    })

    # Layer 1: 128 -> 128
    # Identity-ish weights
    weights_1 = np.eye(HIDDEN_SIZE) * 0.5
    bias_1 = np.zeros(HIDDEN_SIZE)
    model["layers"].append({
        "weights": weights_1.tolist(),
        "bias": bias_1.tolist()
    })

    # Layer 2: 128 -> 128
    # Identity-ish weights
    weights_2 = np.eye(HIDDEN_SIZE) * 0.5
    bias_2 = np.zeros(HIDDEN_SIZE)
    model["layers"].append({
        "weights": weights_2.tolist(),
        "bias": bias_2.tolist()
    })

    # Layer 3: 128 -> 1
    # Sum all hidden values
    weights_3 = np.ones((1, HIDDEN_SIZE)) * 0.1
    bias_3 = np.array([0.0])
    model["layers"].append({
        "weights": weights_3.tolist(),
        "bias": bias_3.tolist()
    })

    return model

if __name__ == "__main__":
    model = generate_test_model()

    output_path = "fixtures/test_model.json"
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Generated test model at {output_path}")
    print(f"Config: buffer_length={BUFFER_LENGTH}, hidden_size={HIDDEN_SIZE}, num_layers={NUM_LAYERS}")
