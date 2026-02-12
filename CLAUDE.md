# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lyrebird is a neural audio effect cloning framework using PyTorch. It trains feed-forward neural networks to emulate audio effects (distortion, EQ, etc.) by learning the mapping between clean input audio and processed output audio using a Finite Impulse Response (FIR) inspired architecture. Note: FIR-based models cannot emulate effects with feedback/decay like reverb or delay.

## Commands

**Install for development:**
```bash
pip install -e ".[dev]"
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run a single test:**
```bash
pytest tests/test_lyrebird.py::TestFiniteImpulseResponseModel::test_forward_pass -v
```

**Launch Jupyter notebooks:**
```bash
jupyter notebook
```

## Architecture

The core module (`src/lyrebird_audio/core.py`) has three main components:

### FiniteImpulseResponseDataSet
PyTorch Dataset that loads paired WAV files (clean input + processed output). Returns `(input_buffer, output_sample)` tuples where:
- `input_buffer`: Last N samples from input waveform (shape: `channels, buffer_length`)
- `output_sample`: Single sample from output waveform at the same position (shape: `channels,`)

Requirements: Input/output files must have matching sample rates and lengths.

### FiniteImpulseResponseModel
Feed-forward neural network: `Flatten → Linear → ReLU → [Hidden layers] → Linear`

Constructor parameters:
- `input_size`: `buffer_length * num_channels`
- `hidden_size`: Width of hidden layers
- `num_layers`: Number of hidden layers (>= 1)
- `output_size`: Typically `num_channels`

### Training/Inference Functions
- `train_epoch()`: Single epoch training loop
- `evaluate()`: Validation/test loss computation
- `transform()`: Applies trained model to audio file sample-by-sample. First `buffer_length` samples are copied from input (no history available).

## Key Design Decisions

- **One-sample prediction**: Model predicts one output sample from N input samples (streaming-friendly design)
- **Buffer length**: Critical hyperparameter determining model's receptive field
- **No temporal padding**: First `buffer_length` samples copied directly from input
- **MSE loss**: Standard for audio regression
- **Device-agnostic**: CPU/CUDA support via `device` parameter

## Notebooks

- `fir_demo.ipynb`: Main training demonstration with piano distortion
- `big_muff_demo.ipynb`: Real guitar effect (Big Muff pedal) cloning
- `hyperparameter_exploration.ipynb`: Parameter tuning experiments

## Development Workflow

**IMPORTANT:** Follow this workflow for all code changes:

1. **Create a feature branch** before making changes:
   ```bash
   git checkout main && git pull
   git checkout -b feature/your-feature-name  # or fix/bug-description
   ```

2. **Make changes and commit** with clear messages

3. **Create a Pull Request** when done:
   ```bash
   git push -u origin <branch-name>
   gh pr create --title "Your PR title" --body "Description"
   ```

4. **Add reviewers:** `mfrancza`

5. **Use GitHub App identity** for ALL GitHub API interactions (comments, reviews, reactions, etc.) so they appear from `mfrancza-s-claude-code[bot]`:
   ```bash
   TOKEN=$(~/.claude-code/get-token.sh)
   curl -s -X POST \
     -H "Authorization: Bearer $TOKEN" \
     -H "Accept: application/vnd.github+json" \
     -H "Content-Type: application/json" \
     https://api.github.com/repos/mfrancza/lyrebird/issues/{PR_NUMBER}/comments \
     -d '{"body": "Your comment here"}'
   ```

   **IMPORTANT:** Always use this GitHub App token instead of `gh` CLI for any GitHub API calls that create attributable content (comments, reviews, reactions). Use `gh` CLI only for operations like creating PRs, merging, or reading data.

See `CONTRIBUTING.md` for full details on PR requirements, code style, and testing.

## Project Components

- **Python library** (`src/lyrebird_audio/`): Training framework (published as `lyrebird-audio`)
- **VST plugin** (`vst/`): Real-time audio plugin using RTNeural
- **Pi runner** (`pi/`): Raspberry Pi deployment
