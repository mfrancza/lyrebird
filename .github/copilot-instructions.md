# Copilot Instructions for Lyrebird

## Project Overview

Lyrebird is a neural audio effect cloning framework using PyTorch. It trains feed-forward neural networks to emulate audio effects (distortion, EQ, etc.) by learning the mapping between clean input audio and processed output audio using a Finite Impulse Response (FIR) inspired architecture.

**Important limitation:** FIR-based models cannot emulate effects with feedback/decay like reverb or delay.

The project has three deployment targets:
- **Python library** (`src/lyrebird_audio/`): Training framework (published as `lyrebird-audio`)
- **VST3 plugin** (`vst/`): Real-time audio plugin for DAWs using C++/JUCE/RTNeural
- **Raspberry Pi runner** (`pi/`): Lightweight edge deployment

## Environment Setup

### Python Library (Primary)

```bash
# Install for development (includes pytest)
pip install -e ".[dev]"

# CRITICAL: FFmpeg system libraries are required for torchaudio/torchcodec
# Without these, most tests will fail with:
#   RuntimeError: Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed
# Install them with:
sudo apt-get update
sudo apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswresample-dev

# Install linting/formatting tools (used in CI but not in [dev] dependencies)
pip install black ruff
```

### VST Plugin (C++)

The VST plugin uses CMake + JUCE + RTNeural (via git submodules). Build with:
```bash
cd vst
# Docker build (recommended):
./build.sh         # build
./build.sh test    # build + run tests

# Native Linux build:
sudo apt-get install -y build-essential cmake libasound2-dev libjack-jackd2-dev \
  libwebkit2gtk-4.0-dev libx11-dev libxcomposite-dev libxcursor-dev libxext-dev \
  libxinerama-dev libxrandr-dev libxrender-dev libglu1-mesa-dev mesa-common-dev
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure -E rtneural_test_functional
```

### Raspberry Pi

```bash
cd pi
pytest tests/test_ring_buffer.py tests/test_model_sizes.py tests/test_export_model.py -v
# Note: test_latency.py requires audio hardware (sounddevice/portaudio) - skip in CI
```

## Running Tests

```bash
# Python library - full test suite (36 tests)
pytest tests/ -v

# Run a single test
pytest tests/test_lyrebird.py::TestFiniteImpulseResponseModel::test_forward_pass -v

# Run a test class
pytest tests/test_lyrebird.py::TestFiniteImpulseResponseDataSet -v
```

Test classes in `tests/test_lyrebird.py`:
- `TestFiniteImpulseResponseDataSet` — WAV file loading, validation, dataset indexing
- `TestFiniteImpulseResponseModel` — Model initialization, forward pass, gradient flow, layer structure
- `TestTrainingFunctions` — `train_epoch()` and `evaluate()` integration tests
- `TestTransformFunction` — End-to-end audio transformation tests

## Linting & Formatting

```bash
# Check formatting (must pass in CI)
black --check src/lyrebird_audio/ tests/

# Auto-format
black src/lyrebird_audio/ tests/

# Lint (must pass in CI)
ruff check src/lyrebird_audio/ tests/

# Pi code formatting/linting
cd pi && find . -name "*.py" -type f | xargs black --check
cd pi && find . -name "*.py" -type f | xargs ruff check
```

## CI/CD Workflows

### `pr-checks.yml` — Runs on all PRs to `main`
Uses `dorny/paths-filter` to run only relevant checks based on changed files:

| Job | Triggers on changes to | Checks |
|-----|----------------------|--------|
| `python-checks` | `src/lyrebird_audio/`, `tests/`, `pyproject.toml` | black, ruff, pytest |
| `vst-checks` | `vst/` | CMake build, Catch2 tests |
| `pi-checks` | `pi/`, `src/lyrebird_audio/`, `pyproject.toml` | black, ruff, pytest (no hardware tests) |

### `build-vst.yml` — VST plugin builds
Triggers on pushes to `main` (vst/ changes) and PRs. Builds for Windows (MSVC) and Linux (GCC).

## Architecture

### Core Module (`src/lyrebird_audio/core.py`)

- **`FiniteImpulseResponseDataSet`**: PyTorch Dataset that loads paired WAV files (clean input + processed output). Returns `(input_buffer, output_sample)` tuples where `input_buffer` has shape `(channels, buffer_length)` and `output_sample` has shape `(channels,)`.
- **`FiniteImpulseResponseModel`**: Feed-forward neural network: `Flatten → Linear → ReLU → [Hidden layers] → Linear`. Constructor takes `(input_size, hidden_size, num_layers, output_size)`.
- **`train_epoch()`**: Single epoch training loop.
- **`evaluate()`**: Validation/test loss computation.
- **`transform()`**: Applies trained model to audio file sample-by-sample. First `buffer_length` samples are copied from input (no history available).

### CLI Entry Points (defined in `pyproject.toml`)
- `lyrebird-train` → `lyrebird_audio.train:main` — Train models in all size presets
- `lyrebird-export` → `lyrebird_audio.export_rtneural:main` — Export to RTNeural JSON
- `lyrebird-benchmark` → `lyrebird_audio.benchmark:main` — Inference benchmarking

### Model Size Presets (in `src/lyrebird_audio/export_rtneural.py`)
| Size | buffer_length | hidden_size | num_layers | Params | Target |
|------|---------------|-------------|------------|--------|--------|
| small | 128 | 32 | 2 | ~5K | Raspberry Pi |
| medium | 256 | 64 | 2 | ~21K | Laptop/Desktop |
| large | 512 | 128 | 3 | ~99K | High-end Desktop |

## Key Design Decisions

- **One-sample prediction**: Model predicts one output sample from N input samples (streaming-friendly)
- **Buffer length**: Critical hyperparameter determining the model's receptive field
- **No temporal padding**: First `buffer_length` samples copied directly from input
- **MSE loss**: Standard for audio regression
- **Device-agnostic**: CPU/CUDA support via `device` parameter

## Project File Layout

```
src/lyrebird_audio/       # Python package source
  core.py                 # Dataset, Model, train/eval/transform
  train.py                # Training CLI
  export_rtneural.py      # RTNeural export + MODEL_SIZE_PRESETS
  benchmark.py            # Inference benchmarking
tests/
  test_lyrebird.py        # All Python tests (pytest)
vst/                      # C++ VST3 plugin (CMake + JUCE + RTNeural)
  plugin/Source/           # Plugin source code
  test/                   # Catch2 tests
  modules/                # Git submodules (JUCE, RTNeural)
pi/                       # Raspberry Pi deployment
  tests/                  # Pi-specific tests
  tools/                  # Export/benchmark tools
  config/                 # System config examples
models/                   # Pre-trained model files (.pth, .json, .deploy.pth)
pyproject.toml            # Build config (hatchling), dependencies, entry points
```

## Common Errors and Workarounds

### Tests fail with `RuntimeError: Could not load libtorchcodec`
**Cause:** FFmpeg system libraries are not installed. The `torchaudio` and `torchcodec` packages require FFmpeg runtime libraries.
**Fix:**
```bash
sudo apt-get install -y ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswresample-dev
```
The CI workflow (`pr-checks.yml`) installs these in the `Install FFmpeg libraries` step.

### `black` or `ruff` not found
**Cause:** These are not included in the `[dev]` optional dependencies in `pyproject.toml` — they must be installed separately.
**Fix:**
```bash
pip install black ruff
```

### VST build fails with missing headers
**Cause:** Git submodules (JUCE, RTNeural) not initialized.
**Fix:**
```bash
git submodule update --init --recursive
```

### Pi tests fail importing `sounddevice`
**Cause:** `sounddevice` requires PortAudio system library and real audio hardware.
**Fix:** Skip hardware-dependent tests. Only run: `pytest pi/tests/test_ring_buffer.py pi/tests/test_model_sizes.py pi/tests/test_export_model.py -v`

## Development Workflow

1. Changes to Python code in `src/lyrebird_audio/` or `tests/` should be formatted with `black` and pass `ruff` linting
2. Always run `pytest tests/ -v` before submitting changes to the Python library
3. The build system uses `hatchling` — dependencies are defined in `pyproject.toml`
4. Commit messages should use present tense, keep the first line under 72 characters
5. See `CONTRIBUTING.md` for full PR requirements
