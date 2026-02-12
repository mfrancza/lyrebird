# Lyrebird VST3 Plugin

A real-time VST3 audio plugin that runs trained Lyrebird neural FIR models for DAW integration.

## Features

- Real-time neural network inference using RTNeural (XSIMD backend)
- Dry/wet mix control
- Bypass functionality
- Stereo support
- Automatic latency compensation

## Requirements

- Docker (recommended), OR:
- CMake 3.21+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Git (for submodules)
- Linux: JUCE dependencies (see Native Build below)

## Building with Docker (Recommended)

Docker provides a consistent build environment with all dependencies pre-installed.

### Quick Start

```bash
cd lyrebird/vst

# Build the plugin
./build.sh

# Build and run tests
./build.sh test

# Open interactive shell
./build.sh shell

# Clean build artifacts
./build.sh clean
```

### Using Docker Compose

```bash
# Build plugin
docker-compose run --rm build

# Run tests
docker-compose run --rm test

# Interactive shell
docker-compose run --rm shell
```

### Manual Docker Commands

```bash
# Build the Docker image
docker build -t lyrebird-vst-builder .

# Run full build with tests
docker run --rm -v $(pwd)/..:/workspace lyrebird-vst-builder

# Interactive shell
docker run --rm -it -v $(pwd)/..:/workspace lyrebird-vst-builder bash
```

The VST3 plugin will be output to:
```
build/plugin/LyrebirdVST_artefacts/VST3/Lyrebird.vst3
```

## Native Build (without Docker)

### Linux Dependencies

```bash
sudo apt install -y \
    build-essential cmake git pkg-config \
    libasound2-dev libjack-jackd2-dev ladspa-sdk \
    libcurl4-openssl-dev libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev \
    libxext-dev libxinerama-dev libxrandr-dev \
    libxrender-dev libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev
```

### Clone Submodules

```bash
cd lyrebird/vst
git submodule add https://github.com/juce-framework/JUCE modules/JUCE
git submodule add https://github.com/jatinchowdhury18/RTNeural modules/RTNeural
```

Or if submodules are already configured:

```bash
git submodule update --init --recursive
```

### Configure and Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Build with Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build
```

## Exporting Models

Convert trained PyTorch models to the RTNeural JSON format:

```bash
cd lyrebird/vst/python
python export_model.py ../../models/big_muff_fir_model.pth big_muff.json \
    --buffer_length 512 \
    --hidden_size 128 \
    --num_layers 3
```

### Arguments

| Argument | Description |
|----------|-------------|
| `model_path` | Path to the trained .pth file |
| `output_path` | Path for the output JSON file |
| `--buffer_length` | Buffer length used during training |
| `--hidden_size` | Hidden layer size |
| `--num_layers` | Number of hidden layers |
| `--num_channels` | Number of audio channels (default: 1) |

## Model Configuration

The plugin is compiled with a fixed model architecture for optimal performance:

| Parameter | Value |
|-----------|-------|
| Buffer Length | 512 samples |
| Hidden Size | 128 |
| Hidden Layers | 3 |
| Latency | ~11.6ms @ 44.1kHz |

To use models with different configurations, modify `ModelConfig` in `plugin/Source/dsp/NeuralFIRModel.h` and recompile.

## Usage in DAW

1. Copy `Lyrebird.vst3` to your VST3 plugin folder:
   - **Windows:** `C:\Program Files\Common Files\VST3\`
   - **macOS:** `/Library/Audio/Plug-Ins/VST3/`
   - **Linux:** `~/.vst3/`

2. Load the plugin on an audio track in your DAW

3. Click "Load Model..." and select a `.json` model file

4. Adjust dry/wet mix as desired

## Limitations

- Sample rate: Models trained at 44.1kHz work best at that rate
- Mono models: Stereo is processed as two independent mono channels
- Fixed architecture: Model dimensions must match compile-time config
- CPU only: No GPU acceleration (but SIMD optimized via XSIMD)

## Project Structure

```
vst/
├── CMakeLists.txt           # Top-level CMake
├── README.md                # This file
├── Dockerfile               # Docker build environment
├── docker-compose.yml       # Docker compose config
├── build.sh                 # Build helper script
├── modules/                 # Git submodules
│   ├── JUCE/
│   └── RTNeural/
├── python/
│   ├── export_model.py      # Model conversion script
│   └── test_export_model.py # Export script tests
├── plugin/
│   ├── CMakeLists.txt
│   └── Source/
│       ├── PluginProcessor.cpp/h   # Audio processing
│       ├── PluginEditor.cpp/h      # GUI
│       ├── dsp/
│       │   ├── NeuralFIRModel.cpp/h  # RTNeural wrapper
│       │   └── RingBuffer.cpp/h      # Sample buffer
│       └── utils/
│           └── ModelLoader.h         # JSON utilities
└── test/
    ├── CMakeLists.txt
    ├── test_ring_buffer.cpp
    └── test_neural_model.cpp
```

## Performance

The plugin uses RTNeural with the XSIMD backend for SIMD-optimized inference. Expected performance:

- **Target:** >10x real-time (process 1 second in <100ms)
- **Typical:** 50-100x real-time on modern CPUs

Profile with your specific hardware and model to verify.

## License

See parent project license.
