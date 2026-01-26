# Lyrebird VST Build Environment (Linux)
#
# For Windows builds, use build-windows.bat or GitHub Actions.
#
# Build image:
#   docker build -t lyrebird-vst-builder .
#
# Build plugin:
#   docker run --rm -v $(pwd)/..:/workspace lyrebird-vst-builder
#
# Interactive shell:
#   docker run --rm -it -v $(pwd)/..:/workspace lyrebird-vst-builder bash

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and JUCE dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    pkg-config \
    # JUCE Linux dependencies
    libasound2-dev \
    libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev \
    libfreetype6-dev \
    libx11-dev \
    libxcomposite-dev \
    libxcursor-dev \
    libxext-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev \
    mesa-common-dev \
    # Additional useful tools
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Configure git to trust mounted volumes
RUN git config --global --add safe.directory '*'

# Set up workspace
WORKDIR /workspace

# Default command: build the plugin
CMD ["/bin/bash", "-c", "\
    cd /workspace/vst && \
    git submodule update --init --recursive && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON && \
    cmake --build build --config Release -j$(nproc) && \
    echo '=== Build Complete ===' && \
    echo 'VST3 plugin: vst/build/plugin/LyrebirdVST_artefacts/VST3/' && \
    echo 'Running tests...' && \
    cd build && ctest --output-on-failure \
"]
