#!/bin/bash
#
# Lyrebird VST Build Script (Linux)
#
# For Windows builds, use build-windows.bat or GitHub Actions.
#
# Usage:
#   ./build.sh          # Build the plugin
#   ./build.sh test     # Build and run tests
#   ./build.sh shell    # Open interactive shell
#   ./build.sh clean    # Remove build artifacts
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="lyrebird-vst-builder"

# Build Docker image if it doesn't exist
build_image() {
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Building Docker image..."
        docker build -t "$IMAGE_NAME" .
    fi
}

# Run command in Docker container
run_docker() {
    docker run --rm \
        -v "$(dirname "$SCRIPT_DIR")":/workspace \
        -w /workspace/vst \
        "$@"
}

case "${1:-build}" in
    build)
        build_image
        echo "Building Lyrebird VST plugin..."
        run_docker "$IMAGE_NAME" bash -c "
            git submodule update --init --recursive &&
            cmake -B build -DCMAKE_BUILD_TYPE=Release &&
            cmake --build build --config Release -j\$(nproc)
        "
        echo ""
        echo "Build complete!"
        echo "VST3 plugin: vst/build/plugin/LyrebirdVST_artefacts/VST3/"
        ;;

    test)
        build_image
        echo "Building and testing Lyrebird VST..."
        run_docker "$IMAGE_NAME" bash -c "
            git submodule update --init --recursive &&
            cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON &&
            cmake --build build --config Release -j\$(nproc) &&
            cd build && ctest --output-on-failure
        "
        ;;

    shell)
        build_image
        echo "Opening interactive shell..."
        run_docker -it "$IMAGE_NAME" bash
        ;;

    clean)
        echo "Cleaning build artifacts..."
        rm -rf build/
        echo "Done."
        ;;

    rebuild)
        echo "Rebuilding Docker image..."
        docker build --no-cache -t "$IMAGE_NAME" .
        ;;

    *)
        echo "Usage: $0 {build|test|shell|clean|rebuild}"
        exit 1
        ;;
esac
