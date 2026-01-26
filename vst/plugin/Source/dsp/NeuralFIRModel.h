#pragma once

#include "RingBuffer.h"
#include <RTNeural/RTNeural.h>
#include <memory>
#include <string>
#include <array>

namespace lyrebird {

/**
 * Default model configuration.
 * These values define the compile-time model architecture.
 * Change these and recompile to support different model sizes.
 */
struct ModelConfig {
    static constexpr int BUFFER_LENGTH = 512;
    static constexpr int HIDDEN_SIZE = 128;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int INPUT_SIZE = BUFFER_LENGTH;
    static constexpr int OUTPUT_SIZE = 1;
    static constexpr int CROSSFADE_LENGTH = 64;  // ~1.5ms at 44.1kHz
};

/**
 * Neural FIR model wrapper using RTNeural for real-time inference.
 *
 * This class encapsulates the RTNeural model and provides a simple
 * interface for sample-by-sample audio processing.
 *
 * The model architecture is fixed at compile time for optimal performance:
 * - Input: BUFFER_LENGTH samples
 * - Hidden: NUM_LAYERS x HIDDEN_SIZE with ReLU activation
 * - Output: 1 sample
 */
class NeuralFIRModel {
public:
    NeuralFIRModel();
    ~NeuralFIRModel();

    /**
     * Load model weights from a JSON file.
     *
     * @param jsonPath Path to the RTNeural JSON model file
     * @return true if loading succeeded
     */
    bool loadModel(const std::string& jsonPath);

    /**
     * Check if a model is currently loaded.
     *
     * @return true if a model has been loaded successfully
     */
    bool isLoaded() const { return modelLoaded_; }

    /**
     * Process a single input sample through the model.
     *
     * The sample is pushed to an internal ring buffer. Once the buffer
     * is full, the model performs inference and returns the output.
     * During the initial buffer fill period, the input is passed through.
     *
     * @param input The input sample
     * @return The processed output sample
     */
    float processSample(float input);

    /**
     * Reset the internal state (ring buffer).
     * Call this when starting a new audio stream.
     */
    void reset();

    /**
     * Warm up the model by running dummy inferences.
     * Call this in prepareToPlay() to avoid choppy audio at startup.
     * This pre-populates CPU caches for smoother real-time performance.
     *
     * @param iterations Number of dummy inferences to run (default: 100)
     */
    void warmup(int iterations = 100);

    /**
     * Get the latency in samples.
     * This is the buffer length - the number of samples delay introduced.
     *
     * @return Latency in samples
     */
    int getLatencySamples() const { return ModelConfig::BUFFER_LENGTH; }

    /**
     * Get the buffer length configuration.
     *
     * @return The buffer length
     */
    int getBufferLength() const { return ModelConfig::BUFFER_LENGTH; }

    /**
     * Get the hidden size configuration.
     *
     * @return The hidden layer size
     */
    int getHiddenSize() const { return ModelConfig::HIDDEN_SIZE; }

    /**
     * Get the number of layers configuration.
     *
     * @return The number of hidden layers
     */
    int getNumLayers() const { return ModelConfig::NUM_LAYERS; }

    /**
     * Get the last error message from model loading.
     *
     * @return The last error message, empty if no error
     */
    const std::string& getLastError() const { return lastError_; }

    /**
     * Clear the last error message.
     */
    void clearError() { lastError_.clear(); }

private:
    // Compile-time RTNeural model for optimal performance
    // Architecture: Linear(512->128) -> ReLU -> Linear(128->128) -> ReLU
    //            -> Linear(128->128) -> ReLU -> Linear(128->1)
    using ModelType = RTNeural::ModelT<float,
        ModelConfig::INPUT_SIZE,
        ModelConfig::OUTPUT_SIZE,
        RTNeural::DenseT<float, ModelConfig::INPUT_SIZE, ModelConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, ModelConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, ModelConfig::HIDDEN_SIZE, ModelConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, ModelConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, ModelConfig::HIDDEN_SIZE, ModelConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, ModelConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, ModelConfig::HIDDEN_SIZE, ModelConfig::OUTPUT_SIZE>
    >;

    ModelType model_;
    RingBuffer<float> ringBuffer_;
    std::array<float, ModelConfig::INPUT_SIZE> inferenceBuffer_;

    bool modelLoaded_ = false;
    int sampleCount_ = 0;  // Track samples for initial buffer fill
    std::string lastError_;
};

} // namespace lyrebird
