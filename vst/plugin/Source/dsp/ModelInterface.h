#pragma once

#include <memory>
#include <string>

namespace lyrebird {

/**
 * Model size presets with their configurations.
 */
enum class ModelSize {
    Small,   // Buffer: 128, Hidden: 32, Layers: 2 - For Raspberry Pi
    Medium,  // Buffer: 256, Hidden: 64, Layers: 2 - For Laptop/Desktop
    Large    // Buffer: 512, Hidden: 128, Layers: 3 - For High-end Desktop
};

/**
 * Abstract interface for neural FIR models.
 *
 * This interface allows the plugin to work with different model sizes
 * through polymorphism while RTNeural uses compile-time templates
 * for optimal performance.
 */
class IModel {
public:
    virtual ~IModel() = default;

    /**
     * Reset the internal state (ring buffer and model state).
     * Call this when starting a new audio stream.
     */
    virtual void reset() = 0;

    /**
     * Process a single input sample through the model.
     *
     * @param input The input sample
     * @return The processed output sample
     */
    virtual float processSample(float input) = 0;

    /**
     * Load model weights from a JSON file.
     *
     * @param path Path to the RTNeural JSON model file
     * @return true if loading succeeded
     */
    virtual bool loadModel(const std::string& path) = 0;

    /**
     * Check if a model is currently loaded.
     *
     * @return true if a model has been loaded successfully
     */
    virtual bool isLoaded() const = 0;

    /**
     * Warm up the model by running dummy inferences.
     * Call this in prepareToPlay() to avoid choppy audio at startup.
     *
     * @param iterations Number of dummy inferences to run
     */
    virtual void warmup(int iterations = 100) = 0;

    /**
     * Get the buffer length (latency in samples).
     *
     * @return Buffer length / latency in samples
     */
    virtual int getBufferLength() const = 0;

    /**
     * Get the hidden layer size.
     *
     * @return Hidden layer size
     */
    virtual int getHiddenSize() const = 0;

    /**
     * Get the number of hidden layers.
     *
     * @return Number of hidden layers
     */
    virtual int getNumLayers() const = 0;

    /**
     * Get the latency in samples.
     *
     * @return Latency in samples (same as buffer length)
     */
    virtual int getLatencySamples() const = 0;

    /**
     * Get the last error message from model loading.
     *
     * @return The last error message, empty if no error
     */
    virtual std::string getLastError() const = 0;

    /**
     * Clear the last error message.
     */
    virtual void clearError() = 0;
};

/**
 * Get the display name for a model size.
 *
 * @param size The model size enum value
 * @return Human-readable name (e.g., "Small", "Medium", "Large")
 */
inline const char* getModelSizeName(ModelSize size) {
    switch (size) {
        case ModelSize::Small:  return "Small";
        case ModelSize::Medium: return "Medium";
        case ModelSize::Large:  return "Large";
        default:                return "Unknown";
    }
}

/**
 * Get the buffer length for a model size.
 */
inline int getBufferLengthForSize(ModelSize size) {
    switch (size) {
        case ModelSize::Small:  return 128;
        case ModelSize::Medium: return 256;
        case ModelSize::Large:  return 512;
        default:                return 512;
    }
}

/**
 * Get the hidden size for a model size.
 */
inline int getHiddenSizeForSize(ModelSize size) {
    switch (size) {
        case ModelSize::Small:  return 32;
        case ModelSize::Medium: return 64;
        case ModelSize::Large:  return 128;
        default:                return 128;
    }
}

/**
 * Get the number of layers for a model size.
 */
inline int getNumLayersForSize(ModelSize size) {
    switch (size) {
        case ModelSize::Small:  return 2;
        case ModelSize::Medium: return 2;
        case ModelSize::Large:  return 3;
        default:                return 3;
    }
}

} // namespace lyrebird
