#pragma once

#include "ModelInterface.h"
#include "RingBuffer.h"
#include <RTNeural/RTNeural.h>
#include <string>

namespace lyrebird {

/**
 * Medium model configuration.
 * Suitable for laptop and desktop computers.
 * ~20,544 MACs per sample.
 */
struct MediumConfig {
    static constexpr int BUFFER_LENGTH = 256;
    static constexpr int HIDDEN_SIZE = 64;
    static constexpr int NUM_LAYERS = 2;
    static constexpr int INPUT_SIZE = BUFFER_LENGTH;
    static constexpr int OUTPUT_SIZE = 1;
    static constexpr int CROSSFADE_LENGTH = 64;
};

/**
 * Medium neural FIR model using RTNeural.
 *
 * Architecture: Linear(256->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->1)
 */
class ModelMedium : public IModel {
public:
    ModelMedium();
    ~ModelMedium() override = default;

    void reset() override;
    float processSample(float input) override;
    bool loadModel(const std::string& path) override;
    bool isLoaded() const override { return modelLoaded_; }
    void warmup(int iterations = 100) override;

    int getBufferLength() const override { return MediumConfig::BUFFER_LENGTH; }
    int getHiddenSize() const override { return MediumConfig::HIDDEN_SIZE; }
    int getNumLayers() const override { return MediumConfig::NUM_LAYERS; }
    int getLatencySamples() const override { return MediumConfig::BUFFER_LENGTH; }

    std::string getLastError() const override { return lastError_; }
    void clearError() override { lastError_.clear(); }

private:
    // RTNeural model: 2 hidden layers
    using ModelType = RTNeural::ModelT<float,
        MediumConfig::INPUT_SIZE,
        MediumConfig::OUTPUT_SIZE,
        RTNeural::DenseT<float, MediumConfig::INPUT_SIZE, MediumConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, MediumConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, MediumConfig::HIDDEN_SIZE, MediumConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, MediumConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, MediumConfig::HIDDEN_SIZE, MediumConfig::OUTPUT_SIZE>
    >;

    ModelType model_;
    ZeroCopyRingBuffer<float> ringBuffer_;

    bool modelLoaded_ = false;
    int sampleCount_ = 0;
    std::string lastError_;
};

} // namespace lyrebird
