#pragma once

#include "ModelInterface.h"
#include "RingBuffer.h"
#include <RTNeural/RTNeural.h>
#include <string>

namespace lyrebird {

/**
 * Small model configuration.
 * Optimized for Raspberry Pi and low-power devices.
 * ~5,184 MACs per sample.
 */
struct SmallConfig {
    static constexpr int BUFFER_LENGTH = 128;
    static constexpr int HIDDEN_SIZE = 32;
    static constexpr int NUM_LAYERS = 2;
    static constexpr int INPUT_SIZE = BUFFER_LENGTH;
    static constexpr int OUTPUT_SIZE = 1;
    static constexpr int CROSSFADE_LENGTH = 64;
};

/**
 * Small neural FIR model using RTNeural.
 *
 * Architecture: Linear(128->32) -> ReLU -> Linear(32->32) -> ReLU -> Linear(32->1)
 */
class ModelSmall : public IModel {
public:
    ModelSmall();
    ~ModelSmall() override = default;

    void reset() override;
    float processSample(float input) override;
    bool loadModel(const std::string& path) override;
    bool isLoaded() const override { return modelLoaded_; }
    void warmup(int iterations = 100) override;

    int getBufferLength() const override { return SmallConfig::BUFFER_LENGTH; }
    int getHiddenSize() const override { return SmallConfig::HIDDEN_SIZE; }
    int getNumLayers() const override { return SmallConfig::NUM_LAYERS; }
    int getLatencySamples() const override { return SmallConfig::BUFFER_LENGTH; }

    std::string getLastError() const override { return lastError_; }
    void clearError() override { lastError_.clear(); }

private:
    // RTNeural model: 2 hidden layers
    using ModelType = RTNeural::ModelT<float,
        SmallConfig::INPUT_SIZE,
        SmallConfig::OUTPUT_SIZE,
        RTNeural::DenseT<float, SmallConfig::INPUT_SIZE, SmallConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, SmallConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, SmallConfig::HIDDEN_SIZE, SmallConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, SmallConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, SmallConfig::HIDDEN_SIZE, SmallConfig::OUTPUT_SIZE>
    >;

    ModelType model_;
    ZeroCopyRingBuffer<float> ringBuffer_;

    bool modelLoaded_ = false;
    int sampleCount_ = 0;
    std::string lastError_;
};

} // namespace lyrebird
