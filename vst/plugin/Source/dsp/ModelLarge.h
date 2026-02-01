#pragma once

#include "ModelInterface.h"
#include "RingBuffer.h"
#include <RTNeural/RTNeural.h>
#include <string>

namespace lyrebird {

/**
 * Large model configuration.
 * Highest quality, for high-end desktop computers.
 * ~98,560 MACs per sample.
 */
struct LargeConfig {
    static constexpr int BUFFER_LENGTH = 512;
    static constexpr int HIDDEN_SIZE = 128;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int INPUT_SIZE = BUFFER_LENGTH;
    static constexpr int OUTPUT_SIZE = 1;
    static constexpr int CROSSFADE_LENGTH = 64;
};

/**
 * Large neural FIR model using RTNeural.
 *
 * Architecture: Linear(512->128) -> ReLU -> Linear(128->128) -> ReLU
 *            -> Linear(128->128) -> ReLU -> Linear(128->1)
 */
class ModelLarge : public IModel {
public:
    ModelLarge();
    ~ModelLarge() override = default;

    void reset() override;
    float processSample(float input) override;
    bool loadModel(const std::string& path) override;
    bool isLoaded() const override { return modelLoaded_; }
    void warmup(int iterations = 100) override;

    int getBufferLength() const override { return LargeConfig::BUFFER_LENGTH; }
    int getHiddenSize() const override { return LargeConfig::HIDDEN_SIZE; }
    int getNumLayers() const override { return LargeConfig::NUM_LAYERS; }
    int getLatencySamples() const override { return LargeConfig::BUFFER_LENGTH; }

    std::string getLastError() const override { return lastError_; }
    void clearError() override { lastError_.clear(); }

private:
    // RTNeural model: 3 hidden layers
    using ModelType = RTNeural::ModelT<float,
        LargeConfig::INPUT_SIZE,
        LargeConfig::OUTPUT_SIZE,
        RTNeural::DenseT<float, LargeConfig::INPUT_SIZE, LargeConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, LargeConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, LargeConfig::HIDDEN_SIZE, LargeConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, LargeConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, LargeConfig::HIDDEN_SIZE, LargeConfig::HIDDEN_SIZE>,
        RTNeural::ReLuActivationT<float, LargeConfig::HIDDEN_SIZE>,
        RTNeural::DenseT<float, LargeConfig::HIDDEN_SIZE, LargeConfig::OUTPUT_SIZE>
    >;

    ModelType model_;
    ZeroCopyRingBuffer<float> ringBuffer_;

    bool modelLoaded_ = false;
    int sampleCount_ = 0;
    std::string lastError_;
};

} // namespace lyrebird
