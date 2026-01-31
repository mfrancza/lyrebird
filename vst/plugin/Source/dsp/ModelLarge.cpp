#include "ModelLarge.h"
#include "../utils/ModelLoader.h"
#include <fstream>
#include <cmath>

namespace lyrebird {

ModelLarge::ModelLarge() {
    ringBuffer_.prepare(LargeConfig::BUFFER_LENGTH);
}

bool ModelLarge::loadModel(const std::string& jsonPath) {
    lastError_.clear();

    try {
        std::ifstream jsonFile(jsonPath);
        if (!jsonFile.is_open()) {
            lastError_ = "Failed to open model file: " + jsonPath;
            return false;
        }

        nlohmann::json modelJson;
        jsonFile >> modelJson;

        // Validate config matches our compile-time model
        if (modelJson.contains("config")) {
            auto& config = modelJson["config"];
            int bufferLength = config.value("buffer_length", 0);
            int hiddenSize = config.value("hidden_size", 0);
            int numLayers = config.value("num_layers", 0);

            if (bufferLength != LargeConfig::BUFFER_LENGTH) {
                lastError_ = "Model buffer_length mismatch: expected " +
                             std::to_string(LargeConfig::BUFFER_LENGTH) +
                             ", got " + std::to_string(bufferLength);
                return false;
            }
            if (hiddenSize != LargeConfig::HIDDEN_SIZE) {
                lastError_ = "Model hidden_size mismatch: expected " +
                             std::to_string(LargeConfig::HIDDEN_SIZE) +
                             ", got " + std::to_string(hiddenSize);
                return false;
            }
            if (numLayers != LargeConfig::NUM_LAYERS) {
                lastError_ = "Model num_layers mismatch: expected " +
                             std::to_string(LargeConfig::NUM_LAYERS) +
                             ", got " + std::to_string(numLayers);
                return false;
            }
        }

        if (!modelJson.contains("layers")) {
            lastError_ = "Model JSON missing 'layers' array";
            return false;
        }

        auto& layers = modelJson["layers"];

        // Large model has 4 dense layers (3 hidden + 1 output)
        if (layers.size() < 4) {
            lastError_ = "Expected 4 layers, got " + std::to_string(layers.size());
            return false;
        }

        // Load layer 0: input -> hidden
        {
            auto& layerWeights = layers[0]["weights"];
            auto& layerBias = layers[0]["bias"];
            auto& dense = model_.get<0>();

            std::vector<std::vector<float>> weights(LargeConfig::HIDDEN_SIZE,
                std::vector<float>(LargeConfig::INPUT_SIZE));
            std::vector<float> bias(LargeConfig::HIDDEN_SIZE);

            for (int i = 0; i < LargeConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < LargeConfig::INPUT_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 1: hidden -> hidden (index 2 in RTNeural after ReLU)
        {
            auto& layerWeights = layers[1]["weights"];
            auto& layerBias = layers[1]["bias"];
            auto& dense = model_.get<2>();

            std::vector<std::vector<float>> weights(LargeConfig::HIDDEN_SIZE,
                std::vector<float>(LargeConfig::HIDDEN_SIZE));
            std::vector<float> bias(LargeConfig::HIDDEN_SIZE);

            for (int i = 0; i < LargeConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < LargeConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 2: hidden -> hidden (index 4 in RTNeural)
        {
            auto& layerWeights = layers[2]["weights"];
            auto& layerBias = layers[2]["bias"];
            auto& dense = model_.get<4>();

            std::vector<std::vector<float>> weights(LargeConfig::HIDDEN_SIZE,
                std::vector<float>(LargeConfig::HIDDEN_SIZE));
            std::vector<float> bias(LargeConfig::HIDDEN_SIZE);

            for (int i = 0; i < LargeConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < LargeConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 3: hidden -> output (index 6 in RTNeural)
        {
            auto& layerWeights = layers[3]["weights"];
            auto& layerBias = layers[3]["bias"];
            auto& dense = model_.get<6>();

            std::vector<std::vector<float>> weights(LargeConfig::OUTPUT_SIZE,
                std::vector<float>(LargeConfig::HIDDEN_SIZE));
            std::vector<float> bias(LargeConfig::OUTPUT_SIZE);

            for (int i = 0; i < LargeConfig::OUTPUT_SIZE; ++i) {
                for (int j = 0; j < LargeConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        modelLoaded_ = true;
        reset();
        return true;

    } catch (const std::exception& e) {
        lastError_ = std::string("Error loading model: ") + e.what();
        return false;
    }
}

float ModelLarge::processSample(float input) {
    ringBuffer_.push(input);
    sampleCount_++;

    if (sampleCount_ < LargeConfig::BUFFER_LENGTH || !modelLoaded_) {
        return input;
    }

    float modelOutput = model_.forward(ringBuffer_.data());

    int samplesAfterBufferFill = sampleCount_ - LargeConfig::BUFFER_LENGTH;
    if (samplesAfterBufferFill < LargeConfig::CROSSFADE_LENGTH) {
        float t = static_cast<float>(samplesAfterBufferFill) /
                  static_cast<float>(LargeConfig::CROSSFADE_LENGTH);
        return input * (1.0f - t) + modelOutput * t;
    }

    return modelOutput;
}

void ModelLarge::reset() {
    ringBuffer_.reset();
    model_.reset();
    sampleCount_ = 0;
}

void ModelLarge::warmup(int iterations) {
    if (!modelLoaded_) {
        return;
    }

    for (int i = 0; i < LargeConfig::INPUT_SIZE; ++i) {
        ringBuffer_.push(0.1f * std::sin(static_cast<float>(i) * 0.1f));
    }

    volatile float dummy = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        dummy = model_.forward(ringBuffer_.data());
    }

    model_.reset();
    ringBuffer_.reset();
}

} // namespace lyrebird
