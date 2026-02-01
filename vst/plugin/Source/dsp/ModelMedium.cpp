#include "ModelMedium.h"
#include "../utils/ModelLoader.h"
#include <fstream>
#include <cmath>

namespace lyrebird {

ModelMedium::ModelMedium() {
    ringBuffer_.prepare(MediumConfig::BUFFER_LENGTH);
}

bool ModelMedium::loadModel(const std::string& jsonPath) {
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

            if (bufferLength != MediumConfig::BUFFER_LENGTH) {
                lastError_ = "Model buffer_length mismatch: expected " +
                             std::to_string(MediumConfig::BUFFER_LENGTH) +
                             ", got " + std::to_string(bufferLength);
                return false;
            }
            if (hiddenSize != MediumConfig::HIDDEN_SIZE) {
                lastError_ = "Model hidden_size mismatch: expected " +
                             std::to_string(MediumConfig::HIDDEN_SIZE) +
                             ", got " + std::to_string(hiddenSize);
                return false;
            }
            if (numLayers != MediumConfig::NUM_LAYERS) {
                lastError_ = "Model num_layers mismatch: expected " +
                             std::to_string(MediumConfig::NUM_LAYERS) +
                             ", got " + std::to_string(numLayers);
                return false;
            }
        }

        if (!modelJson.contains("layers")) {
            lastError_ = "Model JSON missing 'layers' array";
            return false;
        }

        auto& layers = modelJson["layers"];

        // Medium model has 3 dense layers (2 hidden + 1 output)
        if (layers.size() < 3) {
            lastError_ = "Expected 3 layers, got " + std::to_string(layers.size());
            return false;
        }

        // Load layer 0: input -> hidden
        {
            auto& layerWeights = layers[0]["weights"];
            auto& layerBias = layers[0]["bias"];
            auto& dense = model_.get<0>();

            std::vector<std::vector<float>> weights(MediumConfig::HIDDEN_SIZE,
                std::vector<float>(MediumConfig::INPUT_SIZE));
            std::vector<float> bias(MediumConfig::HIDDEN_SIZE);

            for (int i = 0; i < MediumConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < MediumConfig::INPUT_SIZE; ++j) {
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

            std::vector<std::vector<float>> weights(MediumConfig::HIDDEN_SIZE,
                std::vector<float>(MediumConfig::HIDDEN_SIZE));
            std::vector<float> bias(MediumConfig::HIDDEN_SIZE);

            for (int i = 0; i < MediumConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < MediumConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 2: hidden -> output (index 4 in RTNeural)
        {
            auto& layerWeights = layers[2]["weights"];
            auto& layerBias = layers[2]["bias"];
            auto& dense = model_.get<4>();

            std::vector<std::vector<float>> weights(MediumConfig::OUTPUT_SIZE,
                std::vector<float>(MediumConfig::HIDDEN_SIZE));
            std::vector<float> bias(MediumConfig::OUTPUT_SIZE);

            for (int i = 0; i < MediumConfig::OUTPUT_SIZE; ++i) {
                for (int j = 0; j < MediumConfig::HIDDEN_SIZE; ++j) {
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

float ModelMedium::processSample(float input) {
    ringBuffer_.push(input);
    sampleCount_++;

    if (sampleCount_ < MediumConfig::BUFFER_LENGTH || !modelLoaded_) {
        return input;
    }

    float modelOutput = model_.forward(ringBuffer_.data());

    int samplesAfterBufferFill = sampleCount_ - MediumConfig::BUFFER_LENGTH;
    if (samplesAfterBufferFill < MediumConfig::CROSSFADE_LENGTH) {
        float t = static_cast<float>(samplesAfterBufferFill) /
                  static_cast<float>(MediumConfig::CROSSFADE_LENGTH);
        return input * (1.0f - t) + modelOutput * t;
    }

    return modelOutput;
}

void ModelMedium::reset() {
    ringBuffer_.reset();
    model_.reset();
    sampleCount_ = 0;
}

void ModelMedium::warmup(int iterations) {
    if (!modelLoaded_) {
        return;
    }

    for (int i = 0; i < MediumConfig::INPUT_SIZE; ++i) {
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
