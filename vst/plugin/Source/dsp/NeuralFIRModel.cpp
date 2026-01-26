#include "NeuralFIRModel.h"
#include "../utils/ModelLoader.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

namespace lyrebird {

NeuralFIRModel::NeuralFIRModel() {
    ringBuffer_.prepare(ModelConfig::BUFFER_LENGTH);
    inferenceBuffer_.fill(0.0f);
}

NeuralFIRModel::~NeuralFIRModel() = default;

bool NeuralFIRModel::loadModel(const std::string& jsonPath) {
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

            if (bufferLength != ModelConfig::BUFFER_LENGTH) {
                lastError_ = "Model buffer_length mismatch: expected " +
                             std::to_string(ModelConfig::BUFFER_LENGTH) +
                             ", got " + std::to_string(bufferLength);
                return false;
            }
            if (hiddenSize != ModelConfig::HIDDEN_SIZE) {
                lastError_ = "Model hidden_size mismatch: expected " +
                             std::to_string(ModelConfig::HIDDEN_SIZE) +
                             ", got " + std::to_string(hiddenSize);
                return false;
            }
            if (numLayers != ModelConfig::NUM_LAYERS) {
                lastError_ = "Model num_layers mismatch: expected " +
                             std::to_string(ModelConfig::NUM_LAYERS) +
                             ", got " + std::to_string(numLayers);
                return false;
            }
        }

        // Load weights into RTNeural model
        // RTNeural expects weights in a specific format within the JSON
        if (!modelJson.contains("layers")) {
            lastError_ = "Model JSON missing 'layers' array";
            return false;
        }

        auto& layers = modelJson["layers"];

        // Load each layer's weights
        // Our model has 4 dense layers (with ReLU after first 3)
        // Layer 0: input -> hidden1
        // Layer 1: hidden1 -> hidden2
        // Layer 2: hidden2 -> hidden3
        // Layer 3: hidden3 -> output

        if (layers.size() < 4) {
            lastError_ = "Expected 4 layers, got " + std::to_string(layers.size());
            return false;
        }

        // Load layer 0 (first dense layer): input -> hidden
        {
            auto& layerWeights = layers[0]["weights"];
            auto& layerBias = layers[0]["bias"];
            auto& dense = model_.get<0>();

            // RTNeural expects weights as [out_size][in_size] array
            std::vector<std::vector<float>> weights(ModelConfig::HIDDEN_SIZE,
                std::vector<float>(ModelConfig::INPUT_SIZE));
            std::vector<float> bias(ModelConfig::HIDDEN_SIZE);

            for (int i = 0; i < ModelConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < ModelConfig::INPUT_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 1 (second dense layer) - index 2 in RTNeural (after ReLU)
        {
            auto& layerWeights = layers[1]["weights"];
            auto& layerBias = layers[1]["bias"];
            auto& dense = model_.get<2>();

            std::vector<std::vector<float>> weights(ModelConfig::HIDDEN_SIZE,
                std::vector<float>(ModelConfig::HIDDEN_SIZE));
            std::vector<float> bias(ModelConfig::HIDDEN_SIZE);

            for (int i = 0; i < ModelConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < ModelConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 2 (third dense layer) - index 4 in RTNeural
        {
            auto& layerWeights = layers[2]["weights"];
            auto& layerBias = layers[2]["bias"];
            auto& dense = model_.get<4>();

            std::vector<std::vector<float>> weights(ModelConfig::HIDDEN_SIZE,
                std::vector<float>(ModelConfig::HIDDEN_SIZE));
            std::vector<float> bias(ModelConfig::HIDDEN_SIZE);

            for (int i = 0; i < ModelConfig::HIDDEN_SIZE; ++i) {
                for (int j = 0; j < ModelConfig::HIDDEN_SIZE; ++j) {
                    weights[i][j] = layerWeights[i][j].get<float>();
                }
                bias[i] = layerBias[i].get<float>();
            }
            dense.setWeights(weights);
            dense.setBias(bias.data());
        }

        // Load layer 3 (output dense layer) - index 6 in RTNeural
        {
            auto& layerWeights = layers[3]["weights"];
            auto& layerBias = layers[3]["bias"];
            auto& dense = model_.get<6>();

            std::vector<std::vector<float>> weights(ModelConfig::OUTPUT_SIZE,
                std::vector<float>(ModelConfig::HIDDEN_SIZE));
            std::vector<float> bias(ModelConfig::OUTPUT_SIZE);

            for (int i = 0; i < ModelConfig::OUTPUT_SIZE; ++i) {
                for (int j = 0; j < ModelConfig::HIDDEN_SIZE; ++j) {
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

float NeuralFIRModel::processSample(float input) {
    // Push sample to ring buffer
    ringBuffer_.push(input);
    sampleCount_++;

    // During initial buffer fill, pass through input
    if (sampleCount_ < ModelConfig::BUFFER_LENGTH || !modelLoaded_) {
        return input;
    }

    // Copy ring buffer to inference buffer (chronological order)
    ringBuffer_.copyTo(inferenceBuffer_.data());

    // Run inference
    float modelOutput = model_.forward(inferenceBuffer_.data());

    // Crossfade from passthrough to model output during transition period
    int samplesAfterBufferFill = sampleCount_ - ModelConfig::BUFFER_LENGTH;
    if (samplesAfterBufferFill < ModelConfig::CROSSFADE_LENGTH) {
        float t = static_cast<float>(samplesAfterBufferFill) /
                  static_cast<float>(ModelConfig::CROSSFADE_LENGTH);
        return input * (1.0f - t) + modelOutput * t;
    }

    return modelOutput;
}

void NeuralFIRModel::reset() {
    ringBuffer_.reset();
    model_.reset();
    sampleCount_ = 0;
}

void NeuralFIRModel::warmup(int iterations) {
    if (!modelLoaded_) {
        return;
    }

    // Fill inference buffer with realistic audio-like values
    for (int i = 0; i < ModelConfig::INPUT_SIZE; ++i) {
        inferenceBuffer_[i] = 0.1f * std::sin(static_cast<float>(i) * 0.1f);
    }

    // Run dummy inferences to warm CPU caches
    volatile float dummy = 0.0f;  // volatile prevents optimization
    for (int i = 0; i < iterations; ++i) {
        dummy = model_.forward(inferenceBuffer_.data());
    }

    // Reset model state after warmup
    model_.reset();
}

} // namespace lyrebird
