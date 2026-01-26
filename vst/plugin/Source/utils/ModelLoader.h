#pragma once

// Include nlohmann/json for JSON parsing
// Using the json.hpp bundled with RTNeural
#include <json.hpp>

namespace lyrebird {

/**
 * Utility functions for loading neural network models.
 *
 * The ModelLoader provides helpers for reading and validating
 * model JSON files in the RTNeural format.
 */
namespace ModelLoader {

/**
 * Check if a model file exists and is valid JSON.
 *
 * @param path Path to the model file
 * @return true if the file exists and contains valid JSON
 */
inline bool isValidModelFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    try {
        nlohmann::json json;
        file >> json;
        return json.contains("config") && json.contains("layers");
    } catch (...) {
        return false;
    }
}

/**
 * Get model configuration from a JSON file.
 *
 * @param path Path to the model file
 * @param bufferLength Output: buffer length
 * @param hiddenSize Output: hidden layer size
 * @param numLayers Output: number of hidden layers
 * @return true if configuration was read successfully
 */
inline bool getModelConfig(const std::string& path,
                           int& bufferLength,
                           int& hiddenSize,
                           int& numLayers) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    try {
        nlohmann::json json;
        file >> json;

        if (!json.contains("config")) {
            return false;
        }

        auto& config = json["config"];
        bufferLength = config.value("buffer_length", 0);
        hiddenSize = config.value("hidden_size", 0);
        numLayers = config.value("num_layers", 0);

        return bufferLength > 0 && hiddenSize > 0 && numLayers > 0;
    } catch (...) {
        return false;
    }
}

} // namespace ModelLoader
} // namespace lyrebird
