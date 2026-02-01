#pragma once

#include "ModelInterface.h"
#include <memory>

namespace lyrebird {

/**
 * Factory for creating neural FIR models of different sizes.
 *
 * Since RTNeural uses compile-time templates, each model size is a separate
 * class with fixed dimensions. This factory provides a runtime interface
 * to create models by size enum.
 */
class ModelFactory {
public:
    /**
     * Create a new model instance of the specified size.
     *
     * @param size The model size preset
     * @return A unique pointer to the created model
     */
    static std::unique_ptr<IModel> create(ModelSize size);

    /**
     * Get the display name for a model size.
     *
     * @param size The model size enum value
     * @return Human-readable name
     */
    static const char* getSizeName(ModelSize size) {
        return getModelSizeName(size);
    }
};

} // namespace lyrebird
