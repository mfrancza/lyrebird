#include "ModelFactory.h"
#include "ModelSmall.h"
#include "ModelMedium.h"
#include "ModelLarge.h"

namespace lyrebird {

std::unique_ptr<IModel> ModelFactory::create(ModelSize size) {
    switch (size) {
        case ModelSize::Small:
            return std::make_unique<ModelSmall>();
        case ModelSize::Medium:
            return std::make_unique<ModelMedium>();
        case ModelSize::Large:
            return std::make_unique<ModelLarge>();
        default:
            return std::make_unique<ModelLarge>();
    }
}

} // namespace lyrebird
