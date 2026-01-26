#include "RingBuffer.h"

// Template implementations are in the header file.
// This file exists for potential future non-template additions.

namespace lyrebird {

// Explicit template instantiations for common types
template class RingBuffer<float>;
template class RingBuffer<double>;

} // namespace lyrebird
