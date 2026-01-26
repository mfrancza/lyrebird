#pragma once

#include <vector>
#include <cstddef>
#include <cassert>

namespace lyrebird {

/**
 * A fixed-size ring buffer for audio sample history.
 *
 * Used to maintain the last N samples needed for neural FIR inference.
 * Samples are pushed one at a time and can be copied out in chronological order.
 */
template <typename T>
class RingBuffer {
public:
    RingBuffer() = default;

    /**
     * Prepare the buffer for a given size.
     * This clears any existing data and allocates storage.
     *
     * @param bufferLength The number of samples to store
     */
    void prepare(size_t bufferLength) {
        buffer_.resize(bufferLength);
        bufferLength_ = bufferLength;
        reset();
    }

    /**
     * Push a new sample into the buffer.
     * The oldest sample is overwritten.
     *
     * @param sample The sample to push
     */
    void push(T sample) {
        buffer_[writeIndex_] = sample;
        writeIndex_ = (writeIndex_ + 1) % bufferLength_;
    }

    /**
     * Copy all samples to an output array in chronological order.
     * The oldest sample comes first, newest last.
     *
     * @param output Pointer to output array (must have bufferLength elements)
     */
    void copyTo(T* output) const {
        // writeIndex_ points to the oldest sample (next to be overwritten)
        // Copy from writeIndex_ to end, then from 0 to writeIndex_
        size_t firstPartSize = bufferLength_ - writeIndex_;

        for (size_t i = 0; i < firstPartSize; ++i) {
            output[i] = buffer_[writeIndex_ + i];
        }

        for (size_t i = 0; i < writeIndex_; ++i) {
            output[firstPartSize + i] = buffer_[i];
        }
    }

    /**
     * Reset the buffer to all zeros.
     */
    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), T{0});
        writeIndex_ = 0;
    }

    /**
     * Get the buffer length.
     *
     * @return The number of samples the buffer holds
     */
    size_t getLength() const {
        return bufferLength_;
    }

    /**
     * Check if the buffer has been prepared.
     *
     * @return true if prepare() has been called
     */
    bool isPrepared() const {
        return bufferLength_ > 0;
    }

private:
    std::vector<T> buffer_;
    size_t bufferLength_ = 0;
    size_t writeIndex_ = 0;
};

} // namespace lyrebird
