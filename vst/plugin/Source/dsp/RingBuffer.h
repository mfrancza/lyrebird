#pragma once

#include <vector>
#include <cstddef>
#include <cassert>
#include <cstring>

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

        // Use memcpy for better performance (can be vectorized)
        std::memcpy(output, buffer_.data() + writeIndex_, firstPartSize * sizeof(T));
        std::memcpy(output + firstPartSize, buffer_.data(), writeIndex_ * sizeof(T));
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

/**
 * A zero-copy ring buffer that maintains a contiguous view of the data.
 *
 * Uses a double-sized internal buffer so that the last N samples are always
 * available as a contiguous block without copying. This is ideal for neural
 * network inference where we need fast access to the buffer contents.
 */
template <typename T>
class ZeroCopyRingBuffer {
public:
    ZeroCopyRingBuffer() = default;

    /**
     * Prepare the buffer for a given size.
     *
     * @param bufferLength The number of samples to store
     */
    void prepare(size_t bufferLength) {
        bufferLength_ = bufferLength;
        // Allocate double size so we always have a contiguous view
        buffer_.resize(bufferLength * 2);
        reset();
    }

    /**
     * Push a new sample into the buffer.
     *
     * @param sample The sample to push
     */
    void push(T sample) {
        // Write to both halves to maintain contiguous view
        buffer_[writeIndex_] = sample;
        buffer_[writeIndex_ + bufferLength_] = sample;
        writeIndex_ = (writeIndex_ + 1) % bufferLength_;
    }

    /**
     * Get a pointer to the contiguous buffer data (oldest to newest).
     * This is a zero-copy operation.
     *
     * @return Pointer to bufferLength contiguous samples
     */
    const T* data() const {
        // writeIndex_ points to oldest sample, and we have bufferLength
        // contiguous samples starting there (thanks to double-buffer)
        return buffer_.data() + writeIndex_;
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

private:
    std::vector<T> buffer_;
    size_t bufferLength_ = 0;
    size_t writeIndex_ = 0;
};

} // namespace lyrebird
