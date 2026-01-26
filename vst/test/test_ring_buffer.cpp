#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "dsp/RingBuffer.h"

using namespace lyrebird;
using Catch::Matchers::WithinAbs;

TEST_CASE("RingBuffer initialization", "[ring_buffer]") {
    RingBuffer<float> buffer;

    SECTION("default state") {
        REQUIRE_FALSE(buffer.isPrepared());
        REQUIRE(buffer.getLength() == 0);
    }

    SECTION("after prepare") {
        buffer.prepare(512);
        REQUIRE(buffer.isPrepared());
        REQUIRE(buffer.getLength() == 512);
    }
}

TEST_CASE("RingBuffer push and copy", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(4);

    SECTION("fill buffer sequentially") {
        buffer.push(1.0f);
        buffer.push(2.0f);
        buffer.push(3.0f);
        buffer.push(4.0f);

        float output[4];
        buffer.copyTo(output);

        // Should be in chronological order: oldest first
        REQUIRE_THAT(output[0], WithinAbs(1.0f, 0.0001f));
        REQUIRE_THAT(output[1], WithinAbs(2.0f, 0.0001f));
        REQUIRE_THAT(output[2], WithinAbs(3.0f, 0.0001f));
        REQUIRE_THAT(output[3], WithinAbs(4.0f, 0.0001f));
    }

    SECTION("wrap around") {
        buffer.push(1.0f);
        buffer.push(2.0f);
        buffer.push(3.0f);
        buffer.push(4.0f);
        buffer.push(5.0f);  // Overwrites 1.0
        buffer.push(6.0f);  // Overwrites 2.0

        float output[4];
        buffer.copyTo(output);

        // Should be: 3, 4, 5, 6 (oldest to newest)
        REQUIRE_THAT(output[0], WithinAbs(3.0f, 0.0001f));
        REQUIRE_THAT(output[1], WithinAbs(4.0f, 0.0001f));
        REQUIRE_THAT(output[2], WithinAbs(5.0f, 0.0001f));
        REQUIRE_THAT(output[3], WithinAbs(6.0f, 0.0001f));
    }
}

TEST_CASE("RingBuffer reset", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(4);

    buffer.push(1.0f);
    buffer.push(2.0f);
    buffer.push(3.0f);
    buffer.push(4.0f);

    buffer.reset();

    float output[4];
    buffer.copyTo(output);

    // Should be all zeros after reset
    for (int i = 0; i < 4; ++i) {
        REQUIRE_THAT(output[i], WithinAbs(0.0f, 0.0001f));
    }
}

TEST_CASE("RingBuffer various sizes", "[ring_buffer]") {
    SECTION("size 1") {
        RingBuffer<float> buffer;
        buffer.prepare(1);

        buffer.push(42.0f);

        float output[1];
        buffer.copyTo(output);
        REQUIRE_THAT(output[0], WithinAbs(42.0f, 0.0001f));

        buffer.push(99.0f);
        buffer.copyTo(output);
        REQUIRE_THAT(output[0], WithinAbs(99.0f, 0.0001f));
    }

    SECTION("size 512 (typical model buffer)") {
        RingBuffer<float> buffer;
        buffer.prepare(512);

        // Fill with incrementing values
        for (int i = 0; i < 512; ++i) {
            buffer.push(static_cast<float>(i));
        }

        float output[512];
        buffer.copyTo(output);

        for (int i = 0; i < 512; ++i) {
            REQUIRE_THAT(output[i], WithinAbs(static_cast<float>(i), 0.0001f));
        }

        // Push 100 more samples
        for (int i = 512; i < 612; ++i) {
            buffer.push(static_cast<float>(i));
        }

        buffer.copyTo(output);

        // Should now have samples 100-611
        for (int i = 0; i < 512; ++i) {
            REQUIRE_THAT(output[i], WithinAbs(static_cast<float>(i + 100), 0.0001f));
        }
    }
}

TEST_CASE("RingBuffer double precision", "[ring_buffer]") {
    RingBuffer<double> buffer;
    buffer.prepare(3);

    buffer.push(1.1);
    buffer.push(2.2);
    buffer.push(3.3);

    double output[3];
    buffer.copyTo(output);

    REQUIRE_THAT(output[0], WithinAbs(1.1, 0.0001));
    REQUIRE_THAT(output[1], WithinAbs(2.2, 0.0001));
    REQUIRE_THAT(output[2], WithinAbs(3.3, 0.0001));
}

TEST_CASE("RingBuffer partial fill", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(8);

    SECTION("read before full") {
        buffer.push(1.0f);
        buffer.push(2.0f);
        buffer.push(3.0f);

        float output[8];
        buffer.copyTo(output);

        // First 5 should be zeros (unfilled), last 3 are our values
        REQUIRE_THAT(output[0], WithinAbs(0.0f, 0.0001f));
        REQUIRE_THAT(output[4], WithinAbs(0.0f, 0.0001f));
        REQUIRE_THAT(output[5], WithinAbs(1.0f, 0.0001f));
        REQUIRE_THAT(output[6], WithinAbs(2.0f, 0.0001f));
        REQUIRE_THAT(output[7], WithinAbs(3.0f, 0.0001f));
    }
}

TEST_CASE("RingBuffer multiple wrap arounds", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(4);

    // Push 20 samples (5 full cycles)
    for (int i = 0; i < 20; ++i) {
        buffer.push(static_cast<float>(i));
    }

    float output[4];
    buffer.copyTo(output);

    // Should have last 4 samples: 16, 17, 18, 19
    REQUIRE_THAT(output[0], WithinAbs(16.0f, 0.0001f));
    REQUIRE_THAT(output[1], WithinAbs(17.0f, 0.0001f));
    REQUIRE_THAT(output[2], WithinAbs(18.0f, 0.0001f));
    REQUIRE_THAT(output[3], WithinAbs(19.0f, 0.0001f));
}

TEST_CASE("RingBuffer re-prepare", "[ring_buffer]") {
    RingBuffer<float> buffer;

    SECTION("prepare with different size clears data") {
        buffer.prepare(4);
        buffer.push(1.0f);
        buffer.push(2.0f);
        buffer.push(3.0f);
        buffer.push(4.0f);

        // Re-prepare with different size
        buffer.prepare(2);
        REQUIRE(buffer.getLength() == 2);

        float output[2];
        buffer.copyTo(output);

        // Should be zeros after re-prepare
        REQUIRE_THAT(output[0], WithinAbs(0.0f, 0.0001f));
        REQUIRE_THAT(output[1], WithinAbs(0.0f, 0.0001f));
    }

    SECTION("prepare with same size clears data") {
        buffer.prepare(4);
        buffer.push(1.0f);
        buffer.push(2.0f);

        buffer.prepare(4);

        float output[4];
        buffer.copyTo(output);

        for (int i = 0; i < 4; ++i) {
            REQUIRE_THAT(output[i], WithinAbs(0.0f, 0.0001f));
        }
    }
}

TEST_CASE("RingBuffer negative values", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(4);

    buffer.push(-1.0f);
    buffer.push(-0.5f);
    buffer.push(0.0f);
    buffer.push(0.5f);

    float output[4];
    buffer.copyTo(output);

    REQUIRE_THAT(output[0], WithinAbs(-1.0f, 0.0001f));
    REQUIRE_THAT(output[1], WithinAbs(-0.5f, 0.0001f));
    REQUIRE_THAT(output[2], WithinAbs(0.0f, 0.0001f));
    REQUIRE_THAT(output[3], WithinAbs(0.5f, 0.0001f));
}

TEST_CASE("RingBuffer audio-like signal", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(64);

    // Simulate a sine wave
    std::vector<float> sineWave(128);
    for (size_t i = 0; i < sineWave.size(); ++i) {
        sineWave[i] = std::sin(2.0f * 3.14159f * static_cast<float>(i) / 64.0f);
    }

    // Push all samples
    for (float sample : sineWave) {
        buffer.push(sample);
    }

    float output[64];
    buffer.copyTo(output);

    // Should have last 64 samples (second half of sine wave)
    for (int i = 0; i < 64; ++i) {
        REQUIRE_THAT(output[i], WithinAbs(sineWave[64 + i], 0.0001f));
    }
}

TEST_CASE("RingBuffer stress test", "[ring_buffer][stress]") {
    RingBuffer<float> buffer;
    buffer.prepare(1024);

    // Push many samples
    for (int i = 0; i < 100000; ++i) {
        buffer.push(static_cast<float>(i % 1000) / 1000.0f);
    }

    float output[1024];
    buffer.copyTo(output);

    // Verify last 1024 samples
    int startIdx = 100000 - 1024;
    for (int i = 0; i < 1024; ++i) {
        float expected = static_cast<float>((startIdx + i) % 1000) / 1000.0f;
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.0001f));
    }
}

TEST_CASE("RingBuffer copy consistency", "[ring_buffer]") {
    RingBuffer<float> buffer;
    buffer.prepare(8);

    for (int i = 0; i < 8; ++i) {
        buffer.push(static_cast<float>(i));
    }

    // Multiple copies should return same data
    float output1[8];
    float output2[8];

    buffer.copyTo(output1);
    buffer.copyTo(output2);

    for (int i = 0; i < 8; ++i) {
        REQUIRE_THAT(output1[i], WithinAbs(output2[i], 0.0001f));
    }

    // Copy doesn't modify state - push more and verify
    buffer.push(100.0f);

    float output3[8];
    buffer.copyTo(output3);

    // Should have shifted by one
    REQUIRE_THAT(output3[0], WithinAbs(1.0f, 0.0001f));
    REQUIRE_THAT(output3[7], WithinAbs(100.0f, 0.0001f));
}

TEST_CASE("RingBuffer integer type", "[ring_buffer]") {
    RingBuffer<int> buffer;
    buffer.prepare(4);

    buffer.push(10);
    buffer.push(20);
    buffer.push(30);
    buffer.push(40);

    int output[4];
    buffer.copyTo(output);

    REQUIRE(output[0] == 10);
    REQUIRE(output[1] == 20);
    REQUIRE(output[2] == 30);
    REQUIRE(output[3] == 40);
}
