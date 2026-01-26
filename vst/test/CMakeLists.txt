# Unit tests for Lyrebird DSP components
#
# Can be built standalone or as part of the main project

# Only set project if building standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.21)
    project(LyrebirdTests)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(LYREBIRD_STANDALONE_TESTS ON)
else()
    set(LYREBIRD_STANDALONE_TESTS OFF)
endif()

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# Only add RTNeural if building standalone (not already added by parent)
if(LYREBIRD_STANDALONE_TESTS AND NOT TARGET RTNeural)
    set(RTNEURAL_XSIMD ON CACHE BOOL "Use XSIMD backend")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../modules/RTNeural rtneural_build)
endif()

# Ring buffer tests (no RTNeural dependency)
add_executable(test_ring_buffer
    test_ring_buffer.cpp
)

target_include_directories(test_ring_buffer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source
)

target_link_libraries(test_ring_buffer
    PRIVATE
        Catch2::Catch2WithMain
)

# Neural model tests
add_executable(test_neural_model
    test_neural_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source/dsp/NeuralFIRModel.cpp
)

target_include_directories(test_neural_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/Source
        ${CMAKE_CURRENT_SOURCE_DIR}/../modules/RTNeural/modules/json
)

target_link_libraries(test_neural_model
    PRIVATE
        Catch2::Catch2WithMain
        RTNeural
)

# Register tests with CTest
include(CTest)
include(Catch)
catch_discover_tests(test_ring_buffer)
catch_discover_tests(test_neural_model)
