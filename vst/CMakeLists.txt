cmake_minimum_required(VERSION 3.21)

project(LyrebirdVST VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add JUCE
add_subdirectory(modules/JUCE)

# Add RTNeural with XSIMD backend for optimal performance
set(RTNEURAL_XSIMD ON CACHE BOOL "Use XSIMD backend for RTNeural")
add_subdirectory(modules/RTNeural)

# Add the plugin
add_subdirectory(plugin)

# Add tests (optional, requires Catch2)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
