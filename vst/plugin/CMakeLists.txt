# Lyrebird VST3 Plugin

juce_add_plugin(LyrebirdVST
    COMPANY_NAME "Lyrebird"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Lyre
    PLUGIN_CODE Lyfr
    FORMATS VST3 Standalone
    PRODUCT_NAME "Lyrebird"
)

target_sources(LyrebirdVST
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/dsp/RingBuffer.cpp
        Source/dsp/RingBuffer.h
        Source/dsp/NeuralFIRModel.cpp
        Source/dsp/NeuralFIRModel.h
        Source/utils/ModelLoader.h
)

target_compile_definitions(LyrebirdVST
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_include_directories(LyrebirdVST
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

target_link_libraries(LyrebirdVST
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        RTNeural
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)
