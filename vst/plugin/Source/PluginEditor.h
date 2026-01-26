#pragma once

#include "PluginProcessor.h"
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

/**
 * Lyrebird VST3 Plugin Editor (GUI)
 *
 * Provides controls for:
 * - Loading neural FIR models
 * - Dry/Wet mix
 * - Bypass toggle
 * - Model information display
 */
class LyrebirdAudioProcessorEditor : public juce::AudioProcessorEditor,
                                     public juce::Button::Listener,
                                     public juce::Slider::Listener {
public:
    explicit LyrebirdAudioProcessorEditor(LyrebirdAudioProcessor&);
    ~LyrebirdAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

    void buttonClicked(juce::Button* button) override;
    void sliderValueChanged(juce::Slider* slider) override;

private:
    LyrebirdAudioProcessor& audioProcessor;

    // GUI Components
    juce::TextButton loadModelButton;
    juce::ToggleButton bypassButton;
    juce::Slider dryWetSlider;
    juce::Label dryWetLabel;
    juce::Label modelInfoLabel;
    juce::Label titleLabel;

    // File chooser
    std::unique_ptr<juce::FileChooser> fileChooser;

    void updateModelInfo();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LyrebirdAudioProcessorEditor)
};
