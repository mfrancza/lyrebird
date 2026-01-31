#include "PluginEditor.h"

LyrebirdAudioProcessorEditor::LyrebirdAudioProcessorEditor(LyrebirdAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p) {
    // Title
    titleLabel.setText("Lyrebird", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    // Load Model Button
    loadModelButton.setButtonText("Load Model...");
    loadModelButton.addListener(this);
    addAndMakeVisible(loadModelButton);

    // Bypass Button
    bypassButton.setButtonText("Bypass");
    bypassButton.addListener(this);
    bypassButton.setToggleState(audioProcessor.bypassParam->get(), juce::dontSendNotification);
    addAndMakeVisible(bypassButton);

    // Dry/Wet Slider
    dryWetSlider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    dryWetSlider.setRange(0.0, 1.0, 0.01);
    dryWetSlider.setValue(audioProcessor.dryWetParam->get());
    dryWetSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 20);
    dryWetSlider.addListener(this);
    addAndMakeVisible(dryWetSlider);

    // Dry/Wet Label
    dryWetLabel.setText("Dry/Wet", juce::dontSendNotification);
    dryWetLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(dryWetLabel);

    // Model Info Label
    modelInfoLabel.setJustificationType(juce::Justification::centred);
    modelInfoLabel.setFont(juce::Font(12.0f));
    addAndMakeVisible(modelInfoLabel);

    // Model Size ComboBox
    modelSizeCombo.addItem("Small (Pi)", 1);
    modelSizeCombo.addItem("Medium", 2);
    modelSizeCombo.addItem("Large", 3);
    modelSizeCombo.setSelectedId(static_cast<int>(audioProcessor.getModelSize()) + 1,
                                  juce::dontSendNotification);
    modelSizeCombo.addListener(this);
    addAndMakeVisible(modelSizeCombo);

    // Model Size Label
    modelSizeLabel.setText("Model Size", juce::dontSendNotification);
    modelSizeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(modelSizeLabel);

    updateModelInfo();

    setSize(300, 400);
}

LyrebirdAudioProcessorEditor::~LyrebirdAudioProcessorEditor() = default;

void LyrebirdAudioProcessorEditor::paint(juce::Graphics& g) {
    // Background gradient
    g.fillAll(juce::Colour(0xff2d2d2d));

    // Draw subtle border
    g.setColour(juce::Colour(0xff3d3d3d));
    g.drawRect(getLocalBounds(), 2);
}

void LyrebirdAudioProcessorEditor::resized() {
    auto area = getLocalBounds().reduced(20);

    // Title at top
    titleLabel.setBounds(area.removeFromTop(40));

    area.removeFromTop(10);

    // Model info
    modelInfoLabel.setBounds(area.removeFromTop(40));

    area.removeFromTop(10);

    // Model size selector
    modelSizeLabel.setBounds(area.removeFromTop(20));
    modelSizeCombo.setBounds(area.removeFromTop(25).reduced(40, 0));

    area.removeFromTop(10);

    // Load button
    loadModelButton.setBounds(area.removeFromTop(30).reduced(40, 0));

    area.removeFromTop(15);

    // Dry/Wet knob
    auto knobArea = area.removeFromTop(120);
    dryWetSlider.setBounds(knobArea.reduced(60, 0));

    // Dry/Wet label
    dryWetLabel.setBounds(area.removeFromTop(20));

    area.removeFromTop(15);

    // Bypass button at bottom
    bypassButton.setBounds(area.removeFromTop(30).reduced(80, 0));
}

void LyrebirdAudioProcessorEditor::buttonClicked(juce::Button* button) {
    if (button == &loadModelButton) {
        fileChooser = std::make_unique<juce::FileChooser>(
            "Select Model File",
            juce::File{},
            "*.json");

        auto flags = juce::FileBrowserComponent::openMode |
                     juce::FileBrowserComponent::canSelectFiles;

        fileChooser->launchAsync(flags, [this](const juce::FileChooser& fc) {
            auto file = fc.getResult();
            if (file.existsAsFile()) {
                if (audioProcessor.loadModel(file)) {
                    updateModelInfo();
                } else {
                    juce::String errorMsg = audioProcessor.getLastModelError();
                    if (errorMsg.isEmpty()) {
                        errorMsg = juce::String::formatted(
                            "Failed to load model. Expected config: buffer_length=%d, hidden_size=%d, num_layers=%d",
                            audioProcessor.getModelBufferLength(),
                            audioProcessor.getModelHiddenSize(),
                            audioProcessor.getModelNumLayers());
                    }
                    juce::AlertWindow::showMessageBoxAsync(
                        juce::AlertWindow::WarningIcon,
                        "Model Load Error",
                        errorMsg);
                }
            }
        });
    } else if (button == &bypassButton) {
        *audioProcessor.bypassParam = bypassButton.getToggleState();
    }
}

void LyrebirdAudioProcessorEditor::comboBoxChanged(juce::ComboBox* comboBox) {
    if (comboBox == &modelSizeCombo) {
        int selectedId = modelSizeCombo.getSelectedId();
        if (selectedId > 0) {
            auto newSize = static_cast<lyrebird::ModelSize>(selectedId - 1);
            if (newSize != audioProcessor.getModelSize()) {
                audioProcessor.setModelSize(newSize);
                updateModelInfo();
            }
        }
    }
}

void LyrebirdAudioProcessorEditor::sliderValueChanged(juce::Slider* slider) {
    if (slider == &dryWetSlider) {
        *audioProcessor.dryWetParam = static_cast<float>(dryWetSlider.getValue());
    }
}

void LyrebirdAudioProcessorEditor::updateModelInfo() {
    if (audioProcessor.isModelLoaded()) {
        juce::File modelFile(audioProcessor.getLoadedModelPath());
        juce::String info = "Model: " + modelFile.getFileNameWithoutExtension() + "\n";
        info += juce::String::formatted("Buffer: %d | Hidden: %d | Layers: %d",
            audioProcessor.getModelBufferLength(),
            audioProcessor.getModelHiddenSize(),
            audioProcessor.getModelNumLayers());
        modelInfoLabel.setText(info, juce::dontSendNotification);
        modelInfoLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    } else {
        juce::String info = "No model loaded\n";
        info += juce::String::formatted("Size: %s (Buffer: %d)",
            lyrebird::getModelSizeName(audioProcessor.getModelSize()),
            audioProcessor.getModelBufferLength());
        modelInfoLabel.setText(info, juce::dontSendNotification);
        modelInfoLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    }
}
