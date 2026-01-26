#include "PluginProcessor.h"
#include "PluginEditor.h"

LyrebirdAudioProcessor::LyrebirdAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      neuralModel_(std::make_unique<lyrebird::NeuralFIRModel>()),
      neuralModelRight_(std::make_unique<lyrebird::NeuralFIRModel>()) {
    // Create parameters
    addParameter(dryWetParam = new juce::AudioParameterFloat(
        "dryWet", "Dry/Wet", 0.0f, 1.0f, 1.0f));

    addParameter(bypassParam = new juce::AudioParameterBool(
        "bypass", "Bypass", false));
}

LyrebirdAudioProcessor::~LyrebirdAudioProcessor() = default;

const juce::String LyrebirdAudioProcessor::getName() const {
    return JucePlugin_Name;
}

bool LyrebirdAudioProcessor::acceptsMidi() const { return false; }
bool LyrebirdAudioProcessor::producesMidi() const { return false; }
bool LyrebirdAudioProcessor::isMidiEffect() const { return false; }

double LyrebirdAudioProcessor::getTailLengthSeconds() const {
    return 0.0;
}

int LyrebirdAudioProcessor::getNumPrograms() { return 1; }
int LyrebirdAudioProcessor::getCurrentProgram() { return 0; }
void LyrebirdAudioProcessor::setCurrentProgram(int) {}
const juce::String LyrebirdAudioProcessor::getProgramName(int) { return {}; }
void LyrebirdAudioProcessor::changeProgramName(int, const juce::String&) {}

void LyrebirdAudioProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
    currentSampleRate_ = sampleRate;

    // Reset model state
    if (neuralModel_) neuralModel_->reset();
    if (neuralModelRight_) neuralModelRight_->reset();

    // Report latency to DAW for automatic compensation
    setLatencySamples(neuralModel_ ? neuralModel_->getLatencySamples() : 0);
}

void LyrebirdAudioProcessor::releaseResources() {
    // Nothing to release
}

bool LyrebirdAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    // Support mono and stereo
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
        && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // Input and output must match
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void LyrebirdAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& /*midiMessages*/) {
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    // Clear any extra output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Check for pending model swap (atomic check, then swap at safe point)
    if (modelSwapPending_.load(std::memory_order_acquire)) {
        // Swap staging models with active models
        std::swap(neuralModel_, stagingModel_);
        std::swap(neuralModelRight_, stagingModelRight_);
        modelSwapPending_.store(false, std::memory_order_release);

        // Reset bypass gain to smoothly fade in new model
        bypassGain_ = 0.0f;
    }

    // No model loaded - pass through unchanged
    if (!neuralModel_ || !neuralModel_->isLoaded()) {
        return;
    }

    // Target bypass gain: 0 = bypassed (dry only), 1 = active (use model)
    float targetBypassGain = bypassParam->get() ? 0.0f : 1.0f;

    // Calculate fade rate based on current sample rate
    float fadeRate = 1.0f / (BYPASS_FADE_SAMPLES * static_cast<float>(currentSampleRate_ / 44100.0));

    float dryWet = dryWetParam->get();

    // Process each sample
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
        // Smooth bypass transition
        if (bypassGain_ < targetBypassGain) {
            bypassGain_ = std::min(bypassGain_ + fadeRate, targetBypassGain);
        } else if (bypassGain_ > targetBypassGain) {
            bypassGain_ = std::max(bypassGain_ - fadeRate, targetBypassGain);
        }

        // Process left channel (or mono)
        if (totalNumInputChannels >= 1) {
            float* channelData = buffer.getWritePointer(0);
            float drySignal = channelData[sample];
            float wetSignal = neuralModel_->processSample(drySignal);
            // Apply dry/wet mix, then apply bypass fade
            float processedSignal = drySignal * (1.0f - dryWet) + wetSignal * dryWet;
            channelData[sample] = drySignal * (1.0f - bypassGain_) + processedSignal * bypassGain_;
        }

        // Process right channel (separate model instance for stereo)
        if (totalNumInputChannels >= 2) {
            float* channelData = buffer.getWritePointer(1);
            float drySignal = channelData[sample];
            float wetSignal = neuralModelRight_->processSample(drySignal);
            // Apply dry/wet mix, then apply bypass fade
            float processedSignal = drySignal * (1.0f - dryWet) + wetSignal * dryWet;
            channelData[sample] = drySignal * (1.0f - bypassGain_) + processedSignal * bypassGain_;
        }
    }
}

bool LyrebirdAudioProcessor::loadModel(const juce::File& modelFile) {
    std::string path = modelFile.getFullPathName().toStdString();

    // Create new staging models
    auto newModel = std::make_unique<lyrebird::NeuralFIRModel>();
    auto newModelRight = std::make_unique<lyrebird::NeuralFIRModel>();

    // Load into staging models (UI thread)
    bool success = newModel->loadModel(path);
    if (success) {
        success = newModelRight->loadModel(path);
    }

    if (success) {
        // Reset the new models
        newModel->reset();
        newModelRight->reset();

        // Store staging models and signal swap
        stagingModel_ = std::move(newModel);
        stagingModelRight_ = std::move(newModelRight);

        loadedModelPath_ = modelFile.getFullPathName();

        // Signal audio thread to perform swap
        modelSwapPending_.store(true, std::memory_order_release);

        // Clear any previous error
        {
            const juce::ScopedLock sl(errorLock_);
            lastModelError_.clear();
            lastErrorPending_.store(false, std::memory_order_release);
        }
    } else {
        // Store error for UI retrieval
        const juce::ScopedLock sl(errorLock_);
        lastModelError_ = juce::String(newModel->getLastError());
        lastErrorPending_.store(true, std::memory_order_release);
    }

    return success;
}

juce::String LyrebirdAudioProcessor::getLastModelError() const {
    const juce::ScopedLock sl(errorLock_);
    return lastModelError_;
}

bool LyrebirdAudioProcessor::hasEditor() const {
    return true;
}

juce::AudioProcessorEditor* LyrebirdAudioProcessor::createEditor() {
    return new LyrebirdAudioProcessorEditor(*this);
}

void LyrebirdAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    // Save current state
    juce::MemoryOutputStream stream(destData, true);

    stream.writeFloat(*dryWetParam);
    stream.writeBool(*bypassParam);
    stream.writeString(loadedModelPath_);
}

void LyrebirdAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    // Restore saved state
    juce::MemoryInputStream stream(data, static_cast<size_t>(sizeInBytes), false);

    if (stream.getNumBytesRemaining() > 0) {
        *dryWetParam = stream.readFloat();
    }
    if (stream.getNumBytesRemaining() > 0) {
        *bypassParam = stream.readBool();
    }
    if (stream.getNumBytesRemaining() > 0) {
        juce::String modelPath = stream.readString();
        if (modelPath.isNotEmpty()) {
            juce::File modelFile(modelPath);
            if (modelFile.existsAsFile()) {
                loadModel(modelFile);
            }
        }
    }
}

// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new LyrebirdAudioProcessor();
}
