#include "PluginProcessor.h"
#include "PluginEditor.h"

LyrebirdAudioProcessor::LyrebirdAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      currentModelSize_(lyrebird::ModelSize::Large),
      neuralModel_(lyrebird::ModelFactory::create(lyrebird::ModelSize::Large)),
      neuralModelRight_(lyrebird::ModelFactory::create(lyrebird::ModelSize::Large)) {
    // Create parameters
    addParameter(dryWetParam = new juce::AudioParameterFloat(
        "dryWet", "Dry/Wet", 0.0f, 1.0f, 1.0f));

    addParameter(bypassParam = new juce::AudioParameterBool(
        "bypass", "Bypass", false));

    // Model size parameter (Small=0, Medium=1, Large=2)
    addParameter(modelSizeParam = new juce::AudioParameterChoice(
        "modelSize", "Model Size",
        juce::StringArray{"Small", "Medium", "Large"},
        2));  // Default to Large
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

    // Warm up models to pre-populate CPU caches and avoid choppy audio at startup
    if (neuralModel_) neuralModel_->warmup(100);
    if (neuralModelRight_) neuralModelRight_->warmup(100);

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

    // Create new staging models with current size
    auto newModel = lyrebird::ModelFactory::create(currentModelSize_);
    auto newModelRight = lyrebird::ModelFactory::create(currentModelSize_);

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

void LyrebirdAudioProcessor::setModelSize(lyrebird::ModelSize size) {
    if (size == currentModelSize_) {
        return;
    }

    // Create new models with the specified size
    auto newModel = lyrebird::ModelFactory::create(size);
    auto newModelRight = lyrebird::ModelFactory::create(size);

    // If we have a loaded model path, try to reload with new size
    // Note: The loaded model may not match the new size - that's expected
    // User will need to load a compatible model

    // Store in staging for thread-safe swap
    stagingModel_ = std::move(newModel);
    stagingModelRight_ = std::move(newModelRight);

    currentModelSize_ = size;

    // Update the parameter if it doesn't match
    int paramIndex = static_cast<int>(size);
    if (modelSizeParam->getIndex() != paramIndex) {
        *modelSizeParam = paramIndex;
    }

    // Signal audio thread to perform swap
    modelSwapPending_.store(true, std::memory_order_release);

    // Clear loaded model path since we're switching sizes
    loadedModelPath_.clear();

    // Clear any previous error
    {
        const juce::ScopedLock sl(errorLock_);
        lastModelError_.clear();
        lastErrorPending_.store(false, std::memory_order_release);
    }
}

int LyrebirdAudioProcessor::getModelBufferLength() const {
    return lyrebird::getBufferLengthForSize(currentModelSize_);
}

int LyrebirdAudioProcessor::getModelHiddenSize() const {
    return lyrebird::getHiddenSizeForSize(currentModelSize_);
}

int LyrebirdAudioProcessor::getModelNumLayers() const {
    return lyrebird::getNumLayersForSize(currentModelSize_);
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
    stream.writeInt(static_cast<int>(currentModelSize_));
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

    juce::String modelPath;
    if (stream.getNumBytesRemaining() > 0) {
        modelPath = stream.readString();
    }

    // Restore model size (defaults to Large for backwards compatibility)
    if (stream.getNumBytesRemaining() >= sizeof(int)) {
        int sizeInt = stream.readInt();
        if (sizeInt >= 0 && sizeInt <= 2) {
            setModelSize(static_cast<lyrebird::ModelSize>(sizeInt));
        }
    }

    // Load model after setting size
    if (modelPath.isNotEmpty()) {
        juce::File modelFile(modelPath);
        if (modelFile.existsAsFile()) {
            loadModel(modelFile);
        }
    }
}

// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new LyrebirdAudioProcessor();
}
