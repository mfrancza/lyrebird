#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "dsp/ModelInterface.h"
#include "dsp/ModelFactory.h"
#include <atomic>

/**
 * Lyrebird VST3 Audio Processor
 *
 * This processor runs trained neural FIR models in real-time.
 * It processes audio sample-by-sample through the neural network.
 */
class LyrebirdAudioProcessor : public juce::AudioProcessor {
public:
    LyrebirdAudioProcessor();
    ~LyrebirdAudioProcessor() override;

    // AudioProcessor interface
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Model loading
    bool loadModel(const juce::File& modelFile);
    bool isModelLoaded() const { return neuralModel_ && neuralModel_->isLoaded(); }
    juce::String getLoadedModelPath() const { return loadedModelPath_; }
    juce::String getLastModelError() const;
    bool hasModelSwapPending() const { return modelSwapPending_.load(); }

    // Model size management
    lyrebird::ModelSize getModelSize() const { return currentModelSize_; }
    void setModelSize(lyrebird::ModelSize size);
    int getModelBufferLength() const;
    int getModelHiddenSize() const;
    int getModelNumLayers() const;

    // Parameters
    juce::AudioParameterFloat* dryWetParam = nullptr;
    juce::AudioParameterBool* bypassParam = nullptr;
    juce::AudioParameterChoice* modelSizeParam = nullptr;

private:
    // Active models used by audio thread (using IModel interface)
    std::unique_ptr<lyrebird::IModel> neuralModel_;
    std::unique_ptr<lyrebird::IModel> neuralModelRight_;

    // Staging models for thread-safe loading (loaded by UI thread)
    std::unique_ptr<lyrebird::IModel> stagingModel_;
    std::unique_ptr<lyrebird::IModel> stagingModelRight_;

    // Current model size
    lyrebird::ModelSize currentModelSize_ = lyrebird::ModelSize::Large;

    // Atomic flag to signal audio thread that new model is ready
    std::atomic<bool> modelSwapPending_{false};

    // Last error from model loading (protected by atomic flag)
    std::atomic<bool> lastErrorPending_{false};
    juce::String lastModelError_;
    juce::CriticalSection errorLock_;

    juce::String loadedModelPath_;

    double currentSampleRate_ = 44100.0;

    // Bypass smoothing to prevent clicks
    float bypassGain_ = 0.0f;  // 0 = bypassed (dry), 1 = active (wet)
    static constexpr float BYPASS_FADE_SAMPLES = 441.0f;  // ~10ms at 44.1kHz

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LyrebirdAudioProcessor)
};
