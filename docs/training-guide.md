# Lyrebird Training Guide

This guide walks you through installing Lyrebird and training your first neural audio effect model. No programming experience required.

## Prerequisites

You need two things installed on your computer before you start:

### Python 3.10 or newer

Download and install Python from [python.org](https://www.python.org/downloads/). During installation on Windows, make sure to check **"Add Python to PATH"**.

To verify your installation, open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```
python --version
```

You should see `Python 3.10.x` or higher.

### FFmpeg and libsndfile

Lyrebird depends on system libraries for reading and writing audio files.

**macOS** (using [Homebrew](https://brew.sh/)):
```
brew install ffmpeg libsndfile
```

**Ubuntu / Debian Linux:**
```
sudo apt install ffmpeg libsndfile1
```

**Windows** (using [Chocolatey](https://chocolatey.org/)):
```
choco install ffmpeg
```

On Windows, `libsndfile` is bundled with the Python `soundfile` package automatically.

## Installation

Install Lyrebird from PyPI:

```
pip install lyrebird-audio
```

This installs the `lyrebird-train` and `lyrebird-export` command-line tools along with all Python dependencies (PyTorch, torchaudio, etc.).

## Preparing your audio

To train a model, you need two WAV files recorded from the same audio source:

- **Dry file** — the clean, unprocessed audio (e.g., a guitar plugged directly into your audio interface)
- **Wet file** — the same audio after passing through the effect you want to clone (e.g., the guitar through a distortion pedal)

### Requirements

- Both files must be **WAV format**
- Both files must have the **same sample rate** (e.g., both at 44100 Hz)
- Both files must be the **same length**
- The audio must be **time-aligned** — each sample in the dry file corresponds to the same moment in the wet file

### Recording tips

1. **Use a splitter or re-amping setup.** Record the dry signal directly into your audio interface while simultaneously recording the wet signal through your effect pedal or amp.
2. **Aim for 30 seconds to 2 minutes of audio.** Shorter clips train faster but may not capture the full range of the effect. Longer clips give the model more to learn from.
3. **Play a variety of dynamics and notes.** Include quiet passages, loud passages, single notes, and chords so the model learns how the effect responds to different inputs.
4. **Avoid silence and noise.** Trim dead air from the beginning and end of your recordings.

## Training a model

Run the `lyrebird-train` command with your dry and wet audio files:

```
lyrebird-train --input dry.wav --output wet.wav --name my_effect
```

Arguments:
- `--input` (`-i`) — path to your dry WAV file
- `--output` (`-o`) — path to your wet WAV file
- `--name` (`-n`) — a name for your effect (used in output filenames)

By default, this trains **all three model sizes** (small, medium, large) for 5 epochs each. To train only specific sizes:

```
lyrebird-train --input dry.wav --output wet.wav --name my_effect --sizes medium
```

```
lyrebird-train --input dry.wav --output wet.wav --name my_effect --sizes small medium
```

### Additional options

| Option | Default | Description |
|---|---|---|
| `--sizes` (`-s`) | `small medium large` | Which model sizes to train |
| `--epochs` (`-e`) | `5` | Number of training passes over the data |
| `--batch-size` (`-b`) | `32` | Samples per training batch |
| `--lr` | `0.001` | Learning rate |
| `--output-dir` (`-d`) | `models` | Directory to save trained models |
| `--no-export` | off | Skip exporting to RTNeural JSON format |

## Model sizes

Lyrebird includes three size presets. Larger models have more parameters and a longer receptive field (buffer length), which can capture more complex effects but require more processing power.

| Size | Buffer length | Hidden size | Layers | Parameters | Intended use |
|---|---|---|---|---|---|
| **small** | 128 samples | 32 | 2 | ~5K | Raspberry Pi, embedded devices |
| **medium** | 256 samples | 64 | 2 | ~21K | Laptop, desktop |
| **large** | 512 samples | 128 | 3 | ~99K | High-end desktop, GPU |

Start with **medium** if you're unsure — it's a good balance of quality and performance for most computers.

## Output files

After training, you'll find two files per model size in the `models/` directory (or whichever directory you specified with `--output-dir`):

- **`.pth` file** — the PyTorch model checkpoint. Used for further training or inference in Python.
- **`.json` file** — the RTNeural export. **This is the file you load into the Lyrebird VST plugin.**

For example, training with `--name my_effect --sizes medium` produces:
```
models/my_effect_medium.pth
models/my_effect_medium.json
```

### Using with the VST plugin

1. Open the Lyrebird VST plugin in your DAW
2. Select the matching model size from the dropdown (e.g., "medium")
3. Click "Load Model..." and select the `.json` file

The model size in the plugin must match the size you trained — a "medium" `.json` file only works with the "medium" setting in the plugin.

## Tips and troubleshooting

**Training is slow.** If you have an NVIDIA GPU, Lyrebird will use it automatically via CUDA. Without a GPU, training runs on CPU which is significantly slower. Try reducing the audio length or training only the `small` size to start.

**"Error: Input file not found"** — Check that the file path is correct. If your filename contains spaces, wrap it in quotes:
```
lyrebird-train --input "my dry file.wav" --output "my wet file.wav" --name my_effect
```

**High loss values.** Loss measures how different the model's output is from the target. If loss isn't decreasing, try:
- Increasing epochs with `--epochs 10` or `--epochs 20`
- Checking that your dry and wet files are properly time-aligned
- Making sure your audio contains enough variety (not just silence or a single sustained note)

**The model doesn't sound right.** Some effects are harder to clone than others. Lyrebird works best with **static, memoryless effects** like distortion, overdrive, and EQ. Effects with feedback or decay — like reverb, delay, and chorus — cannot be accurately modeled with this architecture.

**Out of memory errors.** Reduce the batch size with `--batch-size 16` or `--batch-size 8`. If still running out of memory, try training only the `small` size.
