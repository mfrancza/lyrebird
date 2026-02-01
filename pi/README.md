# Lyrebird Raspberry Pi Deployment

Real-time audio effects processing using trained Lyrebird neural FIR models on Raspberry Pi.

## Hardware Requirements

### Recommended Setup

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| **Raspberry Pi** | Pi 4 Model B (4GB) | Pi 5 for more headroom |
| **Audio I/O** | HiFiBerry DAC+ ADC | I2S, 192kHz, RCA jacks onboard |
| **Power Supply** | Official 5.1V/3A PSU | Clean power prevents audio noise |
| **Cooling** | Aluminum heatsink case | Prevents thermal throttling |
| **Storage** | 32GB A2 microSD | Fast boot, reliable |

### Audio I/O Options

| Option | Price | SNR | Notes |
|--------|-------|-----|-------|
| **HiFiBerry DAC+ ADC** | ~$65 | 100dB | Recommended, RCA jacks onboard |
| **HiFiBerry DAC+ ADC Pro** | ~$80 | 100dB | Software gain, balanced input |
| **Raspberry Pi Codec Zero** | ~$20 | 88dB | Budget, wire RCA to AUX pins |
| **USB Interface** | $50+ | varies | Higher latency, but has Hi-Z input |

## Quick Start

### 1. Install

```bash
cd lyrebird/pi
sudo ./install.sh
```

### 2. Configure Audio HAT

Edit `/boot/config.txt` and add your HAT's overlay:

```
dtoverlay=hifiberry-dacplusadc
dtparam=audio=off
```

Reboot after configuration.

### 3. Verify Audio

```bash
source ../venv/bin/activate
python realtime_processor.py --list-devices
```

### 4. Run with a Trained Model

```bash
# Using model size preset (recommended)
python realtime_processor.py \
    --model ../models/big_muff_small.pth \
    --size small

# Or with explicit parameters
python realtime_processor.py \
    --model ../models/big_muff_small.pth \
    --buffer-length 128 \
    --hidden-size 32 \
    --num-layers 2
```

## Project Structure

```
pi/
├── realtime_processor.py    # Main real-time audio processor
├── ring_buffer.py           # Circular buffer for sample history
├── audio_io.py              # Low-latency audio I/O
├── requirements-pi.txt      # Pi-specific dependencies
├── install.sh               # Installation script
│
├── config/
│   ├── config.txt.example   # /boot/config.txt additions
│   ├── cmdline.txt.example  # Kernel parameters
│   ├── asound.conf.example  # ALSA configuration
│   └── lyrebird.service     # systemd service
│
├── tools/
│   ├── export_onnx.py       # Export models to ONNX
│   └── benchmark_pi.py      # Pi-specific benchmarks
│
└── tests/
    └── test_latency.py      # Latency measurement
```

## Performance Tuning

### Model Size Presets

The following presets match the VST plugin's model variants:

| Name | buffer_length | hidden_size | layers | Params | Use Case |
|------|---------------|-------------|--------|--------|----------|
| **Small** | 128 | 32 | 2 | ~5K | Raspberry Pi real-time |
| **Medium** | 256 | 64 | 2 | ~21K | Laptop/Desktop |
| **Large** | 512 | 128 | 3 | ~99K | High-end systems |

Use the `--size` argument to select a preset:

```bash
python realtime_processor.py --model ../models/big_muff_small.pth --size small
python realtime_processor.py --model ../models/big_muff_medium.pth --size medium
```

**Computational cost comparison** (operations per sample):
- Small: ~5,184 MACs - suitable for Raspberry Pi 4
- Medium: ~20,544 MACs - 4x more compute
- Large: ~98,560 MACs - 19x more compute (desktop only)

### Kernel Optimization

For lowest latency, add to `/boot/cmdline.txt`:

```
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3
```

This reserves CPU cores 2-3 exclusively for audio processing.

### ONNX Runtime

ONNX Runtime can provide better performance than PyTorch on ARM:

```bash
# Export model to ONNX
python tools/export_onnx.py \
    --model ../models/big_muff_small.pth \
    --size small

# Run with ONNX
python realtime_processor.py \
    --model ../models/big_muff_small.pth \
    --size small \
    --use-onnx
```

## Latency

### Target Budget (44.1kHz)

| Component | Samples | Time |
|-----------|---------|------|
| ADC buffer | 128 | 2.9ms |
| Processing | 64 | ~1ms |
| DAC buffer | 128 | 2.9ms |
| **Total** | | **~7ms** |

### Measuring Latency

```bash
# Callback timing test
python tests/test_latency.py --test callback

# Round-trip latency (requires loopback cable)
python tests/test_latency.py --test loopback
```

## Benchmarking

Run benchmarks to verify your Pi can handle real-time processing:

```bash
# Benchmark Small model (recommended for Pi)
python tools/benchmark_pi.py \
    --model ../models/big_muff_small.pth \
    --size small

# Benchmark all sizes to compare performance
python tools/benchmark_pi.py --size small
python tools/benchmark_pi.py --size medium
python tools/benchmark_pi.py --size large

# Include thermal stress test
python tools/benchmark_pi.py --size small --stress-test 60
```

## Running as a Service

For auto-start on boot:

```bash
# Install service
sudo cp config/lyrebird.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable lyrebird

# Start/stop
sudo systemctl start lyrebird
sudo systemctl stop lyrebird

# View logs
journalctl -u lyrebird -f
```

## Troubleshooting

### No Audio Output

1. Check audio device is recognized: `aplay -l`
2. Verify HAT overlay in `/boot/config.txt`
3. Check ALSA mixer settings: `alsamixer`
4. Test with simple playback: `speaker-test -c 2`

### Buffer Underruns (Clicks/Pops)

1. Increase block size: `--block-size 256`
2. Use a smaller model (Tiny instead of Medium)
3. Enable CPU isolation in kernel parameters
4. Check CPU temperature isn't throttling

### High Latency

1. Use I2S HAT instead of USB audio
2. Reduce block size (if no underruns)
3. Enable kernel real-time parameters
4. Check for other processes using CPU

### Model Loading Issues

1. Use `--size` to match the model's training configuration (small/medium/large)
2. If using manual parameters, ensure they match: `--buffer-length`, `--hidden-size`, `--num-layers`
3. Verify model file exists and is readable
4. Check PyTorch version compatibility

## Development

### Testing on Desktop

The code can be tested on any Linux/macOS/Windows machine:

```bash
# Install desktop dependencies
pip install -r requirements-pi.txt

# Run with default audio device
python realtime_processor.py --model model.pth --list-devices
```

### Running Tests

```bash
# From project root
pytest pi/tests/ -v
```
