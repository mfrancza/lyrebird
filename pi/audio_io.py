"""
Low-latency audio I/O abstraction using sounddevice.

Provides a unified interface for audio input/output with support for
different audio backends and devices.
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
import threading
import queue
import time


@dataclass
class AudioConfig:
    """Configuration for audio I/O."""

    sample_rate: int = 44100
    channels: int = 1
    block_size: int = 128
    dtype: np.dtype = np.float32
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    latency: str = "low"  # 'low', 'high', or specific value in seconds


def list_devices() -> List[Dict[str, Any]]:
    """
    List available audio devices.

    Returns:
        List of device info dictionaries
    """
    devices = sd.query_devices()
    return [
        {
            "index": i,
            "name": d["name"],
            "inputs": d["max_input_channels"],
            "outputs": d["max_output_channels"],
            "default_sr": d["default_samplerate"],
            "is_default_input": i == sd.default.device[0],
            "is_default_output": i == sd.default.device[1],
        }
        for i, d in enumerate(devices)
    ]


def print_devices() -> None:
    """Print available audio devices in a readable format."""
    devices = list_devices()
    print("Available audio devices:")
    print("-" * 70)
    for d in devices:
        flags = []
        if d["is_default_input"]:
            flags.append("*IN")
        if d["is_default_output"]:
            flags.append("*OUT")
        flag_str = " ".join(flags)
        print(
            f"[{d['index']:2d}] {d['name']:<40} "
            f"in:{d['inputs']} out:{d['outputs']} "
            f"sr:{int(d['default_sr'])} {flag_str}"
        )
    print("-" * 70)


class AudioIO:
    """
    Low-latency audio I/O handler.

    Provides callback-based audio processing with minimal latency.

    Args:
        config: Audio configuration
        callback: Processing callback, signature:
                 callback(input_data: np.ndarray) -> np.ndarray
                 Input shape: (block_size, channels)
                 Output shape: (block_size, channels)
    """

    def __init__(
        self,
        config: AudioConfig,
        callback: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.config = config
        self._callback = callback
        self._stream: Optional[sd.Stream] = None
        self._running = False
        self._error: Optional[Exception] = None

        # Statistics
        self._stats_lock = threading.Lock()
        self._underruns = 0
        self._overruns = 0
        self._total_blocks = 0
        self._max_callback_time = 0.0
        self._callback_times: List[float] = []

    def set_callback(self, callback: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set or update the processing callback.

        Warning: Do not call this while the audio stream is running.
        Stop the stream first, update the callback, then restart.
        """
        self._callback = callback

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal sounddevice callback."""
        start_time = time.perf_counter()

        # Track errors
        if status.input_underflow:
            with self._stats_lock:
                self._underruns += 1
        if status.input_overflow:
            with self._stats_lock:
                self._overruns += 1

        try:
            if self._callback is not None:
                # Process audio
                result = self._callback(indata.copy())
                if result is not None:
                    outdata[:] = result
                else:
                    outdata[:] = indata  # Passthrough
            else:
                outdata[:] = indata  # Passthrough when no callback
        except Exception as e:
            self._error = e
            outdata.fill(0)  # Silence on error

        # Track timing
        elapsed = time.perf_counter() - start_time
        with self._stats_lock:
            self._total_blocks += 1
            self._max_callback_time = max(self._max_callback_time, elapsed)
            # Keep last 1000 timing samples
            self._callback_times.append(elapsed)
            if len(self._callback_times) > 1000:
                self._callback_times.pop(0)

    def start(self) -> None:
        """Start audio stream."""
        if self._running:
            return

        self._stream = sd.Stream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            device=(self.config.input_device, self.config.output_device),
            channels=self.config.channels,
            dtype=self.config.dtype,
            latency=self.config.latency,
            callback=self._audio_callback,
        )

        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Stop audio stream."""
        if not self._running:
            return

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._stats_lock:
            avg_time = (
                sum(self._callback_times) / len(self._callback_times)
                if self._callback_times
                else 0
            )
            return {
                "underruns": self._underruns,
                "overruns": self._overruns,
                "total_blocks": self._total_blocks,
                "max_callback_ms": self._max_callback_time * 1000,
                "avg_callback_ms": avg_time * 1000,
                "budget_ms": self.config.block_size / self.config.sample_rate * 1000,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._stats_lock:
            self._underruns = 0
            self._overruns = 0
            self._total_blocks = 0
            self._max_callback_time = 0.0
            self._callback_times.clear()

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last error from callback."""
        return self._error

    def clear_error(self) -> None:
        """Clear the last error."""
        self._error = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


class AudioPassthrough:
    """
    Simple audio passthrough for testing latency.

    Minimal processing to measure system latency.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self._audio_io = AudioIO(config, self._passthrough)

    def _passthrough(self, indata: np.ndarray) -> np.ndarray:
        return indata

    def start(self) -> None:
        self._audio_io.start()

    def stop(self) -> None:
        self._audio_io.stop()

    def get_stats(self) -> Dict[str, Any]:
        return self._audio_io.get_stats()


class AudioFileWriter:
    """
    Threaded audio file writer for recording.

    Writes audio to file in background thread to avoid blocking audio callback.
    """

    def __init__(self, filename: str, sample_rate: int, channels: int):
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._total_samples = 0

    def start(self) -> None:
        """Start writer thread."""
        import soundfile as sf

        self._running = True
        self._file = sf.SoundFile(
            self.filename,
            "w",
            samplerate=self.sample_rate,
            channels=self.channels,
            format="WAV",
            subtype="FLOAT",
        )

        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self) -> None:
        """Background write loop."""
        while self._running or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
                self._file.write(data)
                self._total_samples += len(data)
            except queue.Empty:
                continue

    def write(self, data: np.ndarray) -> None:
        """Queue data for writing (non-blocking)."""
        self._queue.put(data.copy())

    def stop(self) -> None:
        """Stop writer and close file."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._file.close()

    @property
    def samples_written(self) -> int:
        return self._total_samples
