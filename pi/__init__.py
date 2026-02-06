"""
Lyrebird Raspberry Pi deployment package.

Real-time audio effects processing using trained neural FIR models.
"""

from .ring_buffer import RingBuffer, BatchRingBuffer

__all__ = [
    "RingBuffer",
    "BatchRingBuffer",
]

# Optional imports that require sounddevice/portaudio
# These are re-exported for package users, so we silence the unused import warning
try:
    from .audio_io import AudioConfig  # noqa: F401
    from .audio_io import AudioIO  # noqa: F401
    from .audio_io import list_devices  # noqa: F401
    from .audio_io import print_devices  # noqa: F401
    from .realtime_processor import BatchedRealtimeProcessor  # noqa: F401
    from .realtime_processor import RealtimeProcessor  # noqa: F401

    __all__.extend(
        [
            "AudioConfig",
            "AudioIO",
            "list_devices",
            "print_devices",
            "RealtimeProcessor",
            "BatchedRealtimeProcessor",
        ]
    )
except (ImportError, OSError):
    pass  # sounddevice/portaudio not available
