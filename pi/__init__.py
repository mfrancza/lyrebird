"""
Lyrebird Raspberry Pi deployment package.

Real-time audio effects processing using trained neural FIR models.
"""

from .ring_buffer import RingBuffer, BatchRingBuffer

__all__ = [
    'RingBuffer',
    'BatchRingBuffer',
]

# Optional imports that require sounddevice/portaudio
try:
    from .audio_io import AudioConfig, AudioIO, list_devices, print_devices
    from .realtime_processor import RealtimeProcessor, BatchedRealtimeProcessor
    __all__.extend([
        'AudioConfig',
        'AudioIO',
        'list_devices',
        'print_devices',
        'RealtimeProcessor',
        'BatchedRealtimeProcessor',
    ])
except (ImportError, OSError):
    pass  # sounddevice/portaudio not available
