# Import base classes
from .base import STTProvider, TranscriptionResult

# Import all provider implementations
from .deepgram_stt import DeepgramProvider
from .assemblyai_stt import AssemblyAIProvider
from .speechmatics_stt import SpeechmaticsProvider

# Import Google provider conditionally
try:
    from .google_stt import GoogleProvider, GOOGLE_AVAILABLE
except ImportError:
    GOOGLE_AVAILABLE = False
    GoogleProvider = None

__all__ = [
    "STTProvider",
    "TranscriptionResult",
    "DeepgramProvider",
    "AssemblyAIProvider",
    "SpeechmaticsProvider",
    "GoogleProvider",
    "GOOGLE_AVAILABLE",
]
