# app/services/audio_service.py
import os
import logging

logger = logging.getLogger(__name__)

_whisper_model = None


def get_whisper_model():
    """Lazy loads whisper model only when audio transcription is requested."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        import whisper
        # Use 'base' model for high balance of speed and accuracy
        logger.info("Loading Whisper transcription model...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully.")
        return _whisper_model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file (mp3, wav, m4a, etc.) into text using Whisper.
    """
    if not file_path or not os.path.exists(file_path):
        return f"❌ Audio file not found at: {file_path}"

    model = get_whisper_model()
    if model is None:
        return (
            "⚠️ Whisper model could not be loaded. "
            "Please ensure 'openai-whisper' and 'ffmpeg' are installed on the server."
        )

    try:
        logger.info(f"Transcribing audio: {file_path}")
        result = model.transcribe(file_path)
        text = result.get("text", "").strip()
        return text if text else "⚠️ No audible speech detected in this audio file."
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}")
        return f"❌ Audio transcription error: {e}"
