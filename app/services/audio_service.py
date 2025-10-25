import whisper
import os

# Load the model once when the service is imported
# 'base' is a good balance of speed and accuracy
try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    model = None

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using the Whisper model.
    """
    if model is None:
        return "Error: Whisper model is not loaded."
        
    print(f"Transcribing audio file: {file_path}...")
    try:
        # We need to copy the file to a local path without spaces
        # because ffmpeg (used by Whisper) can have issues with spaces
        temp_audio_path = "/tmp/temp_audio_for_transcription"
        
        # In a real app, you'd handle different file extensions
        # For now, let's assume the file has a proper extension
        
        # A simple copy (in a production app, you'd handle this more robustly)
        # For local files, this is less of an issue, but for GDrive paths it was
        # Let's try to transcribe directly first.
        result = model.transcribe(file_path)
        transcribed_text = result['text']
        print("✅ Audio transcribed successfully.")
        return transcribed_text
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return f"Error transcribing audio: {e}"