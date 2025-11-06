# app/services/audio_service.py
import whisper
import os

# -----------------------------
# Load Whisper model globally
# -----------------------------
try:
    model = whisper.load_model("base")  # use "tiny" for faster tests
    print("✅ Whisper model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    model = None


# -----------------------------
# Transcription Function
# -----------------------------
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using the Whisper model.
    """
    if model is None:
        return "❌ Error: Whisper model not loaded."

    if not os.path.exists(file_path):
        return f"❌ Error: File not found at {file_path}"

    print(f"🎧 Transcribing audio file: {file_path}...")
    try:
        result = model.transcribe(file_path)
        text = result.get("text", "").strip()
        print("✅ Audio transcribed successfully.")
        return text if text else "⚠️ No speech detected in audio."
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return f"Error transcribing audio: {e}"


# -----------------------------
# Test Code (Run Directly)
# -----------------------------
if __name__ == "__main__":
    test_audio_path = "/Users/abhi/Desktop/Agentic/MemoryPalAI/tests/test.m4a"  # change if needed

    print("🚀 Running Whisper Transcription Test...\n")
    output_text = transcribe_audio(test_audio_path)

    print("\n--- Transcription Output ---")
    print(output_text)
    print("\n✅ Test Completed.")
