import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from app.services.audio_service import transcribe_audio # Import our new service

class IngestionAgent:
    """
    An agent responsible for ingesting data from various sources.
    It detects the file type and uses the appropriate loader or model.
    """
    def __init__(self):
        print("✅ IngestionAgent initialized.")
        # We don't need to load the model here anymore

    def ingest(self, file_path: str):
        """
        Loads a document from a file path and extracts its content.
        For audio files, it transcribes them to text.
        """
        _, file_extension = os.path.splitext(file_path)
        print(f"\n🚀 Ingestion Agent: Processing file '{file_path}'...")

        # Define supported audio extensions
        audio_extensions = ['.mp3', '.m4a', '.wav', '.mpeg']

        if file_extension.lower() == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
        elif file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_extension.lower() in audio_extensions:
            # --- Use our dedicated audio service ---
            print("🔊 Audio file detected. Calling AudioService...")
            transcribed_text = transcribe_audio(file_path)
            
            if "Error:" in transcribed_text:
                print(f"❌ Ingestion Agent: {transcribed_text}")
                return None
                
            metadata = {"source": file_path}
            documents = [Document(page_content=transcribed_text, metadata=metadata)]
        else:
            print(f"⚠️ Ingestion Agent: File type '{file_extension}' not supported.")
            return None

        print(f"✅ Ingestion Agent: Successfully processed content.")
        return documents

if __name__ == "__main__":
    # This block allows us to run this file directly to test it.
    
    # --- Create a dummy file for testing ---
    test_file_path = "test_note.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test note.")
    
    print("--- Testing IngestionAgent with .txt file ---")
    agent = IngestionAgent()
    docs = agent.ingest(test_file_path)
    
    if docs:
        print("\n--- Ingested Content ---")
        print(docs[0].page_content)
        print("------------------------")
    
    # Clean up the dummy file
    os.remove(test_file_path)