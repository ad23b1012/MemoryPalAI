# app/database/vector_db.py
import os
import chromadb

# ✅ Persistent client — stores vector DB to disk
client = chromadb.PersistentClient(path="./chroma_db")


def update_gitignore():
    """Ensure 'chroma_db/' is ignored by Git."""
    gitignore_path = ".gitignore"
    try:
        if not os.path.exists(gitignore_path):
            print("🪶 Creating new .gitignore...")
            with open(gitignore_path, "w") as f:
                f.write("# Gitignore file\n")

        with open(gitignore_path, "r+") as f:
            content = f.read()
            if "chroma_db/" not in content:
                f.write("\n\n# VectorDB Data\nchroma_db/\n")
                print("✅ Updated .gitignore to ignore 'chroma_db/'")
    except Exception as e:
        print(f"⚠️ Error updating .gitignore: {e}")


def get_vector_collection(name="memorypal_collection"):
    """
    Gets or creates a persistent ChromaDB collection.
    """
    try:
        collection = client.get_or_create_collection(name=name)
        print(f"✅ ChromaDB collection '{name}' loaded/created.")
        return collection
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return None


# ✅ Class Wrapper for compatibility
class VectorDB:
    """Wrapper for simple add/query/clear interface around ChromaDB."""

    def __init__(self, collection_name: str = "memorypal_collection"):
        update_gitignore()
        self.collection = get_vector_collection(collection_name)

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Add document to the persistent collection."""
        if not self.collection:
            print("❌ No ChromaDB collection initialized.")
            return
        if not content or not isinstance(content, str):
            print(f"⚠️ Invalid document {doc_id}")
            return
        metadata = metadata or {"source": "unknown"}
        try:
            self.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
            )
            print(f"📥 Added doc '{doc_id}' (source={metadata.get('source')})")
        except Exception as e:
            print(f"❌ Error adding doc {doc_id}: {e}")

    def query(self, query_text: str, top_k: int = 3):
        """Query for similar documents."""
        if not query_text:
            print("⚠️ Empty query text.")
            return {"documents": [], "metadatas": [], "distances": []}
        try:
            return self.collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def clear(self):
        """Delete all documents."""
        try:
            self.collection.delete(where={})
            print("🧹 Cleared ChromaDB collection.")
        except Exception as e:
            print(f"⚠️ Error clearing DB: {e}")

    def count(self):
        """Return collection size."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"⚠️ Count error: {e}")
            return 0


# ---------------- Test Block ----------------
if __name__ == "__main__":
    print("🧠 Testing ChromaDB connection...")
    update_gitignore()
    vdb = VectorDB()
    print(f"Collection Count: {vdb.count()}")
