import chromadb
import os

# Create a persistent client that saves to disk
# This will create a 'chroma_db' folder in your project root
client = chromadb.PersistentClient(path="./chroma_db")

# We must tell .gitignore to ignore this new folder
def update_gitignore():
    gitignore_path = '.gitignore'
    try:
        # Check if the .gitignore file exists
        if not os.path.exists(gitignore_path):
            print("Creating .gitignore file...")
            with open(gitignore_path, 'w') as f:
                f.write("# Gitignore file\n")
        
        # Now, read and append
        with open(gitignore_path, 'r+') as f:
            content = f.read()
            if 'chroma_db/' not in content:
                # Add a new line before adding the new rules
                f.write('\n\n# VectorDB Data\nchroma_db/\n')
                print("Updated .gitignore to ignore 'chroma_db/'")
    except Exception as e:
        print(f"Error updating .gitignore: {e}")

def get_vector_collection(name="memorypal_collection"):
    """
    Gets or creates a ChromaDB collection.
    """
    try:
        collection = client.get_or_create_collection(name=name)
        print(f"✅ ChromaDB collection '{name}' loaded/created.")
        return collection
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")
        return None

if __name__ == "__main__":
    # This block allows us to run this file directly to test it.
    print("Testing ChromaDB connection...")
    update_gitignore()
    collection = get_vector_collection()
    if collection:
        print(f"Collection count: {collection.count()}")