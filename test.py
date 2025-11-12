from pinecone.grpc import PineconeGRPC as Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print("Listing all indexes in your account:")
try:
    print(pc.list_indexes())
except Exception as e:
    print("❌ Error listing indexes:", e)
