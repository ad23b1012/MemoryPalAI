from sqlalchemy import create_engine, text
# Import the database URL from our config file
from app.config import DATABASE_URL 

# Create the main connection engine
engine = create_engine(DATABASE_URL)

def initialize_database():
    """
    Connects to the database and creates the necessary tables
    for the knowledge graph (nodes and edges).
    """
    print("Attempting to connect to the database...")
    try:
        with engine.connect() as connection:
            print("Connection successful. Checking/creating tables...")
            
            # Create Nodes Table (for knowledge graph entities)
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS nodes (
                id VARCHAR(255) PRIMARY KEY,
                label VARCHAR(255),
                type VARCHAR(100)
            );
            """))
            
            # Create Edges Table (for knowledge graph relationships)
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS edges (
                id SERIAL PRIMARY KEY,
                source_id VARCHAR(255) REFERENCES nodes(id) ON DELETE CASCADE,
                target_id VARCHAR(255) REFERENCES nodes(id) ON DELETE CASCADE,
                label VARCHAR(255)
            );
            """))
            
            # Commit the changes to create the tables
            connection.commit()
            print("✅ Database tables checked/created successfully.")
            
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

if __name__ == "__main__":
    # This block allows us to run this file directly to test it.
    print("Initializing database...")
    initialize_database()