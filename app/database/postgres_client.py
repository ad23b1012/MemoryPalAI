# app/database/postgres_client.py
import logging
from sqlalchemy import create_engine, text
from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# Fallback SQLite engine if Postgres is unavailable
_engine = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _engine = engine
        logger.info(f"Connected to database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'local'}")
        return _engine
    except Exception as e:
        logger.warning(f"Database connection to '{DATABASE_URL}' failed: {e}. Falling back to SQLite.")
        fallback_url = "sqlite:///app/database/memorypal_fallback.db"
        _engine = create_engine(fallback_url)
        return _engine


def initialize_database():
    """
    Connects to the database and creates the necessary tables
    for the knowledge graph (nodes, edges) and document tracking.
    """
    engine = get_engine()
    try:
        with engine.connect() as connection:
            # Nodes Table
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS nodes (
                id VARCHAR(255) PRIMARY KEY,
                label VARCHAR(255),
                type VARCHAR(100)
            );
            """))

            # Edges Table
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS edges (
                id SERIAL PRIMARY KEY,
                source_id VARCHAR(255),
                target_id VARCHAR(255),
                label VARCHAR(255)
            );
            """))

            # Quiz Analytics Table
            connection.execute(text("""
            CREATE TABLE IF NOT EXISTS quiz_history (
                id SERIAL PRIMARY KEY,
                subject VARCHAR(255),
                score FLOAT,
                total_questions INT,
                evaluation_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """))

            connection.commit()
            logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")


def save_graph_to_db(graph_data: dict):
    """Saves extracted nodes and edges into the relational database."""
    engine = get_engine()
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    try:
        with engine.connect() as conn:
            for node in nodes:
                nid = node.get("id")
                ntype = node.get("type", "Concept")
                if nid:
                    # SQLite and Postgres compatible upsert or ignore
                    try:
                        conn.execute(
                            text("INSERT INTO nodes (id, label, type) VALUES (:id, :label, :type) ON CONFLICT DO NOTHING"),
                            {"id": nid, "label": nid, "type": ntype}
                        )
                    except Exception:
                        pass

            for edge in edges:
                src = edge.get("source")
                tgt = edge.get("target")
                lbl = edge.get("label", "related_to")
                if src and tgt:
                    try:
                        conn.execute(
                            text("INSERT INTO edges (source_id, target_id, label) VALUES (:src, :tgt, :lbl)"),
                            {"src": src, "tgt": tgt, "lbl": lbl}
                        )
                    except Exception:
                        pass

            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving graph to database: {e}")
        return False


if __name__ == "__main__":
    print("Initializing database...")
    initialize_database()
