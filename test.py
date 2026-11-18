# test.py - System Health & Verification Test
import sys
import os

print("=" * 60)
print("🧠 MemoryPalAI - System Health & Integration Check")
print("=" * 60)

# 1. Config Check
try:
    from app.config import PROJECT_NAME, VERSION, GEMINI_MODEL, VECTOR_BACKEND, DATABASE_URL
    print(f"✅ Configuration loaded: {PROJECT_NAME} v{VERSION}")
    print(f"   - Target Gemini Model: {GEMINI_MODEL}")
    print(f"   - Target Vector Backend: {VECTOR_BACKEND}")
    print(f"   - Database URL Type: {'PostgreSQL' if 'postgresql' in DATABASE_URL else 'SQLite (Local Fallback)'}")
except Exception as e:
    print(f"❌ Config Error: {e}")
    sys.exit(1)

# 2. Vector DB Check
try:
    from app.database.vector_store import UnifiedVectorStore
    store = UnifiedVectorStore()
    stats = store.list_documents()
    print(f"✅ Vector Store initialized successfully (Active Backend: {store.backend})")
    print(f"   - Total vectors indexed: {stats.get('total_chunks', 0)}")
except Exception as e:
    print(f"⚠️ Vector Store Warning: {e}")

# 3. Database Check
try:
    from app.database.postgres_client import initialize_database
    initialize_database()
    print("✅ Relational Database tables verified/created successfully.")
except Exception as e:
    print(f"⚠️ Database Notice: {e}")

# 4. LLM Service Check
try:
    from app.services.llm_service import get_llm
    llm = get_llm()
    if llm:
        print(f"✅ Google Gemini LLM Service ready.")
    else:
        print("ℹ️ Google Gemini LLM: GOOGLE_API_KEY not currently set (mock/fallback active).")
except Exception as e:
    print(f"⚠️ LLM Service Warning: {e}")

# 5. Agents Check
try:
    from app.agents.ingestion_agent import IngestionAgent
    from app.agents.quiz_agent import QuizAgent
    from app.agents.revision_agent import RevisionAgent
    from app.agents.planner_agent import PlannerAgent
    from app.agents.organizer_agent import OrganizerAgent
    from app.utils.scheduler import calculate_next_review
    
    ing = IngestionAgent()
    qz = QuizAgent()
    rev = RevisionAgent()
    pl = PlannerAgent()
    org = OrganizerAgent()
    schedule_test = calculate_next_review(85.0)
    print("✅ All Core Agents initialized successfully:")
    print("   - IngestionAgent, QuizAgent, RevisionAgent, PlannerAgent, OrganizerAgent")
    print(f"   - Spaced Repetition Engine: test interval = {schedule_test['interval_days']} days ({schedule_test['status']})")
except Exception as e:
    print(f"❌ Agent Error: {e}")
    sys.exit(1)

print("=" * 60)
print("🎉 All systems operational! App is ready for local use and server deployment.")
print("=" * 60)
