#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  🧠 Starting MemoryPal AI (FastAPI Backend + Next.js 15) "
echo "=========================================================="

# 1. Determine Python executable
if [ -f ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
  source .venv/bin/activate
else
  PYTHON_BIN="python3"
fi

# 2. Start FastAPI Backend in background
echo "Starting FastAPI Backend on http://localhost:8501..."
$PYTHON_BIN -m app.server &
BACKEND_PID=$!

# 3. Start Next.js Frontend in background
echo "Starting Next.js Frontend on http://localhost:3000..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

trap "echo 'Stopping MemoryPal AI...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM EXIT

echo ""
echo "=========================================================="
echo "  🚀 MemoryPal AI Platform is Running!"
echo "  👉 Next.js 15 UI     : http://localhost:3000"
echo "  👉 FastAPI Engine    : http://localhost:8501"
echo "  👉 Interactive API   : http://localhost:8501/docs"
echo "  ⚠️ Always open http://localhost:3000 (avoid 0.0.0.0)"
echo "=========================================================="
echo "Press Ctrl+C to gracefully shut down both services."
echo ""

wait
