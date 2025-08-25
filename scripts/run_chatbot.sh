#!/bin/bash

# Ben AI Chatbot - One-Command Launcher
# This script handles everything: kills old processes, starts backend, opens frontend

echo "🚀 Starting Ben AI Chatbot..."

# Kill any existing processes on port 8000
echo "📍 Checking for existing servers..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "   Stopping existing server on port 8000..."
    kill $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null
    sleep 2
fi

# Navigate to backend directory
cd "$(dirname "$0")/../backend" || exit

# Start the backend server
echo "🔧 Starting backend server..."
uvicorn app:app --reload --host 0.0.0.0 --port 8000 > server.log 2>&1 &

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
for i in {1..10}; do
    if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo "✅ Backend server is running!"
        break
    fi
    sleep 1
done

# Open the frontend
echo "🌐 Opening frontend in browser..."
open ../frontend/index.html

echo ""
echo "✨ Ben AI Chatbot is ready!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: file://$(cd ../frontend && pwd)/index.html"
echo ""
echo "📝 Server logs: backend/server.log"
echo "🛑 To stop: Press Ctrl+C or run 'pkill -f uvicorn'"
echo ""

# Keep script running and show logs
tail -f server.log