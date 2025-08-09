#!/bin/bash

# Ben AI Enhanced UI Startup Script

echo "🚀 Starting Ben AI Enhanced UI..."
echo ""

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OPEN_CMD="open"
else
    OPEN_CMD="xdg-open"
fi

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your API keys:"
    echo "  OPENAI_API_KEY=your_key_here"
    echo "  PINECONE_API_KEY=your_key_here"
    echo "  PINECONE_ENV=your_env_here"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start backend server
echo "📦 Starting backend server..."
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check the logs above."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Open frontend in browser
echo "🌐 Opening frontend in browser..."
$OPEN_CMD frontend/index.html

echo ""
echo "✨ Ben AI Enhanced UI is running!"
echo ""
echo "📍 Backend: http://localhost:8000"
echo "📍 Frontend: file://$(pwd)/frontend/index.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Keep script running
wait $BACKEND_PID