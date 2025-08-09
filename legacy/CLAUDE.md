# Ben AI Enhanced UI - Project Documentation

## Project Overview
This is an enhanced web-based UI for the Ben AI Chatbot that provides a modern, flexible alternative to the Streamlit interface. It features a Claude-inspired design with real-time WebSocket support, responsive layouts, and a rich user experience.

## Architecture
- **Backend**: FastAPI with WebSocket support for real-time chat
- **Frontend**: Vanilla HTML/CSS/JavaScript with modern design patterns
- **Database**: Pinecone vector database for benchmark data
- **AI**: OpenAI GPT-4 for conversational AI

## Key Features
- Real-time chat with WebSocket connection
- Claude-inspired warm, organic UI design
- Sidebar with quick-access sample queries
- Auto-scrolling chat interface
- Responsive design for all devices
- Markdown formatting support
- Session management
- Graceful fallback to REST API if WebSocket fails

## Directory Structure
```
Chatbot_BenAI_Enhanced_UI/
├── backend/               # FastAPI backend server
│   ├── app.py            # Main API server with WebSocket
│   ├── config/           # Configuration files
│   └── requirements.txt  # Python dependencies
├── frontend/             # Web UI
│   ├── index.html        # Main HTML structure
│   ├── css/              # Styling
│   └── js/               # JavaScript modules
├── docker-compose.yml    # Docker configuration
└── start_new_ui.sh      # Startup script
```

## Setting Up Virtual Environment in Windsurf/Cursor

### Windows Setup

1. **Open the project in Windsurf/Cursor**
   - Open the `Chatbot_BenAI_Enhanced_UI` folder

2. **Open the integrated terminal**
   - Press `Ctrl+`` or View → Terminal

3. **Create a virtual environment**:
   ```powershell
   # In the project root directory
   python -m venv venv
   ```

4. **Activate the virtual environment**:
   ```powershell
   # For Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # If you get an execution policy error, run:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   
   # For Windows Command Prompt
   venv\Scripts\activate.bat
   ```

5. **Install dependencies**:
   ```powershell
   # Make sure you're in the project root with venv activated
   pip install -r backend/requirements.txt
   ```

6. **Configure Windsurf/Cursor to use the virtual environment**:
   - Press `Ctrl+Shift+P` to open command palette
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.\venv\Scripts\python.exe`

### macOS/Linux Setup

1. **Open the integrated terminal** (`Cmd+`` or View → Terminal)

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate virtual environment**:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

5. **Configure Windsurf/Cursor**:
   - Press `Cmd+Shift+P` (Command Palette)
   - Type "Python: Select Interpreter"
   - Choose `./venv/bin/python`

## Environment Variables Setup

1. **Copy the example env file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file** with your API keys:
   ```env
   OPENAI_API_KEY=your_actual_openai_key
   PINECONE_API_KEY=your_actual_pinecone_key
   PINECONE_ENV=your_pinecone_environment
   ```

## Running the Application

### Development Mode (Recommended)

1. **Start the backend** (with venv activated):
   ```bash
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the frontend**:
   - Simply open `frontend/index.html` in your browser
   - Or serve it locally:
     ```bash
     # In a new terminal
     cd frontend
     python -m http.server 3000
     ```

### Using the Startup Script

**macOS/Linux**:
```bash
chmod +x start_new_ui.sh
./start_new_ui.sh
```

**Windows** (create `start_new_ui.bat`):
```batch
@echo off
echo Starting Ben AI Enhanced UI...

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start backend
cd backend
start /B uvicorn app:app --reload --host 0.0.0.0 --port 8000
cd ..

REM Wait for backend to start
timeout /t 3

REM Open frontend in browser
start frontend\index.html

echo Ben AI Enhanced UI is running!
echo Backend: http://localhost:8000
echo Press Ctrl+C in the backend window to stop
pause
```

## API Endpoints

- `GET /` - Health check
- `GET /api/health` - Backend status
- `POST /api/chat` - Send chat message
- `GET /api/suggestions` - Get sidebar suggestions
- `GET /api/benchmarks` - List benchmarks
- `POST /api/search` - Search benchmarks
- `WS /ws/{session_id}` - WebSocket connection

## Troubleshooting

### Virtual Environment Issues

**"python not found" error**:
- Windows: Use `py -m venv venv` instead
- Mac/Linux: Use `python3 -m venv venv`

**"cannot be loaded because running scripts is disabled"** (Windows):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windsurf/Cursor not recognizing venv**:
1. Restart Windsurf/Cursor after creating venv
2. Manually select interpreter: `Ctrl/Cmd+Shift+P` → "Python: Select Interpreter"
3. Choose the path to your venv Python executable

### Backend Connection Issues

**"Connection refused" error**:
1. Ensure backend is running on port 8000
2. Check firewall settings
3. Verify `.env` file has correct API keys

**WebSocket connection fails**:
- The app automatically falls back to REST API
- Check browser console for specific errors
- Ensure no proxy/firewall blocking WebSocket

### Frontend Issues

**Blank page**:
1. Open browser developer console (F12)
2. Check for JavaScript errors
3. Ensure all files are loaded correctly
4. Try a different browser

## Development Tips

1. **Hot Reload**: Backend auto-reloads with `--reload` flag
2. **Browser DevTools**: Use for debugging frontend
3. **Network Tab**: Monitor API calls and WebSocket messages
4. **Console Logs**: Check both browser and terminal for errors

## Testing the Setup

1. **Test backend health**:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. **Test chat endpoint**:
   ```bash
   curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the minimum for S&P 500?"}'
   ```

3. **Test WebSocket** (in browser console):
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/ws/test123');
   ws.onmessage = (e) => console.log('Received:', e.data);
   ws.send(JSON.stringify({type: 'message', message: 'Hello'}));
   ```

## Project Status
- ✅ Backend API implementation
- ✅ WebSocket real-time chat
- ✅ Frontend UI with Claude-inspired design
- ✅ Sidebar with sample queries
- ✅ Session management
- ✅ Docker configuration
- ✅ Deployment documentation

## Next Steps
- Add user authentication (optional)
- Implement chat history persistence
- Add export chat functionality
- Enhanced error handling
- Performance monitoring
- Analytics integration