# Ben AI - Enhanced HTML/CSS/JS UI

A modern, flexible web interface for the Ben AI Chatbot with real-time WebSocket support and a Claude-inspired design.

## Features

- 🎨 **Modern UI**: Clean, warm, organic design inspired by Claude
- ⚡ **Real-time Chat**: WebSocket support for instant messaging
- 📱 **Responsive**: Works seamlessly on desktop, tablet, and mobile
- 🎯 **Smart Sidebar**: Quick-access sample queries to get started
- 🔄 **Auto-scroll**: Smooth scrolling to latest messages
- ✨ **Rich Formatting**: Markdown support, code highlighting, and more

## Quick Start

### Option 1: Local Development (Recommended for Testing)

1. **Install Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

2. **Set Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENV=your_pinecone_environment
   ```

3. **Start the Backend Server**:
   ```bash
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Open the Frontend**:
   - Simply open `frontend/index.html` in your web browser
   - Or use a local server:
     ```bash
     cd frontend
     python -m http.server 3000
     ```
   - Navigate to `http://localhost:3000`

### Option 2: Docker Deployment

1. **Set Environment Variables**:
   Create a `.env` file with your API keys

2. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

3. **Access the Application**:
   - Open `http://localhost` in your browser

### Option 3: Cloud Deployment

#### Frontend (Vercel/Netlify)
1. Push the `frontend` folder to a GitHub repository
2. Connect to Vercel or Netlify
3. Deploy with default settings

#### Backend (Railway/Render)
1. Push the backend code to a GitHub repository
2. Connect to Railway or Render
3. Set environment variables in the platform
4. Deploy with Python buildpack

## Usage

### Welcome Screen
- Enter your question in the main input field
- Press Enter or click the send button
- Or press Tab to open the sidebar with suggestions

### Chat Interface
- Type messages in the input field at the bottom
- Press Enter to send (Shift+Enter for new line)
- Click the menu button to open the sidebar
- Click "New Chat" to start fresh

### Sidebar Suggestions
Click any suggestion to automatically submit it:
- **Getting Started**: Basic benchmark queries
- **Portfolio Analysis**: Eligibility checks
- **Benchmark Search**: Find alternatives

### Keyboard Shortcuts
- `Enter`: Send message
- `Shift+Enter`: New line in message
- `Tab`: Open sidebar (from welcome screen)
- `Cmd/Ctrl+K`: Toggle sidebar
- `Escape`: Close sidebar

## Architecture

```
├── backend/
│   ├── app.py              # FastAPI server with WebSocket support
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Container configuration
├── frontend/
│   ├── index.html          # Main HTML structure
│   ├── css/
│   │   ├── styles.css      # Main styles
│   │   └── animations.css  # Smooth transitions
│   ├── js/
│   │   ├── app.js          # Main application controller
│   │   ├── chat.js         # Chat message handling
│   │   ├── websocket.js    # Real-time connection
│   │   └── sidebar.js      # Sidebar interactions
│   └── assets/
├── docker-compose.yml      # Multi-container setup
└── nginx.conf             # Web server configuration
```

## Comparison with Streamlit Version

| Feature | Streamlit | New HTML/CSS/JS |
|---------|-----------|-----------------|
| Real-time Updates | Page refresh | WebSocket |
| Customization | Limited | Full control |
| Performance | Good | Excellent |
| Mobile Support | Basic | Responsive |
| Animations | Limited | Smooth |
| Deployment | Streamlit Cloud | Any static host |

## Development

### Backend API Endpoints
- `POST /api/chat`: Send message and receive response
- `GET /api/suggestions`: Get sidebar suggestions
- `GET /api/benchmarks`: List available benchmarks
- `POST /api/search`: Search benchmarks
- `WS /ws/{session_id}`: WebSocket connection

### Frontend Structure
- **app.js**: Main application controller
- **chat.js**: Message handling and formatting
- **websocket.js**: Real-time connection management
- **sidebar.js**: Sidebar and gesture controls

## Troubleshooting

### Backend Connection Issues
1. Ensure the backend is running on port 8000
2. Check CORS settings if accessing from different domain
3. Verify API keys are set correctly

### WebSocket Connection
1. Check browser console for errors
2. Ensure WebSocket port is not blocked
3. Falls back to REST API automatically

### UI Issues
1. Clear browser cache
2. Check browser compatibility (Chrome, Firefox, Safari, Edge)
3. Ensure JavaScript is enabled

## License

This project maintains the same license as the original Ben AI Chatbot.