# 🚀 Ben AI Enhanced UI - Quick Start Guide

## First Time Setup (Choose Your OS)

### 🪟 Windows Users

1. **Run the setup script** (double-click or run in terminal):
   ```
   setup_venv.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install all dependencies
   - Set up your project

2. **Configure your API keys**:
   - Edit the `.env` file that was created
   - Add your actual API keys (don't share these!)

3. **Start the application**:
   ```
   start_new_ui.bat
   ```
   - The backend will start
   - Your browser will open automatically
   - You're ready to chat!

### 🍎 macOS/Linux Users

1. **Run the setup script**:
   ```bash
   ./setup_venv.sh
   ```
   This will:
   - Create a Python virtual environment
   - Install all dependencies
   - Set up your project

2. **Configure your API keys**:
   - Edit the `.env` file that was created
   - Add your actual API keys

3. **Start the application**:
   ```bash
   ./start_new_ui.sh
   ```
   - The backend will start
   - Your browser will open automatically
   - You're ready to chat!

## 💡 Using in Windsurf/Cursor

### Setting Up the IDE

1. **Open the project folder** in Windsurf/Cursor

2. **Select Python Interpreter**:
   - Press `Ctrl/Cmd + Shift + P`
   - Type "Python: Select Interpreter"
   - Choose:
     - Windows: `.\venv\Scripts\python.exe`
     - Mac/Linux: `./venv/bin/python`

3. **Open integrated terminal** and the virtual environment should auto-activate

### Running from Windsurf/Cursor

**Option 1: Use the scripts**
- Terminal → New Terminal
- Windows: `start_new_ui.bat`
- Mac/Linux: `./start_new_ui.sh`

**Option 2: Manual start**
```bash
# Activate virtual environment (if not auto-activated)
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Start backend
cd backend
uvicorn app:app --reload

# Open frontend/index.html in browser
```

## 🎯 What You'll See

1. **Welcome Screen**: Clean interface with input field
2. **Sidebar**: Click menu icon or press Tab for quick suggestions
3. **Chat Interface**: Real-time messaging with Ben AI
4. **Auto-features**: Scrolling, markdown formatting, responsive design

## ⚡ Quick Commands

- **Send message**: `Enter`
- **New line**: `Shift + Enter`
- **Toggle sidebar**: `Ctrl/Cmd + K`
- **Open suggestions**: `Tab` (from welcome screen)
- **New chat**: Click the `+` button

## 🔧 Troubleshooting

**"Python not found"**:
- Install Python 3.8+ from python.org
- Windows: Try using `py` instead of `python`

**"Permission denied"** (Mac/Linux):
```bash
chmod +x setup_venv.sh
chmod +x start_new_ui.sh
```

**Backend won't start**:
- Check `.env` file has valid API keys
- Ensure port 8000 is not in use
- Check firewall settings

**Frontend blank page**:
- Open browser console (F12)
- Check for errors
- Try a different browser

## 📚 More Information

- Full documentation: See `CLAUDE.md`
- API endpoints: http://localhost:8000/docs (when running)
- Original Streamlit version: Run `streamlit run streamlit_app.py`

## 🎉 Ready to Go!

Your enhanced UI is now set up. The new interface provides:
- Better performance than Streamlit
- Real-time WebSocket updates
- Beautiful, responsive design
- Full customization capabilities

Enjoy using Ben AI with the new enhanced interface!