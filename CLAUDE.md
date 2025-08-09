# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Ben AI Enhanced UI is a modern web-based financial advisor chatbot that provides benchmark eligibility guidance. It features a FastAPI backend with WebSocket support and a Claude-inspired frontend with real-time chat capabilities.

## USER PREFERENCE
When proposing changes, explain the changes and justify why they are necessary.

## Architecture & Core Components

### Backend Architecture (FastAPI + WebSockets)
- **Main Server**: `backend/app.py` - FastAPI application with REST API and WebSocket endpoints
- **Core Logic**: `backend/chatbot_core.py` - Chat processing, OpenAI integration, Pinecone vector search
- **Session Management**: In-memory sessions with WebSocket connection tracking via ConnectionManager class
- **Function Routing**: OpenAI function calling for benchmark queries (get_minimum, search_by_characteristics, search_benchmarks)

### Frontend Architecture (Modular JavaScript)
- **Main Controller**: `frontend/js/app.js` - BenAIApp class orchestrates all components
- **Chat Management**: `frontend/js/chat.js` - ChatManager handles message rendering and API calls
- **WebSocket Handler**: `frontend/js/websocket.js` - WebSocketManager with auto-reconnect and fallback to REST
- **UI Components**: `frontend/js/sidebar.js` - SidebarManager for quick-start queries with swipe gestures

### Data Flow
1. Frontend sends messages via WebSocket (preferred) or REST API fallback
2. Backend processes through chatbot_core.py using OpenAI function calling
3. Pinecone vector search retrieves relevant benchmark data from config/benchmarks.json
4. Real-time responses stream back through WebSocket connection

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r backend/requirements.txt
```

### Running the Application
```bash
# Start backend server (from project root)
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Open frontend (from project root)
open frontend/index.html
# Or serve with: python -m http.server 3000 (from frontend/)
```

### Testing & Debugging
```bash
# Test backend health
curl http://localhost:8000/api/health

# Test chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the minimum for S&P 500?"}'

# Run import diagnostics
cd backend && python test_imports.py
```

### Quick Start Scripts
```bash
# Automated setup (creates venv, installs deps)
./scripts/setup_venv.sh        # macOS/Linux
scripts/setup_venv.bat         # Windows

# Start application (backend + opens frontend)
./scripts/start_new_ui.sh      # macOS/Linux
scripts/start_new_ui.bat       # Windows
```

## Configuration Requirements

### Environment Variables (.env file required)
```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

### Critical File Dependencies
- `backend/config/system_prompt.txt` - AI behavior configuration and function routing logic
- `backend/config/benchmarks.json` - Financial benchmark data for Pinecone vector search
- `description_utils.py` - Semantic description generation for benchmarks

## Key Integration Points

### WebSocket Communication Protocol
- Client connects to `/ws/{session_id}`
- Message format: `{"type": "message", "message": "user_input"}`
- Server responses: `{"type": "message", "data": ChatResponse}` or `{"type": "typing", "status": "start/stop"}`
- Auto-fallback to REST `/api/chat` endpoint if WebSocket fails

### OpenAI Function Calling
The system uses OpenAI's function calling feature with three main functions:
- `get_minimum(name)` - Get specific benchmark minimum investment
- `search_by_characteristics(reference_benchmark, portfolio_size)` - Find similar benchmarks
- `search_benchmarks(query, filters)` - General benchmark search

### Session Management
- UUID-based session tracking across WebSocket and REST endpoints
- In-memory storage with automatic cleanup
- Message history trimming based on token limits (MAX_TOKENS_PER_REQUEST)

## Common Development Issues

### Version Compatibility
- OpenAI client must be >= 1.50.0 (older versions have proxy parameter conflicts)
- Pinecone client version 3.0.0+ required for proper Pinecone() initialization
- FastAPI app imports chatbot_core which must load config files from backend/config/

### File Path Dependencies
- Backend expects `../env` relative path for environment variables
- Config files must be in `backend/config/` directory
- Frontend assumes backend running on localhost:8000

### WebSocket Connection Management
- ConnectionManager class handles multiple simultaneous connections
- Graceful disconnect handling with cleanup
- Heartbeat mechanism prevents connection drops

## Frontend Module Dependencies
The frontend uses a modular class-based architecture where:
- BenAIApp instantiates and coordinates ChatManager, WebSocketManager, SidebarManager
- Each manager class operates independently but communicates through the main app instance
- View state managed through currentView property ('welcome' vs 'chat' screens)

## Recently Fixed Issues (Aug 2025)

### Critical Bug Fixes
1. **'bool' object is not iterable Error (Fixed)**
   - **Root Cause**: `get_trimmed_history()` in `app.py` was returning boolean instead of message list
   - **Solution**: Modified to return copy of trimmed messages instead of `trim_history()` return value
   - **Location**: `app.py:83-87`

2. **WebSocket Ping Message Rejection (Fixed)**
   - **Root Cause**: Backend rejected heartbeat `{"type": "ping"}` messages as invalid format
   - **Solution**: Added ping message handling in WebSocket endpoint to respond with pong
   - **Location**: `app.py:335-338`

3. **International Benchmark Search Failure (Fixed)**
   - **Root Cause**: "International" queries didn't map to actual region values in data
   - **Solution**: Added smart region mapping in `get_all_benchmarks()` - "international" → ["International Developed", "Global", "Emerging Markets"]
   - **Location**: `chatbot_core.py:816-834`

4. **Auto-scroll Issues (Fixed)**
   - **Root Cause**: Timing issues with scroll-to-bottom after message rendering
   - **Solution**: Enhanced scrollToBottom with requestAnimationFrame + improved timing
   - **Location**: `frontend/js/chat.js:193-204`

5. **Chat Layout & Input Box Disappearing (Fixed)**
   - **Root Cause**: CSS flex layout hierarchy issue - chat-container taking 100% space, pushing input off-screen
   - **Solution**: Fixed flex layout with `flex: 1 1 0`, `min-height: 0` for chat-container and `flex-shrink: 0` for input-container
   - **Location**: `frontend/css/styles.css:392-400, 518-523`
   - **Additional**: Added proper message spacing and ensured manual scrolling works

### Troubleshooting Guide
- **Version conflicts**: Ensure OpenAI>=1.50.0 and pinecone-client==3.0.0 (not latest pinecone package)
- **Chat gets stuck "thinking"**: Check for `'bool' object is not iterable` errors - usually session history issues
- **WebSocket connection errors**: Look for ping message rejections - backend should handle ping/pong properly
- **International benchmarks not found**: Verify region mapping logic handles "international" queries correctly
- **Input box disappearing**: Check flex layout hierarchy - chat-container needs `min-height: 0` and input-container needs `flex-shrink: 0`
- **Can't scroll manually**: Ensure chat-container has `overflow-y: auto` and proper flex sizing
- **Incorrect minimum values returned**: Often resolved by server restart to reload data; check for browser caching issues; verify data in benchmarks.json vs. API responses
- **Expensive alternatives being suggested**: Check if alternatives have minimums higher than user's portfolio size; indicates search logic bug or missing portfolio size filtering
- **Multi-benchmark comparisons give "sales rep" message**: Fixed - was security validation issue; system now supports full multi-benchmark comparisons with iterative function calling

6. **Search Logic Bug: Expensive Alternatives Suggested (Fixed)**
   - **Root Cause**: System prompt had incorrect examples ($300K instead of $2M for Russell 2000 Value) AND search logic allowed expensive alternatives to slip through portfolio size filtering
   - **Solution**: Fixed incorrect examples in system_prompt.txt + added defensive portfolio size filtering in search_by_characteristics function
   - **Location**: `backend/config/system_prompt.txt` (corrected examples) + `backend/chatbot_core.py:796-811` (defensive filtering)
   - **Key Fix**: Final safety check ensures NO alternatives with account_minimum > portfolio_size are ever returned

7. **Multi-Benchmark Comparison Bug (FULLY FIXED)**
   - **Root Cause**: Two issues: (1) OpenAI integration limited to single function call per request, (2) Security validation flagging multi-benchmark responses as suspicious
   - **Symptom**: Queries like "Compare minimums for A, B, and C" returned either first benchmark only OR "please ask sales rep" message
   - **Complete Fix**: 
     - **Added iterative function calling** with safety guardrails in `app.py:36-261` (max 5 calls, timeout protection, duplicate detection)
     - **Enhanced security validation** in `chatbot_core.py:1113-1186` to recognize legitimate multi-benchmark responses  
     - **Added fuzzy name matching** for NASDAQ variations (NASDAQ-100 → Nasdaq 100)
   - **Location**: `app.py:180-261` (iterative calling), `chatbot_core.py:1143-1177` (security updates), `system_prompt.txt:72-81` (guidance)
   - **Result**: Multi-benchmark comparisons now work perfectly from both welcome screen and ongoing chat

8. **Function Routing Issues - Technology Search and Geographic Consistency (FULLY FIXED)**
   - **Root Cause**: Three systematic issues: (1) NASDAQ-100 not appearing for technology queries, (2) Global benchmark alternatives returning US-only options, (3) Inconsistent function selection for similar queries
   - **Technology Search Issue**: Vector search didn't return NASDAQ-100 for "technology" queries, showing only Clean Technology benchmarks
   - **Geographic Inconsistency**: MSCI ACWI alternatives returned Russell benchmarks (US-only) instead of global/international options
   - **Complete Solution**: 
     - **Enhanced search_benchmarks function** in `chatbot_core.py:516-640` with sector-specific boosting and fallback injection
     - **Improved search_by_characteristics function** in `chatbot_core.py:822-903` with region-priority fallback logic  
     - **Systematic function routing** in `system_prompt.txt:47-129` with clear decision tree and flowchart
   - **Technical Fixes**:
     - **Sector Boosting**: Technology queries boost NASDAQ-100 relevance score by 2x
     - **Fallback Injection**: Missing sector benchmarks automatically injected (NASDAQ-100 for tech, MSCI World/EAFE for international)
     - **Geographic Priority**: Global benchmark alternatives prioritize global/international options over regional ones
     - **Region Preservation**: Fallback logic maintains geographic consistency in alternatives search
   - **Location**: `chatbot_core.py:562-647` (boosting), `chatbot_core.py:827-902` (geographic), `system_prompt.txt:47-129` (routing)
   - **Results**: 
     - ✅ Technology searches now return NASDAQ-100 as top result
     - ✅ MSCI ACWI alternatives return MSCI World/EAFE (global/international) instead of Russell benchmarks (US)
     - ✅ Function routing more consistent and predictable across similar query types

9. **Scalable Architecture Enhancement - Factor Benchmarks and Security Validation (FULLY FIXED)**
   - **Root Cause**: Two critical scalability issues: (1) Security validation blocking legitimate factor benchmark queries, (2) Hardcoded sector logic not scaling to new query types
   - **Factor Query Issue**: "factor benchmarks for US exposure" returned fallback security message despite successful function calls
   - **Scalability Problem**: Each new sector (factor, dividend, ESG, healthcare) required manual code additions rather than data-driven detection
   - **Complete Solution**:
     - **Enhanced security validation** in `chatbot_core.py:1237-1346` with function call success bypass and expanded keywords
     - **Data-driven sector detection** in `chatbot_core.py:612-673` using benchmark metadata tags instead of hardcoded lists
     - **Dynamic fallback system** that automatically detects and injects relevant benchmarks for any sector
     - **Function call success bypass** that allows legitimate responses when search functions return valid results
   - **Technical Architecture**:
     - **Expanded Financial Keywords**: Added "factor", "dividend", "value", "growth", "ESG", "sustainable", "momentum", "quality", sector terms
     - **Function Success Logic**: If function calls succeed, only block obvious injection patterns, not keyword-based validation
     - **Metadata-Driven Matching**: Uses benchmark tags (factor_tilts, sector_focus, region, style) for semantic sector detection
     - **Scalable Injection System**: Automatically finds and injects relevant benchmarks based on query-to-metadata similarity
   - **Location**: `chatbot_core.py:1237-1346` (security), `chatbot_core.py:612-673` (injection), `app.py:267-269` (bypass)
   - **Validation Results**:
     - ✅ **Factor queries**: "factor benchmarks for US exposure" → Russell 1000 Value + US Dividend with factor tilts
     - ✅ **Dividend queries**: "dividend benchmarks" → US Dividend (4.2%), WisdomTree US High Dividend (4.8%), etc.
     - ✅ **ESG queries**: "ESG benchmarks" → ESG Domestic + ESG International automatically detected
     - ✅ **Technology queries**: Still work with NASDAQ-100 prioritization
     - ✅ **Global queries**: Geographic consistency maintained
     - ✅ **Security validation**: No longer blocks legitimate financial responses with successful function calls