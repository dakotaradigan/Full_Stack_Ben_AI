@echo off
echo ========================================
echo    Ben AI Enhanced UI Startup Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        echo Trying with py command...
        py -m venv venv
        if errorlevel 1 (
            echo Failed to create virtual environment. Please ensure Python is installed.
            pause
            exit /b 1
        )
    )
    echo Virtual environment created successfully!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated!
echo.

REM Check if dependencies are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies!
        pause
        exit /b 1
    )
    echo Dependencies installed successfully!
    echo.
)

REM Check for .env file
if not exist ".env" (
    echo WARNING: .env file not found!
    echo.
    echo Please create a .env file with your API keys:
    echo   OPENAI_API_KEY=your_key_here
    echo   PINECONE_API_KEY=your_key_here  
    echo   PINECONE_ENV=your_env_here
    echo.
    echo You can copy .env.example to .env and edit it.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        pause
        exit /b 1
    )
)

REM Start backend server
echo Starting backend server...
cd backend
start /B cmd /c "uvicorn app:app --reload --host 0.0.0.0 --port 8000"
cd ..

REM Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Check if backend is running
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 (
    echo Backend may not be fully started yet.
    echo Please check http://localhost:8000/api/health manually.
) else (
    echo Backend is running!
)

REM Open frontend in browser
echo Opening frontend in browser...
start "" "%cd%\frontend\index.html"

echo.
echo ========================================
echo    Ben AI Enhanced UI is running!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend UI: file:///%cd:\=/%/frontend/index.html
echo.
echo API Documentation: http://localhost:8000/docs
echo.
echo Press any key to stop the server...
pause >nul

REM Kill the backend process
taskkill /F /FI "WINDOWTITLE eq uvicorn*" >nul 2>&1
echo.
echo Server stopped.
pause