@echo off
echo ========================================
echo  Setting up Virtual Environment for
echo      Ben AI Enhanced UI (Windows)
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    echo Found Python: 
    python --version
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
        echo Found Python: 
        py --version
    ) else (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.8 or higher from python.org
        pause
        exit /b 1
    )
)

echo.

REM Create virtual environment
if not exist "venv\" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists
)

echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated!

echo.

REM Upgrade pip
echo Updating pip...
python -m pip install --upgrade pip

echo.

REM Install dependencies
echo Installing dependencies...
pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo Failed to install some dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo All dependencies installed successfully!

echo.

REM Check for .env file
if not exist ".env" (
    echo WARNING: No .env file found!
    echo.
    
    if exist ".env.example" (
        echo Creating .env from .env.example...
        copy .env.example .env >nul
        echo .env file created
        echo.
        echo IMPORTANT: Edit the .env file with your actual API keys:
        echo   1. OPENAI_API_KEY
        echo   2. PINECONE_API_KEY
        echo   3. PINECONE_ENV
    ) else (
        echo Please create a .env file with your API keys
    )
) else (
    echo .env file exists
)

echo.
echo =========================================
echo         Setup Complete!
echo =========================================
echo.
echo Your virtual environment is ready to use.
echo.
echo To start the application:
echo   Run: start_new_ui.bat
echo.
echo Or manually:
echo   1. venv\Scripts\activate.bat
echo   2. cd backend
echo   3. uvicorn app:app --reload
echo   4. Open frontend\index.html in browser
echo.
pause