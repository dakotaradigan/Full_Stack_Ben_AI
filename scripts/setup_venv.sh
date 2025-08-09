#!/bin/bash

# Virtual Environment Setup Script for macOS/Linux

echo "🔧 Setting up Python Virtual Environment for Ben AI Enhanced UI"
echo ""

# Check Python installation
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    if [ $? -eq 0 ]; then
        echo "✅ Virtual environment created successfully!"
    else
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated!"
    echo "   Python: $(which python)"
    echo "   Pip: $(which pip)"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo ""

# Install/Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r backend/requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
else
    echo "❌ Failed to install some dependencies"
    echo "   Please check the error messages above"
    exit 1
fi

echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found!"
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit the .env file with your actual API keys:"
    echo "   1. OPENAI_API_KEY"
    echo "   2. PINECONE_API_KEY"
    echo "   3. PINECONE_ENV"
else
    echo "✅ .env file exists"
fi

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "To start the application:"
echo "  ./start_new_ui.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  uvicorn app:app --reload"
echo ""
echo "Then open frontend/index.html in your browser"
echo ""