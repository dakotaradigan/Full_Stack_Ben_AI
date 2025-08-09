#!/bin/bash
# 🧪 Quick Test Environment Setup Script

echo "🤖 Enhanced Chatbot - Test Environment Setup"
echo "============================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version || {
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install pinecone-client openai tiktoken python-dotenv

# Check for required files
echo "📁 Checking required files..."
required_files=("benchmarks.json" "system_prompt.txt" "description_utils.py" "chatbot_enhanced.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check environment variables
echo "🔑 Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
    echo "   Or create a .env file with: OPENAI_API_KEY=your-key-here"
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo "⚠️  PINECONE_API_KEY not set"
    echo "   Set it with: export PINECONE_API_KEY='your-key-here'"
    echo "   Or create a .env file with: PINECONE_API_KEY=your-key-here"
fi

if [ -z "$PINECONE_ENV" ]; then
    echo "⚠️  PINECONE_ENV not set"
    echo "   Set it with: export PINECONE_ENV='your-environment'"
    echo "   Or create a .env file with: PINECONE_ENV=your-environment"
fi

# Test security features
echo "🧪 Testing security features..."
python3 test_security_core.py || {
    echo "⚠️  Security tests had issues (this is expected without API keys)"
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Set your API keys (see warnings above)"
echo "2. Build the index: python3 build_index.py"
echo "3. Run the chatbot: python3 chatbot_enhanced.py"
echo ""
echo "📊 Monitor usage: python3 log_analyzer.py"
echo "🧪 Test security: python3 test_security_core.py"
echo ""
echo "📖 Full documentation: README.md"