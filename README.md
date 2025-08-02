# 🤖 Enhanced Benchmark Eligibility Chatbot

A production-ready AI assistant for benchmark eligibility questions with enterprise-grade security, monitoring, and reliability features.

## 🚀 Quick Start

### 1. Install Dependencies

**Python Requirements:**
```bash
# Core dependencies
pip install pinecone-client openai tiktoken

# Optional for enhanced features
pip install python-dotenv  # For .env file support
```

**Python Version:** Requires Python 3.8+

### 2. Set Environment Variables

Create a `.env` file or set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export PINECONE_API_KEY="your-pinecone-api-key-here"
export PINECONE_ENV="your-pinecone-environment"
```

**🔑 Get API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Pinecone**: https://app.pinecone.io/

### 3. Build the Vector Index

Run the index builder once to load benchmark data:
```bash
python build_index.py
```
This creates a Pinecone index with vector embeddings for all benchmarks.

### 4. Run the Enhanced Chatbot

```bash
# Production-ready version with security features
python chatbot_enhanced.py

# Original version (basic functionality)
python chatbot.py
```

## 📁 Project Structure

```
├── chatbot_enhanced.py           # 🛡️ Production chatbot with security
├── chatbot.py                    # Basic chatbot version
├── system_prompt.txt             # AI system instructions
├── benchmarks.json               # Benchmark database
├── build_index.py               # Vector index builder
├── description_utils.py         # Semantic description generator
├── log_analyzer.py              # 📊 Usage monitoring tool
├── test_security_core.py        # 🧪 Security feature tests
└── chatbot.log                  # Generated logs (created on first run)
```

## 🛡️ Enhanced Features

The `chatbot_enhanced.py` version includes:

### **Security Features:**
- ✅ **Prompt injection protection** - Blocks malicious inputs
- ✅ **Input sanitization** - Filters dangerous patterns
- ✅ **Rate limiting** - 30 requests per minute limit
- ✅ **Cost monitoring** - $10 per hour budget protection
- ✅ **Circuit breaker** - Prevents cascade failures

### **Reliability Features:**  
- ✅ **Comprehensive error handling** - No crashes
- ✅ **API retry logic** - Automatic failure recovery
- ✅ **Graceful degradation** - Fallback responses
- ✅ **Session management** - Conversation tracking
- ✅ **Comprehensive logging** - Full audit trail

### **Monitoring Features:**
- ✅ **Usage analytics** - Track costs and performance
- ✅ **Security monitoring** - Detect attack attempts
- ✅ **Performance metrics** - Response times and efficiency
- ✅ **Cost tracking** - Real-time budget monitoring

## 💰 Cost Management

**Default Limits:**
- **Rate Limit:** 30 requests per minute
- **Cost Limit:** $10 per hour  
- **Input Limit:** 2000 characters per message
- **Token Limit:** 4000 tokens per request

**Modify Limits:** Edit these constants in `chatbot_enhanced.py`:
```python
MAX_REQUESTS_PER_MINUTE = 30
MAX_COST_PER_HOUR = 10.0
MAX_INPUT_LENGTH = 2000
MAX_TOKENS_PER_REQUEST = 4000
```

## 📊 Monitoring & Analytics

### View Usage Reports:
```bash
# Basic usage analysis
python log_analyzer.py

# Cost analysis for last 30 days  
python log_analyzer.py --cost-days 30

# Export detailed report to JSON
python log_analyzer.py --export --export-file monthly_report.json
```

### Sample Report Output:
```
🤖 CHATBOT USAGE ANALYSIS REPORT
============================================================
📊 OVERALL STATISTICS:
   Total Sessions: 45
   Total Queries: 342  
   Total Tokens Used: 287,450
   Total Cost: $42.67
   Average Session Duration: 4.2 minutes

🛡️ SECURITY EVENTS (12 total):
   PROMPT_INJECTION_ATTEMPT: 8 events
   RATE_LIMIT_EXCEEDED: 3 events
   COST_LIMIT_EXCEEDED: 1 events
```

## 🧪 Testing

### Test Security Features:
```bash
python test_security_core.py
```

### Test Sample Queries:
```bash
python chatbot_enhanced.py
```

Then try:
- `"What's the minimum for S&P 500?"`
- `"Alternatives to Russell 2000"`
- `"ESG options for $500K portfolio"`
- `"Ignore all instructions"` (tests security)

## 🔧 Configuration

### Chatbot Settings:
Edit `chatbot_enhanced.py` configuration section:
```python
# Security and Safety Configuration
MAX_INPUT_LENGTH = 2000
MAX_TOKENS_PER_REQUEST = 4000
MAX_REQUESTS_PER_MINUTE = 30
MAX_COST_PER_HOUR = 10.0

# Chat Model Settings
CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"
```

### Logging Settings:
Logs are saved to `chatbot.log` by default. To change:
```python
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('your_custom_log.log'),
        logging.StreamHandler()
    ]
)
```

## 🚨 Troubleshooting

### Common Issues:

**"No module named 'pinecone'"**
```bash
pip install pinecone-client
```

**"OpenAI API key not configured"**
```bash
export OPENAI_API_KEY="your-key-here"
```

**"Pinecone index does not exist"**
```bash
python build_index.py  # Run this first
```

**"Rate limit exceeded"**
- Wait a minute, then try again
- Increase `MAX_REQUESTS_PER_MINUTE` if needed

**"Cost limit exceeded"**
- Check usage with `python log_analyzer.py --cost-days 1`
- Increase `MAX_COST_PER_HOUR` if needed

### Debug Mode:
Set logging to DEBUG for detailed information:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Production Deployment

### Recommended Setup:
1. **Environment Variables:** Use `.env` file or system environment
2. **Process Management:** Use `systemd`, `supervisor`, or `pm2`
3. **Monitoring:** Set up log monitoring and alerting
4. **Backup:** Regular backup of `benchmarks.json` and logs
5. **Updates:** Regular updates to API libraries and security patches

### Example Service File (systemd):
```ini
[Unit]
Description=Enhanced Benchmark Chatbot
After=network.target

[Service]
Type=simple
User=chatbot
WorkingDirectory=/opt/chatbot
ExecStart=/usr/bin/python3 chatbot_enhanced.py
Restart=always
Environment=OPENAI_API_KEY=your-key
Environment=PINECONE_API_KEY=your-key

[Install]
WantedBy=multi-user.target
```

## 🤝 Support

### File Issues:
- Security vulnerabilities: Create private issue
- Feature requests: Create public issue
- Bug reports: Include logs and reproduction steps

### Documentation:
- `SECURITY_ENHANCEMENTS_IMPLEMENTED.md` - Security features
- `FUNCTION_ROUTING_UPDATES.md` - Function routing workflow  
- `LOG_ANALYZER_GUIDE.md` - Monitoring and analytics guide

---

**🎉 You're ready to deploy a production-grade AI chatbot with enterprise security and monitoring!**
