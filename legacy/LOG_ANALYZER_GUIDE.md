# 📊 Log Analyzer Usage Guide

The `log_analyzer.py` script provides comprehensive monitoring and analysis of your chatbot's usage, security events, and performance.

## 🚀 **Quick Start**

```bash
# Basic analysis of chatbot.log
python3 log_analyzer.py

# Analyze specific log file  
python3 log_analyzer.py --log-file /path/to/your/chatbot.log

# Get cost analysis for last 30 days
python3 log_analyzer.py --cost-days 30

# Export results to JSON
python3 log_analyzer.py --export --export-file my_report.json
```

## 📈 **What It Analyzes**

### **1. Usage Statistics**
- **Sessions**: Total sessions, duration, queries per session
- **API Usage**: Token consumption, cost tracking, rate limiting
- **Function Calls**: Most popular functions, usage patterns
- **Peak Hours**: When your chatbot is most active

### **2. Security Monitoring**
- **Prompt Injection Attempts**: Blocked malicious inputs
- **Rate Limiting**: When limits are exceeded
- **Cost Overruns**: Budget protection triggers
- **Circuit Breaker Events**: API failure protections

### **3. Error Tracking** 
- **API Failures**: OpenAI/Pinecone connectivity issues
- **System Errors**: Application crashes, data issues
- **Performance Issues**: Timeouts, slow responses

### **4. Cost Analysis**
- **Daily/Weekly/Monthly Costs**: Track spending over time
- **Token Usage**: Understand consumption patterns
- **Cost Per Query**: Efficiency metrics
- **Budget Alerts**: When limits are approached

## 📋 **Sample Report Output**

```
🤖 CHATBOT USAGE ANALYSIS REPORT
============================================================

📊 OVERALL STATISTICS:
   Total Sessions: 4
   Total Queries: 9
   Total Tokens Used: 8,120
   Total Cost: $0.0131
   Average Session Duration: 4.2 minutes
   Average Tokens per Query: 902
   Average Cost per Query: $0.0015

🔧 MOST USED FUNCTIONS:
   search_by_characteristics: 3 times (33.3%)
   get_minimum: 3 times (33.3%)
   search_benchmarks: 2 times (22.2%)
   blend_minimum: 1 times (11.1%)

⏰ PEAK USAGE:
   Peak Hour: 14:00
   Hourly Distribution:
   09:00 [ 3] ███
   10:00 [ 2] ██
   14:00 [ 4] ████
   16:00 [ 2] ██

🛡️ SECURITY EVENTS (5 total):
   PROMPT_INJECTION_ATTEMPT: 2 events
   RATE_LIMIT_EXCEEDED: 1 events
   CIRCUIT_BREAKER_TRIGGERED: 1 events
   COST_LIMIT_EXCEEDED: 1 events

💰 COST ANALYSIS (Last 7 days)
   Total Cost: $45.67
   Total Tokens: 312,450
   Average Daily Cost: $6.52
   Daily Breakdown:
     2024-01-15: $12.34 (85,230 tokens)
     2024-01-14: $8.91 (61,180 tokens)
     2024-01-13: $15.42 (106,040 tokens)
```

## 🔧 **Command Line Options**

```bash
python3 log_analyzer.py [options]

Options:
  --log-file, -f         Path to log file (default: chatbot.log)
  --cost-days, -d        Days for cost analysis (default: 7)
  --export, -e           Export results to JSON file
  --export-file          JSON export filename (default: usage_report.json)
  --help, -h             Show help message
```

## 📁 **Output Files**

### **Console Report**
- Real-time analysis displayed in terminal
- Color-coded security events and errors
- Visual charts for usage patterns

### **JSON Export** (`usage_report.json`)
```json
{
  "generated_at": "2024-01-15T16:30:00",
  "usage_stats": {
    "total_sessions": 25,
    "total_queries": 187,
    "total_cost": 45.67,
    "most_used_functions": {...}
  },
  "security_events": [
    {
      "timestamp": "2024-01-15T09:10:15",
      "type": "PROMPT_INJECTION_ATTEMPT",
      "severity": "MEDIUM",
      "details": "Potentially malicious input detected"
    }
  ]
}
```

## 🛡️ **Security Event Types**

| Event Type | Description | Severity |
|------------|-------------|----------|
| `PROMPT_INJECTION_ATTEMPT` | Malicious input blocked | MEDIUM |
| `RATE_LIMIT_EXCEEDED` | Too many requests | MEDIUM |
| `COST_LIMIT_EXCEEDED` | Budget limit hit | HIGH |
| `CIRCUIT_BREAKER_TRIGGERED` | API failures detected | HIGH |

## 📊 **Monitoring Best Practices**

### **Daily Monitoring**
```bash
# Quick daily check
python3 log_analyzer.py --cost-days 1

# Look for security issues
grep "SECURITY" chatbot.log | tail -10
```

### **Weekly Reviews**
```bash
# Comprehensive weekly analysis
python3 log_analyzer.py --cost-days 7 --export

# Compare with previous week
python3 log_analyzer.py --cost-days 14
```

### **Cost Alerts**
```bash
# Check if approaching monthly budget
python3 log_analyzer.py --cost-days 30 | grep "Total Cost"

# Set up cron job for daily cost monitoring
0 9 * * * cd /path/to/chatbot && python3 log_analyzer.py --cost-days 1 | mail -s "Daily Chatbot Costs" admin@company.com
```

## 🔍 **Troubleshooting**

### **Common Issues**

**"Log file not found"**
```bash
# Check if chatbot has been run and created logs
ls -la chatbot.log
# If missing, run the enhanced chatbot first
python3 chatbot_enhanced.py
```

**"No cost data found"**
- Make sure the enhanced chatbot is being used (not the original)
- Check that API calls are being made successfully
- Verify log format matches expected pattern

**"Security events not detected"**
- Try some test inputs with prompt injection attempts
- Check that input sanitization is working
- Verify log level is set to INFO or higher

### **Advanced Usage**

**Custom Analysis Scripts**
```python
from log_analyzer import LogAnalyzer

# Create custom analysis
analyzer = LogAnalyzer("my_chatbot.log")
analyzer.parse_log_file()

# Get raw data for custom processing
stats = analyzer.generate_usage_stats()
print(f"Functions used: {stats.most_used_functions}")
```

**Integration with Monitoring Systems**
```bash
# Export to monitoring dashboard
python3 log_analyzer.py --export
curl -X POST -H "Content-Type: application/json" \
     -d @usage_report.json \
     https://your-monitoring-system.com/api/chatbot-metrics
```

## 🎯 **Key Metrics to Watch**

### **🚨 Red Flags**
- **High security event count**: Indicates potential attacks
- **Frequent circuit breaker trips**: API reliability issues  
- **Escalating costs**: Possible abuse or inefficient usage
- **Low function success rate**: Performance problems

### **✅ Green Indicators**
- **Steady usage patterns**: Consistent user engagement
- **Low error rates**: System stability
- **Cost within budget**: Efficient resource usage
- **No security events**: Proper input validation

### **📈 Growth Metrics**
- **Increasing session count**: Growing user base
- **Higher query complexity**: More sophisticated usage
- **Stable cost per query**: Improving efficiency
- **Diverse function usage**: Feature adoption

---

**🎉 The log analyzer provides everything you need to monitor, secure, and optimize your chatbot deployment!**