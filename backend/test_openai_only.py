#!/usr/bin/env python3

import os
import sys

print("Testing OpenAI client initialization...")

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv('../.env')
    print("✓ Environment loaded")
except Exception as e:
    print(f"✗ Environment loading failed: {e}")

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
    print("✓ OpenAI API key found")
else:
    print("✗ OpenAI API key missing")
    sys.exit(1)

# Test basic OpenAI import
try:
    print("✓ Testing OpenAI import...")
    from openai import OpenAI
    print("✓ OpenAI imported successfully")
except Exception as e:
    print(f"✗ OpenAI import failed: {e}")
    sys.exit(1)

# Test client creation - MOST BASIC
try:
    print("✓ Testing minimal OpenAI client creation...")
    client = OpenAI()  # Should use OPENAI_API_KEY env var
    print("✓ OpenAI client created successfully (using env var)")
except Exception as e:
    print(f"✗ OpenAI client creation failed: {e}")
    print(f"  Error type: {type(e)}")
    print(f"  Error details: {str(e)}")

# Test client creation - EXPLICIT API KEY
try:
    print("✓ Testing explicit OpenAI client creation...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✓ OpenAI client created successfully (explicit key)")
except Exception as e:
    print(f"✗ OpenAI client creation failed: {e}")
    print(f"  Error type: {type(e)}")
    print(f"  Error details: {str(e)}")
    
# Test very basic functionality
try:
    print("✓ Testing OpenAI client functionality...")
    # Just test that client has expected attributes
    if hasattr(client, 'chat'):
        print("✓ Client has chat attribute")
    else:
        print("✗ Client missing chat attribute")
        
    if hasattr(client.chat, 'completions'):
        print("✓ Client has chat.completions attribute")
    else:
        print("✗ Client missing chat.completions attribute")
        
    print("✓ Basic OpenAI client test passed!")
    
except Exception as e:
    print(f"✗ OpenAI client functionality test failed: {e}")
    
print("\nOpenAI test complete.")