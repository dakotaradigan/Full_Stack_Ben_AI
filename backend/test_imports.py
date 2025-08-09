#!/usr/bin/env python3

import sys
import os

print("Testing imports for Ben AI backend...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print()

# Test basic imports
try:
    print("✓ Testing basic Python modules...")
    import json
    import logging
    from datetime import datetime
    print("✓ Basic modules imported successfully")
except Exception as e:
    print(f"✗ Basic modules failed: {e}")
    exit(1)

# Test FastAPI
try:
    print("✓ Testing FastAPI...")
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
except Exception as e:
    print(f"✗ FastAPI failed: {e}")
    print("  Run: pip install fastapi")
    exit(1)

# Test other dependencies
dependencies = [
    ("uvicorn", "uvicorn"),
    ("openai", "openai"),
    ("pinecone", "pinecone"),
    ("tiktoken", "tiktoken"),
    ("pydantic", "pydantic")
]

for dep_name, import_name in dependencies:
    try:
        print(f"✓ Testing {dep_name}...")
        __import__(import_name)
        print(f"✓ {dep_name} imported successfully")
    except Exception as e:
        print(f"✗ {dep_name} failed: {e}")
        print(f"  Run: pip install {dep_name}")

print()

# Test environment variables
print("✓ Testing environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv('../.env')
    
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENV')
    
    if openai_key and openai_key != 'your_openai_api_key_here':
        print("✓ OpenAI API key found")
    else:
        print("✗ OpenAI API key missing or placeholder")
        
    if pinecone_key and pinecone_key != 'your_pinecone_api_key_here':
        print("✓ Pinecone API key found")
    else:
        print("✗ Pinecone API key missing or placeholder")
        
    if pinecone_env:
        print(f"✓ Pinecone environment: {pinecone_env}")
    else:
        print("✗ Pinecone environment missing")
        
except Exception as e:
    print(f"✗ Environment variables test failed: {e}")

print()

# Test config files
print("✓ Testing config files...")
config_files = [
    "config/system_prompt.txt",
    "config/benchmarks.json"
]

for file_path in config_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path} exists")
    else:
        print(f"✗ {file_path} missing")

print()

# Test chatbot_core import
try:
    print("✓ Testing chatbot_core import...")
    import chatbot_core
    print("✓ chatbot_core imported successfully")
except Exception as e:
    print(f"✗ chatbot_core failed: {e}")
    print(f"  Error details: {str(e)}")

print()

# Test FastAPI app creation
try:
    print("✓ Testing FastAPI app creation...")
    from app import app
    print("✓ FastAPI app created successfully")
    print("\n🎉 All tests passed! Backend should work.")
except Exception as e:
    print(f"✗ FastAPI app creation failed: {e}")
    print(f"  Error details: {str(e)}")
    print("\n❌ Backend has issues that need to be fixed.")