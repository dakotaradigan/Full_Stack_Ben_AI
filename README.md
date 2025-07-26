# AIPDM_Test

## Build the Pinecone Index

Install dependencies:
```bash
pip install pinecone-client openai
```

Run the index builder once to load benchmark data (it will skip if the index already contains vectors):
```bash
python build_index.py
```

## Chatbot Usage

After the index is built, start the chatbot:
```bash
python chatbot.py
```

`search_benchmarks` accepts an optional `filters` dictionary to narrow results
by metadata fields (e.g. `{ "region": "US", "pe_ratio": {"$gt": 20} }`).

Set the following environment variables before running either script: `PINECONE_API_KEY`, `PINECONE_ENV`, and `OPENAI_API_KEY`.
The scripts automatically retry failed requests to OpenAI and Pinecone so transient errors won't stop a run.

Both the CLI and web chatbot append a disclaimer every **third** assistant response. The frequency can be changed by modifying `DISCLAIMER_FREQUENCY` in `chatbot.py`.

## Web UI

A simple Flask application provides a modern chat interface.
Install additional dependencies and run the server:
```bash
pip install flask flask-session
export FLASK_SECRET_KEY=some-long-random-string
python app.py
```
The `FLASK_SECRET_KEY` variable controls the session key. Set it before running the server for better security.
Open your browser to `http://localhost:5000` to start chatting.
