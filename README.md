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

`search_benchmarks` returns the most similar benchmarks to a query. The
optional `top_k` argument controls how many matches are returned (default is
**5**). Pass a larger value, such as `top_k=10`, for a broader search. It also
accepts an optional `filters` dictionary to narrow results by metadata fields
(e.g. `{ "region": "US", "pe_ratio": {"$gt": 20} }`).

Set the following environment variables before running either script: `PINECONE_API_KEY`, `PINECONE_ENV`, and `OPENAI_API_KEY`.
The scripts automatically retry failed requests to OpenAI and Pinecone so transient errors won't stop a run.

The chatbot appends a disclaimer every **third** assistant response. The frequency can be changed by modifying `DISCLAIMER_FREQUENCY` in `chatbot.py`.
