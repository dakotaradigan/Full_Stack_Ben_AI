import json
import os
from typing import List

import openai
import pinecone

EMBEDDING_MODEL = "text-embedding-3-small"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

INDEX_NAME = "benchmark-index"
DIMENSION = 1536


def embed(text: str) -> List[float]:
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return resp["data"][0]["embedding"]


def main() -> None:
    with open("benchmarks.json", "r") as f:
        data = json.load(f)["benchmarks"]

    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=DIMENSION)
    index = pinecone.Index(INDEX_NAME)

    if index.describe_index_stats().get("total_vector_count"):
        print(f"Index '{INDEX_NAME}' already populated.")
        return

    items = []
    for bench in data:
        vec = embed(bench["name"])
        items.append((bench["name"], vec, bench))

    for i in range(0, len(items), 100):
        index.upsert(items[i:i + 100])

    print(f"Upserted {len(items)} vectors to '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
