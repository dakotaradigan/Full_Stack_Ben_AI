import json
import os
from typing import List, Dict, Any

import pinecone
import openai

# Load the large system prompt from an external file for readability
with open("system_prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

EMBEDDING_MODEL = "text-embedding-3-small"

def embed(text: str) -> List[float]:
    resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    return resp["data"][0]["embedding"]

INDEX_NAME = "benchmark-index"
if INDEX_NAME not in pinecone.list_indexes():
    raise ValueError(
        f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first."
    )
index = pinecone.Index(INDEX_NAME)

with open("benchmarks.json", "r") as f:
    DATA = json.load(f)["benchmarks"]


def get_benchmark(name: str) -> Dict[str, Any] | None:
    for bench in DATA:
        if bench["name"].lower() == name.lower():
            return bench
    return None


def search_benchmarks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    vec = embed(query)
    res = index.query(vec, top_k=top_k, include_metadata=True)
    results = []
    for match in res.matches:
        bench = match.metadata
        results.append({
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "score": match.score,
        })
    return results


def get_minimum(name: str) -> Dict[str, Any]:
    bench = get_benchmark(name)
    if bench:
        return {"name": bench["name"], "account_minimum": bench["account_minimum"]}
    return {"error": f"Benchmark '{name}' not found"}


def blend_minimum(allocations: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_weight = sum(a.get("weight", 0) for a in allocations)
    if abs(total_weight - 1.0) > 1e-6:
        return {"error": f"weights sum to {total_weight}"}
    total = 0.0
    for a in allocations:
        bench = get_benchmark(a.get("name", ""))
        if not bench:
            return {"error": f"Benchmark '{a.get('name')}' not found"}
        total += bench["account_minimum"] * a["weight"]
    return {"blend_minimum": total}


FUNCTIONS = [
    {
        "name": "search_benchmarks",
        "description": "Search for similar benchmarks in the dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_minimum",
        "description": "Get minimum for a specific benchmark",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "blend_minimum",
        "description": "Calculate minimum for a blend of benchmarks",
        "parameters": {
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["name", "weight"],
                    },
                }
            },
            "required": ["allocations"],
        },
    },
]


def call_function(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_benchmarks":
        return {"results": search_benchmarks(**arguments)}
    if name == "get_minimum":
        return get_minimum(**arguments)
    if name == "blend_minimum":
        return blend_minimum(**arguments)
    return {"error": f"Unknown function {name}"}


def chat():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Hello! I'm here to assist with benchmark eligibility questions. How can I help you today?")
    while True:
        user = input("\nUser: ")
        if user.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=FUNCTIONS,
            function_call="auto",
        )
        msg = response["choices"][0]["message"]
        if msg.get("function_call"):
            func_name = msg["function_call"]["name"]
            args = json.loads(msg["function_call"]["arguments"] or "{}")
            result = call_function(func_name, args)
            messages.append(msg)
            messages.append({"role": "function", "name": func_name, "content": json.dumps(result)})
            follow = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
            )
            final = follow["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")
        else:
            final = msg.get("content", "")
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")


if __name__ == "__main__":
    chat()
