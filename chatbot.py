import json
import os
from typing import List, Dict, Any

from pinecone import Pinecone
from openai import OpenAI

# Load the large system prompt from an external file for readability
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Extract the disclaimer text from the system prompt so it can be
# appended to responses in the chat loop. The line in the prompt looks
# like:
# "- Provide disclaimer every 3-4 interactions: *\u201cThis assistant ... authority.\u201d*"
import re
match = re.search(
    r"disclaimer every 3-4 interactions:\s*\*?[\"\u201c](.+?)[\"\u201d]\*?",
    SYSTEM_PROMPT,
    flags=re.IGNORECASE,
)
DISCLAIMER_TEXT = (
    match.group(1).strip()
    if match
    else "This assistant provides benchmark eligibility guidance only. "
    "No investment advice or account approval authority."
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

EMBEDDING_MODEL = "text-embedding-3-small"

def embed(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

INDEX_NAME = "benchmark-index"
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(
        f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first."
    )
index = pc.Index(INDEX_NAME)

with open("benchmarks.json", "r") as f:
    DATA = json.load(f)["benchmarks"]

# Build a mapping from lowercase benchmark name to the benchmark data for
# constant-time lookup when retrieving a benchmark by name.
BENCHMARK_MAP = {bench["name"].lower(): bench for bench in DATA}


def get_benchmark(name: str) -> Dict[str, Any] | None:
    """Return benchmark data by name using a pre-built map."""
    return BENCHMARK_MAP.get(name.lower())


def search_benchmarks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    vec = embed(query)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    results = []
    for match in res.matches:
        bench = match.metadata
        results.append({
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "dividend_yield": bench.get("dividend_yield"),
            "score": match.score,
        })
    return results


def get_minimum(name: str) -> Dict[str, Any]:
    bench = get_benchmark(name)
    if bench:
        return {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "dividend_yield": bench.get("fundamentals", {}).get("dividend_yield"),
        }
    return {"error": f"Benchmark '{name}' not found"}


def blend_minimum(allocations: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_weight = sum(a.get("weight", 0) for a in allocations)
    if abs(total_weight - 1.0) > 1e-6:
        return {"error": f"weights sum to {total_weight}"}
    total = 0.0
    weighted_yield = 0.0
    has_yield = True
    for a in allocations:
        bench = get_benchmark(a.get("name", ""))
        if not bench:
            return {"error": f"Benchmark '{a.get('name')}' not found"}
        total += bench["account_minimum"] * a["weight"]
        dy = bench.get("fundamentals", {}).get("dividend_yield")
        if dy is None:
            has_yield = False
        else:
            weighted_yield += dy * a["weight"]
    result = {"blend_minimum": total}
    if has_yield:
        result["dividend_yield"] = weighted_yield
    return result


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
    resp_count = 0
    while True:
        user = input("\nUser: ")
        if user.lower() in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=[{"type": "function", "function": func} for func in FUNCTIONS],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            # Add the assistant message with tool calls first
            messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})

            # Handle each tool call individually
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                result = call_function(func_name, args)

                # Add individual tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Now get the final response
            follow = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            final = follow.choices[0].message.content
            resp_count += 1
            if resp_count % 4 == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")
        else:
            final = msg.content or ""
            resp_count += 1
            if resp_count % 4 == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")


if __name__ == "__main__":
    chat()
