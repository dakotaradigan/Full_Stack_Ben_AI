import json
import os
from typing import List, Dict, Any

import tiktoken

import logging
import time
from pinecone import Pinecone
from openai import OpenAI

from description_utils import build_semantic_description

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

# How often to append the disclaimer to assistant responses
DISCLAIMER_FREQUENCY = 3

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "YOUR_PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

EMBEDDING_MODEL = "text-embedding-3-small"

# Chat completion model and token limits
CHAT_MODEL = "gpt-3.5-turbo"
MAX_MODEL_TOKENS = 16000
TOKEN_MARGIN = 1000


def _with_retry(*, max_attempts: int = 3, **kwargs):
    """Call the OpenAI chat API with simple exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt == max_attempts:
                logging.error("OpenAI request failed after retries", exc_info=True)
                raise
            logging.warning(
                "OpenAI request failed (attempt %s/%s): %s",
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(2 ** attempt)

def embed(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str = CHAT_MODEL) -> int:
    """Return total token count for a list of chat messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if model.startswith("gpt-3.5-turbo"):
        tokens_per_message = 4
        tokens_per_name = -1
    elif model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            total_tokens += len(encoding.encode(str(value)))
            if key == "name":
                total_tokens += tokens_per_name
    total_tokens += 3
    return total_tokens


def trim_history(messages: List[Dict[str, Any]], limit: int = MAX_MODEL_TOKENS - TOKEN_MARGIN) -> bool:
    """Trim oldest user/assistant pairs until total tokens are under limit."""
    truncated = False
    while num_tokens_from_messages(messages) > limit and len(messages) > 1:
        # Find first user message after the system prompt
        start = next((i for i, m in enumerate(messages) if i > 0 and m["role"] == "user"), None)
        if start is None:
            break
        end = start + 1
        # remove messages until just before next user message
        while end < len(messages) and messages[end]["role"] != "user":
            end += 1
        del messages[start:end]
        truncated = True
    return truncated

INDEX_NAME = "benchmark-index"
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(
        f"Pinecone index '{INDEX_NAME}' does not exist. Run build_index.py first."
    )
index = pc.Index(INDEX_NAME)

with open("benchmarks.json", "r") as f:
    DATA = json.load(f)["benchmarks"]

for bench in DATA:
    if "description" not in bench:
        bench["description"] = build_semantic_description(bench)

# Build a mapping from lowercase benchmark name to the benchmark data for
# constant-time lookup when retrieving a benchmark by name.
BENCHMARK_MAP = {bench["name"].lower(): bench for bench in DATA}


def get_benchmark(name: str) -> Dict[str, Any] | None:
    """Return benchmark data by name using a pre-built map."""
    return BENCHMARK_MAP.get(name.lower())


def search_benchmarks(
    query: str,
    top_k: int = 5,
    filters: Dict[str, Any] | None = None,
    include_dividend: bool = False,
) -> List[Dict[str, Any]]:
    vec = embed(query)

    pinecone_filter: Dict[str, Any] | None = None
    if filters:
        pinecone_filter = {}
        for key, value in filters.items():
            if isinstance(value, dict) and any(k.startswith("$") for k in value):
                pinecone_filter[key] = value
            else:
                pinecone_filter[key] = {"$eq": value}

    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        **({"filter": pinecone_filter} if pinecone_filter else {}),
    )
    results = []
    for match in res.matches:
        bench = match.metadata
        item = {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "description": bench.get("description"),
            "score": match.score,
        }
        if include_dividend:
            item["dividend_yield"] = bench.get("dividend_yield")
        results.append(item)
    return results


def get_minimum(name: str, include_dividend: bool = False) -> Dict[str, Any]:
    bench = get_benchmark(name)
    if bench:
        result = {
            "name": bench["name"],
            "account_minimum": bench["account_minimum"],
            "description": bench.get("description"),
        }
        if include_dividend:
            result["dividend_yield"] = bench.get("fundamentals", {}).get("dividend_yield")
        return result
    return {"error": f"Benchmark '{name}' not found"}


def blend_minimum(allocations: List[Dict[str, Any]], include_dividend: bool = False) -> Dict[str, Any]:
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
    if include_dividend and has_yield:
        result["dividend_yield"] = weighted_yield
    return result


def search_viable_alternatives(
    query: str,
    portfolio_size: float,
    top_k: int = 5,
    filters: Dict[str, Any] | None = None,
    include_dividend: bool = False,
    max_iterations: int = 3
) -> List[Dict[str, Any]]:
    """
    Search for benchmark alternatives that meet portfolio size requirements.
    Will iterate through search results to find viable options.
    """
    viable_results = []
    current_k = top_k
    iteration = 0
    
    while len(viable_results) < top_k and iteration < max_iterations:
        # Search with increasing result count to find more options
        search_results = search_benchmarks(
            query=query,
            top_k=current_k * 2,  # Get more results each iteration
            filters=filters,
            include_dividend=include_dividend
        )
        
        # Filter results that meet portfolio size requirements
        for result in search_results:
            if result["account_minimum"] <= portfolio_size:
                # Avoid duplicates
                if not any(r["name"] == result["name"] for r in viable_results):
                    viable_results.append(result)
                    if len(viable_results) >= top_k:
                        break
        
        iteration += 1
        current_k += 5  # Increase search scope
    
    return viable_results[:top_k]


def search_by_characteristics(
    reference_benchmark: str,
    portfolio_size: float | None = None,
    top_k: int = 5,
    include_dividend: bool = False
) -> List[Dict[str, Any]]:
    """Search for benchmarks with similar characteristics to a reference
    benchmark.

    The search matches on key tags such as region, asset class, style,
    factor tilts, sector focus and ESG flag, using them both to construct
    Pinecone metadata filters and to build a textual query. This relies on the
    structured benchmark metadata rather than pure text similarity.
    """
    ref_bench = get_benchmark(reference_benchmark)
    if not ref_bench:
        return []
    
    ref_tags = ref_bench.get("tags", {})
    ref_fundamentals = ref_bench.get("fundamentals", {})
    
    # Build filters based on reference benchmark characteristics
    filters = {}
    
    # Match on key characteristics
    if ref_tags.get("region"):
        filters["region"] = {"$in": ref_tags["region"]}

    if ref_tags.get("asset_class"):
        filters["asset_class"] = {"$in": ref_tags["asset_class"]}

    if ref_tags.get("style"):
        filters["style"] = {"$in": ref_tags["style"]}

    if ref_tags.get("factor_tilts"):
        filters["factor_tilts"] = {"$in": ref_tags["factor_tilts"]}

    if ref_tags.get("sector_focus"):
        filters["sector_focus"] = {"$in": ref_tags["sector_focus"]}

    if ref_tags.get("esg") is not None:
        filters["esg"] = {"$eq": ref_tags["esg"]}
    
    # Create a broad search query combining characteristics
    query_parts = []
    if ref_tags.get("region"):
        query_parts.extend(ref_tags["region"])
    if ref_tags.get("style"):
        query_parts.extend(ref_tags["style"])
    if ref_tags.get("factor_tilts"):
        query_parts.extend(ref_tags["factor_tilts"])
    if ref_tags.get("sector_focus"):
        query_parts.extend(ref_tags["sector_focus"])
    
    query = " ".join(query_parts) if query_parts else reference_benchmark
    
    # Search with characteristics-based filters
    if portfolio_size is not None:
        results = search_viable_alternatives(
            query=query,
            portfolio_size=portfolio_size,
            top_k=top_k,
            filters=filters,
            include_dividend=include_dividend
        )
    else:
        results = search_benchmarks(
            query=query,
            top_k=top_k,
            filters=filters,
            include_dividend=include_dividend
        )
    
    # Filter out the reference benchmark itself
    results = [r for r in results if r["name"] != reference_benchmark]
    
    return results[:top_k]


FUNCTIONS = [
    {
        "name": "search_benchmarks",
        "description": "Search for similar benchmarks in the dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters. Example: {\"pe_ratio\": {\"$gt\": 20}, \"region\": \"US\"}",
                },
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in results",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_minimum",
        "description": "Get minimum for a specific benchmark",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in result",
                },
            },
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
                },
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in result",
                },
            },
            "required": ["allocations"],
        },
    },
    {
        "name": "search_viable_alternatives",
        "description": "Search for benchmark alternatives that meet portfolio size requirements",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "portfolio_size": {"type": "number"},
                "top_k": {"type": "integer", "default": 5},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters. Example: {\"pe_ratio\": {\"$gt\": 20}, \"region\": \"US\"}",
                },
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in results",
                },
            },
            "required": ["query", "portfolio_size"],
        },
    },
    {
        "name": "search_by_characteristics",
        "description": "Search for benchmarks with similar characteristics to a reference benchmark",
        "parameters": {
            "type": "object",
            "properties": {
                "reference_benchmark": {"type": "string"},
                "portfolio_size": {"type": "number"},
                "top_k": {"type": "integer", "default": 5},
                "include_dividend": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include dividend_yield in results",
                },
            },
            "required": ["reference_benchmark"],
        },
    },
]


def call_function(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_benchmarks":
        return {
            "results": search_benchmarks(
                query=arguments.get("query", ""),
                top_k=arguments.get("top_k", 5),
                filters=arguments.get("filters"),
                include_dividend=arguments.get("include_dividend", False),
            )
        }
    if name == "get_minimum":
        return get_minimum(
            name=arguments.get("name", ""),
            include_dividend=arguments.get("include_dividend", False),
        )
    if name == "blend_minimum":
        return blend_minimum(
            allocations=arguments.get("allocations", []),
            include_dividend=arguments.get("include_dividend", False),
        )
    if name == "search_viable_alternatives":
        return {
            "results": search_viable_alternatives(
                query=arguments.get("query", ""),
                portfolio_size=arguments.get("portfolio_size", 0.0),
                top_k=arguments.get("top_k", 5),
                filters=arguments.get("filters"),
                include_dividend=arguments.get("include_dividend", False),
            )
        }
    if name == "search_by_characteristics":
        return {
            "results": search_by_characteristics(
                reference_benchmark=arguments.get("reference_benchmark", ""),
                portfolio_size=arguments.get("portfolio_size", None),
                top_k=arguments.get("top_k", 5),
                include_dividend=arguments.get("include_dividend", False),
            )
        }
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
        if trim_history(messages):
            print("[Notice: conversation history truncated to fit token limit]")
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[{"type": "function", "function": func} for func in FUNCTIONS],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            # Add the assistant message with tool calls first
            messages.append({"role": "assistant", "content": None, "tool_calls": msg.tool_calls})
            if trim_history(messages):
                print("[Notice: conversation history truncated to fit token limit]")

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
                if trim_history(messages):
                    print("[Notice: conversation history truncated to fit token limit]")

            # Now get the final response
            follow = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
            )
            final = follow.choices[0].message.content
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")
        else:
            final = msg.content or ""
            resp_count += 1
            if resp_count % DISCLAIMER_FREQUENCY == 0:
                final = f"{final}\n\n{DISCLAIMER_TEXT}"
            messages.append({"role": "assistant", "content": final})
            print(f"\nAssistant: {final}")


if __name__ == "__main__":
    chat()
