import importlib
import os
import types
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Dummy modules to avoid importing real dependencies
class DummyIndex:
    def query(self, *args, **kwargs):
        return types.SimpleNamespace(matches=[])

class DummyPineconeModule(types.ModuleType):
    class Pinecone:
        def __init__(self, *args, **kwargs):
            pass

        def list_indexes(self):
            return types.SimpleNamespace(names=lambda: ["benchmark-index"])

        def Index(self, name):
            return DummyIndex()

    def __init__(self):
        super().__init__("pinecone")

class DummyOpenAIModule(types.ModuleType):
    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    def __init__(self):
        super().__init__("openai")

class DummyTiktokenModule(types.ModuleType):
    def __init__(self):
        super().__init__("tiktoken")

    def encoding_for_model(self, model):
        return self.DummyEncoding()

    class DummyEncoding:
        def encode(self, text):
            return list(text.encode("utf-8"))

    get_encoding = encoding_for_model

@pytest.fixture(autouse=True)
def patch_external(monkeypatch):
    sys.modules["pinecone"] = DummyPineconeModule()
    sys.modules["openai"] = DummyOpenAIModule()
    sys.modules["tiktoken"] = DummyTiktokenModule()
    if "chatbot" in sys.modules:
        del sys.modules["chatbot"]
    yield
    sys.modules.pop("pinecone", None)
    sys.modules.pop("openai", None)
    sys.modules.pop("tiktoken", None)
    if "chatbot" in sys.modules:
        del sys.modules["chatbot"]


def test_filters_and_query(monkeypatch):
    chatbot = importlib.import_module("chatbot")

    captured = {}
    def fake_search_benchmarks(query, top_k=5, filters=None, include_dividend=False):
        captured["query"] = query
        captured["filters"] = filters
        return []

    monkeypatch.setattr(chatbot, "search_benchmarks", fake_search_benchmarks)

    chatbot.search_by_characteristics("S&P 500 Value")

    tags = chatbot.get_benchmark("S&P 500 Value")["tags"]
    expected_filters = {
        "region": {"$in": tags["region"]},
        "asset_class": {"$in": tags["asset_class"]},
        "style": {"$in": tags["style"]},
        "factor_tilts": {"$in": tags["factor_tilts"]},
        "sector_focus": {"$in": tags["sector_focus"]},
        "esg": {"$eq": tags["esg"]},
    }
    assert captured["filters"] == expected_filters
    query_parts = tags["region"] + tags["style"] + tags["factor_tilts"] + tags["sector_focus"]
    assert captured["query"] == " ".join(query_parts)
