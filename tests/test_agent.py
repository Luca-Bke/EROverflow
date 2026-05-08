"""Tests for the minimal LangGraph agent."""

import pytest
from agent import build_graph, State


def test_graph_compiles():
    app = build_graph()
    assert app is not None


def test_echo_response():
    app = build_graph()
    result = app.invoke({"messages": [{"role": "user", "content": "Hello"}]})
    messages = result["messages"]
    assert len(messages) == 2
    assert messages[-1].content == "Echo: Hello"
    assert messages[-1].type == "ai"


def test_multiple_invocations():
    app = build_graph()
    for text in ("ping", "test", "world"):
        result = app.invoke({"messages": [{"role": "user", "content": text}]})
        assert result["messages"][-1].content == f"Echo: {text}"
