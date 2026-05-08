"""Minimal LangGraph agent."""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State) -> State:
    """Echo node: returns the last message as a response."""
    last = state["messages"][-1]
    response = {"role": "assistant", "content": f"Echo: {last.content}"}
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"messages": [{"role": "user", "content": "Hello, agent!"}]})
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")
