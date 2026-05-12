"""LangGraph agent with OpenRouter LLM integration and LangSmith tracing."""

import os
import uuid
from typing import Annotated, Optional
from typing_extensions import TypedDict

# Configure LangSmith for tracing and monitoring
os.environ.setdefault("LANGSMITH_TRACING", "true")

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import trace, Client


class State(TypedDict):
    messages: Annotated[list, add_messages]

def _create_llm() -> ChatOpenAI:
    """Create ChatOpenAI instance configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Check for LangSmith API key (optional but recommended for tracing)
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if not langsmith_key:
        print("⚠️  LANGSMITH_API_KEY not set. Tracing will not be saved to LangSmith.")
        print("   Set LANGSMITH_API_KEY and LANGSMITH_PROJECT to enable tracing.")
    
    return ChatOpenAI(
        model="openai/gpt-oss-120b:free",  # or specify a specific model like "openai/gpt-4"
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


def chatbot(state: State) -> State:
    """Chatbot node: uses OpenRouter LLM to generate responses."""
    llm = _create_llm()
    llm_messages = state["messages"]
    llm_messages.append(SystemMessage(content="You are a Nurse in an Emergency Room in an Hospital. You will be presented with several patients showing symptoms. You must triage them according to the Manchester Code. "))
    response = llm.invoke(llm_messages)
    return {"messages": [response]}


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


def run_chat():
    """Run an interactive chat session with the agent."""
    app = build_graph()
    messages = []
    session_id = str(uuid.uuid4())[:8]

    print("🤖 Chat with EROverflow Agent")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    print(f"📊 Session ID: {session_id} (visible in LangSmith)\n")
    
    # Create a single trace for the entire chat session
    with trace(
        name=f"chat_session_{session_id}",
        inputs={"session_id": session_id, "start_time": str(uuid.uuid4())},
        tags=["chat", "interactive"],
    ) as session_trace:
        while True:
            messages = []
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            try:
                # Create a span for each user query within the session trace
                with trace(
                    name="user_query",
                    inputs={"user_input": user_input, "message_count": len(messages)},
                    tags=["query"],
                ) as query_trace:
                    # Get agent response
                    result = app.invoke({"messages": messages})
                    
                    # Extract and display agent response
                    if result["messages"]:
                        agent_message = result["messages"][-1]
                        agent_content = agent_message.content if hasattr(agent_message, 'content') else str(agent_message)
                        print(f"\nAgent: {agent_content}\n")

                        # Add agent response to history
                        messages.append({"role": "assistant", "content": agent_content})
                        
                        # Log the outputs to the trace
                        query_trace.outputs = {
                            "agent_response": agent_content}
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    run_chat()
