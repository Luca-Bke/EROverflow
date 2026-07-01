
import json
import re
from dataclasses import dataclass
from typing import Any, override

from a2a.types import Message
from langchain_core.messages import BaseMessage, HumanMessage
from langsmith import traceable

from agents.abstract_agent import AbstractAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.terminal_bench_supplementary import utils
from agents.tools.agent_memory import AgentMemory

@dataclass
class PlannerOutput:
    updated_plan: list[str]
    task_formulation: str


class PlannerAgent(AbstractAgent):

    def __init__(self, llm_client: AbstractLLMClient) -> None:
        super().__init__(llm_client)

    @staticmethod
    def _split_agent_response(response: BaseMessage) -> PlannerOutput:
        raw = getattr(response, "content", str(response))

        # Strip markdown code fences the LLM may add despite instructions
        stripped = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        data: dict[str, Any] | None = None
        try:
            data = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if not isinstance(data, dict):
            return PlannerOutput(updated_plan=[], task_formulation=raw.strip())

        plan = data.get("updated_plan", [])
        if not isinstance(plan, list):
            plan = [str(plan)]

        task = str(data.get("task_formulation", ""))
        return PlannerOutput(updated_plan=plan, task_formulation=task)

    @override
    @traceable(name="Planner", run_type="chain")
    async def invoke(self, messages: list[BaseMessage]) -> PlannerOutput:  # type: ignore[override]
        response = await self._llm_client.invoke_async(messages)
        return self._split_agent_response(response)

        