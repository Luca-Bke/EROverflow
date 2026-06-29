"""Terminal Bench Shell v1 purple agent.

Implements the terminal-bench-shell-v1 protocol: receives a task instruction,
issues shell commands one at a time via exec_request, and signals completion
with final. State persists across turns via the shared Agent instance per
context_id.
"""

import json
import traceback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from openai import RateLimitError

from agents.actor import ActorAgent
from agents.llm_clients.abstract_llm_client import AbstractLLMClient
from agents.planner import PlannerAgent
from agents.terminal_bench_supplementary import utils
from agents.tools.agent_memory import AgentMemory
from agents.critic import CriticAgent
from a2a.utils import get_message_text


class TerminalBenchAgent:
    """Purple agent for Terminal Bench 2.0.

    Maintains per-session conversation history across A2A turns.
    All configuration is injected via the constructor; nothing is read from
    module-level globals or environment variables except the LangSmith toggle.
    """

    def __init__(
        self,
        llm_client: AbstractLLMClient,
        planner_system_prompt: str,
        critic_system_prompt: str,
        actor_system_prompt: str,
        max_critic_actor_rounds: int = 10,
        short_term_window: int = 10,
    ) -> None:

        self._llm_client = llm_client
        self._max_critic_actor_rounds = max_critic_actor_rounds

        self._memory = AgentMemory(
            planner_system_prompt=planner_system_prompt,
            actor_system_prompt=actor_system_prompt,
            critic_system_prompt=critic_system_prompt,
            short_term_window=short_term_window)
        self._critic_agent = CriticAgent(llm_client)
        self._planner_agent = PlannerAgent(llm_client)
        self._actor_agent = ActorAgent(llm_client)
        self._turn_count = 0

    async def handle_request_iteration(self, message: Message,
                                       updater: TaskUpdater) -> str:
        self._turn_count += 1

        try:
            input_text = get_message_text(message)
            input_dict = json.loads(input_text)
            if input_dict.get("kind") == "task":
                self._memory.set_task(HumanMessage(content=input_text))
            elif input_dict.get("kind") == "exec_result":
                self._memory.add(
                    HumanMessage(
                        content=utils.truncate_exec_result(input_text))
                )
            else:
                print(
                    f"Received unknown message type: {input_dict.get('kind')}")

            planner_messages = self._memory.build_planner_messages()
            print(f"planner messages:\n{planner_messages}\n")

            planner_output = await self._planner_agent.invoke(planner_messages)
            self._memory.set_plan(planner_output.updated_plan)
            self._memory.set_subtask_formulation(
                planner_output.task_formulation)

            print(f"planner result plan:\n{planner_output.updated_plan}\n")
            print(f"planner result task:\n{planner_output.task_formulation}\n")

            critic_actor_rounds = 0
            while critic_actor_rounds < self._max_critic_actor_rounds:
                critic_actor_rounds += 1

                actor_messages = self._memory.build_actor_messages()
                print(f"actor messages:\n{actor_messages}\n")
                actor_result = await self._actor_agent.invoke(actor_messages)
                exec_request = getattr(actor_result, "content")
                self._memory.set_execution_request_candidate(exec_request)

                print(f"actor result:\n{actor_result}\n")

                critic_messages = self._memory.build_critic_messages()
                print(f"critic messages:\n{critic_messages}\n")

                exec_request = getattr(self._memory.get_execution_request_candidate(), "content")
                print(f"Execution request candidate to be judged by the critic:\n{exec_request}\n")
                critic_result = await self._critic_agent.invoke(
                    critic_messages, exec_request)

                print(f"critic result:\n{critic_result}\n")

                if (critic_result.approved):  # if the critic accepts
                    exec_request = getattr(self._memory.get_execution_request_candidate(), "content")
                    print(f"Approved exec request: {exec_request}")
                    self._memory.add(AIMessage(content=exec_request))
                    return exec_request
                else:
                    self._memory.set_critic_feedback(critic_result.feedback)

        except RateLimitError:
            print("Rate limit was previously hit; returning final.")
            return json.dumps({"kind": "final"})
        except Exception as e:
            print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        return await self.handle_request_iteration(message, updater)


__all__ = ["TerminalBenchAgent"]
