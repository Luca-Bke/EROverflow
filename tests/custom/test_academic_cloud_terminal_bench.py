# import json
# import subprocess
# from uuid import uuid4

# import pytest

# from messenger import send_message, Messenger
# from test_agent import send_text_message, validate_event
# from a2a.types import Message


# @pytest.fixture(scope="session")
# def messenger(request):
#     """Messenger fixture for sending messages to the agent."""
#     return Messenger()


# def _extract_message_text(msg: Message) -> str | None:
#     dumped = msg.model_dump()
#     parts = dumped.get("parts") or []
#     if not parts:
#         return None

#     first_part = parts[0]
#     if isinstance(first_part, dict):
#         return first_part.get("text") or first_part.get("content")

#     return None


# def _parse_exec_request(exec_request: dict[str, object]) -> dict[str, object] | None:
#     return [exec_request.get("command", ""), int(exec_request.get("timeout", 30) or 30)]


# def _execute_command(command: str, timeout: int) -> dict[str, object]:

#     try:
#         # be aware that this might execute differently on windows and linux
#         # => therefore i addded "in cmd" to the instruction
#         result = subprocess.run(
#             command,
#             shell=True,
#             capture_output=True,
#             text=True,
#             timeout=timeout,
#         )

#         stdout = result.stdout.replace("\r\n", "\n")
#         stderr = result.stderr.replace("\r\n", "\n")
#         return {
#             "kind": "exec_result",
#             "stdout": stdout,
#             "stderr": stderr,
#             "exit_code": result.returncode,
#         }
#     except subprocess.TimeoutExpired as exc:
#         stdout = (exc.stdout or "").replace("\r\n", "\n")
#         stderr = (exc.stderr or "").replace("\r\n", "\n")
#         return {
#             "kind": "exec_result",
#             "stdout": stdout,
#             "stderr": stderr,
#             "exit_code": -1,
#         }


# @pytest.mark.asyncio
# # allows for continous polling or something,
# @pytest.mark.parametrize("streaming", [False])
# # but really messes up the test output, so we'll just test non-streaming for now
# async def test_academic_cloud_terminal_bench(agent, streaming, messenger):

#     hello_world_complaint = {
#         "kind": "task",
#         "protocol": "terminal-bench-shell-v1",
#         "instruction": "print 'Hello World' with python in windows cmd"
#     }

#     multi_step_complaint = {
#         "kind": "task",
#         "protocol": "terminal-bench-shell-v1",
#         "instruction": "Create a file named test.txt, write 'Hello World' into it, and then print the contents of the file in windows cmd. Use at least three seperate commands to accomplish this task."
#     }

#     context_id = uuid4().hex
#     current_request = json.dumps(multi_step_complaint)
#     all_errors: list[str] = []
#     result_script: list[str] = []
#     max_iterations = 10
#     completed = False

#     response = None
#     for i in range(max_iterations):  # end after n rounds
#         if i == 0:  # send the initial instruction once
#             print("\nStarting test with request:", current_request)
#             response = await messenger.talk_to_agent(current_request, agent)
#         else:  # loop to execute or end the task based on agent's responses

#             try:
#                 payload = json.loads(response)  # error is error

#             except json.JSONDecodeError:
#                 raise ValueError(
#                     f"Invalid JSON in agent response: '{response}'")

#             if payload.get("kind") == "final":
#                 completed = True
#                 break  # task completed successfully

#             if not payload.get("kind") == "exec_request":
#                 raise ValueError(
#                     f"Unexpected message kind: {payload.get('kind')}")

#             exec_request = payload

#             if exec_request is None:
#                 break

#             print(f"Execution request: {exec_request}")
#             command, timeout = _parse_exec_request(exec_request)
#             result_script.append(command)
#             exec_result_dict = _execute_command(command, timeout)
#             exec_result = json.dumps(exec_result_dict)

#             print(f"Execution result: {exec_result}")

#             # send exec result back to agent and get next request
#             response = await messenger.talk_to_agent(exec_result, agent)

#     print(f"Result script: {result_script}")
#     assert completed, "Agent did not complete the terminal bench task"
