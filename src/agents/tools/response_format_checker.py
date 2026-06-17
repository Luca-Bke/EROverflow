import json
import re

from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ResponseFormatChecker():
    @staticmethod
    def check_agent_response_valid_json(response_text: str):
        """Parse the actor response into a single JSON object.

        Deterministic normalisation that runs before the LLM checker:
          1. Fast path: parse as-is.
          2. Strip <think>...</think> blocks (reasoning models like qwen3) and retry.
          3. raw_decode the first object; trailing content means the actor emitted
             several commands at once — a precise, LLM-free diagnosis.
        """
        # 1. Fast path
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 2. Strip reasoning tags
        stripped = _THINK_RE.sub("", response_text).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 3. Decode the first object; detect multiple concatenated objects
        decoder = json.JSONDecoder()
        try:
            obj, end = decoder.raw_decode(stripped)
        except (json.JSONDecodeError, ValueError):
            raise terminal_bench_format_exception(
                "The given response is not valid JSON: " + response_text)

        if stripped[end:].strip():
            raise terminal_bench_format_exception(
                "Multiple JSON objects detected in one response — send only the "
                "FIRST single command and wait for its execution result before "
                "issuing the next. Offending response: " + response_text)

        return obj
