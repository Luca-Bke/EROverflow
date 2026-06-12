import json

from agents.terminal_bench_supplementary.terminal_bench_format_exception import terminal_bench_format_exception


class ResponseFormatChecker():
    def check_agent_response_valid_json(response_text: str):
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            raise terminal_bench_format_exception(
                "LLM response is not valid JSON: " + response_text)
