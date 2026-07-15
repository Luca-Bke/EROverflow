from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AbstractLLMClient(Protocol):

    @abstractmethod
    def rate_limited(self) -> bool: ...

    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def retry_log(self) -> list[dict]: ...

    async def invoke_async(self, messages: list[Any]) -> Any: ...

    async def invoke_with_tools_async(
        self,
        messages: list[Any],
        tools: list[dict],
    ) -> Any: ...

    async def invoke_with_response_format_async(
        self,
        messages: list[Any],
        response_format: dict,
    ) -> Any: ...
