from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TerminalBenchLLMClientInterface(Protocol):

    @abstractmethod
    def rate_limited(self) -> bool: ...

    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def retry_log(self) -> list[dict]: ...

    async def invoke_async(self, messages: list[Any]) -> Any: ...
