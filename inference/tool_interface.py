"""Opaque observation boundary. No video decoder or tool implementation is shipped."""
from typing import Any, Protocol


class RecoverableToolError(RuntimeError):
    """A failed observation request the model can revise.

    Supply a message suitable for the model, such as an invalid time range or an
    unreadable crop. Do not include credentials or internal tracebacks. Unrelated
    infrastructure failures should use their original exception types.
    """


class ObservationBackend(Protocol):
    def tool_specs(self) -> list[dict]:
        """Return the schemas of tools actually provided by this backend."""
        ...

    def initial_content(self, source: str) -> str | list[dict]:
        """Return initial user content from the caller's observation implementation."""
        ...

    def encode(self, *, messages: list[dict], tools: list[dict], processor: Any) -> dict:
        """Render the full history and return model-ready tensors, without labels.

        The caller owns media preparation and checkpoint-compatible serialization.
        The returned input must include every prior observation and assistant turn.
        """
        ...

    def execute(self, *, name: str, arguments: dict) -> str | list[dict]:
        """Return a real observation, or raise RecoverableToolError for a retryable request.

        The controller reports recoverable failures to the model for replanning.
        Other exceptions propagate to the caller; no fallback evidence is made up.
        """
        ...
