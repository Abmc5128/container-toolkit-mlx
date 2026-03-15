"""
Anthropic Messages API compatibility wrapper for MLX Container.

Provides a drop-in replacement for ``anthropic.Anthropic().messages`` so that
code written against the Anthropic Python SDK works against the local MLX
Container daemon without modification.

Usage:

    from mlx_container.compat.anthropic import Messages

    # Non-streaming
    response = Messages.create(
        model="mlx-community/Llama-3.2-1B-4bit",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello!"}],
        system="You are a helpful assistant.",
    )
    print(response.content[0].text)
    print(f"Usage: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

    # Streaming
    with Messages.stream(
        model="mlx-community/Llama-3.2-1B-4bit",
        max_tokens=256,
        messages=[{"role": "user", "content": "Count to 5."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Iterator, Optional

from mlx_container._grpc_client import get_client
from mlx_container.types import ChatMessage


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """A single content block in an Anthropic message response."""

    type: str = "text"
    text: str = ""


@dataclass
class Usage:
    """Token usage for an Anthropic message response."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class MessageResponse:
    """
    Anthropic-compatible message response.

    Mirrors the shape of ``anthropic.types.Message`` so that callers can
    swap ``anthropic.Anthropic().messages.create(...)`` for
    ``Messages.create(...)`` with zero changes.
    """

    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None   # "end_turn" | "max_tokens" | None
    stop_sequence: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Streaming event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MessageStartEvent:
    type: str = "message_start"
    message: Optional[MessageResponse] = None


@dataclass
class ContentBlockStartEvent:
    type: str = "content_block_start"
    index: int = 0
    content_block: ContentBlock = field(default_factory=ContentBlock)


@dataclass
class ContentBlockDeltaEvent:
    type: str = "content_block_delta"
    index: int = 0
    delta: ContentBlock = field(default_factory=ContentBlock)


@dataclass
class ContentBlockStopEvent:
    type: str = "content_block_stop"
    index: int = 0


@dataclass
class MessageDeltaEvent:
    type: str = "message_delta"
    delta: dict = field(default_factory=dict)   # {"stop_reason": "end_turn"}
    usage: Usage = field(default_factory=Usage)


@dataclass
class MessageStopEvent:
    type: str = "message_stop"


# Union type alias for stream events
StreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
)


# ---------------------------------------------------------------------------
# Stream context manager
# ---------------------------------------------------------------------------


class MessageStream:
    """
    Context manager returned by ``Messages.stream()``.

    Attributes:
        text_stream: Generator yielding plain text strings as tokens arrive.
        response:    The final :class:`MessageResponse` after the stream ends.
                     Only populated after the ``with`` block completes.
    """

    def __init__(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: float,
        msg_id: str,
    ) -> None:
        self._model = model
        self._messages = messages
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._msg_id = msg_id
        self.response: Optional[MessageResponse] = None
        self._token_iter: Optional[Iterator[str]] = None
        # Cached generator — created once so multiple accesses to .text_stream
        # all return the same object and the accumulated state is not reset.
        self._text_stream_gen: Optional[Generator[str, None, None]] = None

    # -- context protocol ----------------------------------------------------

    def __enter__(self) -> "MessageStream":
        self._token_iter = get_client().generate(
            model=self._model,
            messages=self._messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            stream=True,
        )
        return self

    def __exit__(self, *args: object) -> None:
        # Drain any unconsumed tokens so the gRPC stream is fully consumed and
        # the final MessageResponse is correctly populated.
        for _ in self.text_stream:
            pass

    # -- public interface ----------------------------------------------------

    @property
    def text_stream(self) -> Generator[str, None, None]:
        """Yield each text token and build the final :class:`MessageResponse`."""
        if self._token_iter is None:
            raise RuntimeError(
                "MessageStream must be used as a context manager "
                "(i.e. `with Messages.stream(...) as stream:`)"
            )

        # Return the cached generator so that repeated access (e.g. from
        # __exit__) does not create a new generator that would overwrite the
        # already-populated response with empty values.
        if self._text_stream_gen is None:
            self._text_stream_gen = self._generate()
        return self._text_stream_gen

    def _generate(self) -> Generator[str, None, None]:
        """Internal generator that drives the token iterator."""
        assert self._token_iter is not None

        accumulated: list[str] = []
        output_tokens = 0

        for token in self._token_iter:
            accumulated.append(token)
            output_tokens += 1
            yield token

        full_text = "".join(accumulated)
        stop_reason = (
            "max_tokens" if output_tokens >= self._max_tokens else "end_turn"
        )

        self.response = MessageResponse(
            id=self._msg_id,
            model=self._model,
            content=[ContentBlock(type="text", text=full_text)],
            stop_reason=stop_reason,
            usage=Usage(input_tokens=0, output_tokens=output_tokens),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Messages:
    """
    Anthropic Messages API wrapper.

    Mirrors ``anthropic.Anthropic().messages`` so existing Anthropic SDK
    call-sites work unchanged against the local MLX Container daemon.
    """

    @staticmethod
    def create(
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        system: Optional[str] = None,
    ) -> MessageResponse:
        """
        Create a message (non-streaming).

        Args:
            model:      Model ID to use for inference.
            messages:   Conversation turns as dicts with ``role`` and
                        ``content`` keys.  The Anthropic ``user``/
                        ``assistant`` roles are passed through unchanged.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0–1).
            top_p:      Nucleus-sampling probability mass.
            system:     Optional system prompt.  Prepended as a
                        ``system`` role message before the conversation.

        Returns:
            :class:`MessageResponse` with Anthropic-shaped fields.
        """
        chat_msgs = _build_messages(system, messages)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        result = get_client().generate(
            model=model,
            messages=chat_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )

        stop_reason = (
            "max_tokens"
            if result.completion_tokens >= max_tokens
            else "end_turn"
        )

        return MessageResponse(
            id=msg_id,
            model=model,
            content=[ContentBlock(type="text", text=result.text)],
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            ),
        )

    @staticmethod
    def stream(
        model: str,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        system: Optional[str] = None,
    ) -> MessageStream:
        """
        Create a streaming message context manager.

        Usage::

            with Messages.stream(model=..., messages=[...]) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
            print(stream.response.stop_reason)

        Args:
            model:      Model ID.
            messages:   Conversation turns as Anthropic-format dicts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p:      Top-p sampling.
            system:     Optional system prompt.

        Returns:
            :class:`MessageStream` context manager.
        """
        chat_msgs = _build_messages(system, messages)
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        return MessageStream(
            model=model,
            messages=chat_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            msg_id=msg_id,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_messages(
    system: Optional[str],
    messages: list[dict],
) -> list[ChatMessage]:
    """
    Combine an optional system prompt with the conversation messages.

    The Anthropic API keeps the system prompt separate from the messages
    list; the underlying gRPC daemon accepts it as a leading ``system``
    role entry, consistent with how the OpenAI compat layer handles it.
    """
    result: list[ChatMessage] = []

    if system:
        result.append(ChatMessage(role="system", content=system))

    for m in messages:
        result.append(ChatMessage(role=m["role"], content=m["content"]))

    return result
