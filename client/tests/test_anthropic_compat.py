"""
Tests for mlx_container.compat.anthropic — Anthropic Messages API wrapper.

All gRPC I/O is mocked so the suite runs without a live daemon.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from mlx_container.compat.anthropic import (
    ContentBlock,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStream,
    Messages,
    Usage,
    _build_messages,
)
from mlx_container.types import ChatMessage, GenerateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generate_result(
    text: str = "Hello!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> GenerateResult:
    return GenerateResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _token_iter(*tokens: str) -> Iterator[str]:
    """Return a plain iterator of token strings."""
    return iter(tokens)


# ---------------------------------------------------------------------------
# ContentBlock
# ---------------------------------------------------------------------------


class TestContentBlock:
    def test_defaults(self):
        block = ContentBlock()
        assert block.type == "text"
        assert block.text == ""

    def test_custom_values(self):
        block = ContentBlock(type="text", text="Hello, world!")
        assert block.type == "text"
        assert block.text == "Hello, world!"

    def test_is_dataclass(self):
        names = {f.name for f in fields(ContentBlock)}
        assert "type" in names
        assert "text" in names


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class TestUsage:
    def test_defaults(self):
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_custom_values(self):
        usage = Usage(input_tokens=20, output_tokens=80)
        assert usage.input_tokens == 20
        assert usage.output_tokens == 80

    def test_is_dataclass(self):
        names = {f.name for f in fields(Usage)}
        assert "input_tokens" in names
        assert "output_tokens" in names


# ---------------------------------------------------------------------------
# MessageResponse
# ---------------------------------------------------------------------------


class TestMessageResponse:
    def test_defaults(self):
        resp = MessageResponse()
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.content == []
        assert resp.stop_reason is None
        assert resp.stop_sequence is None
        assert isinstance(resp.usage, Usage)

    def test_id_stored(self):
        resp = MessageResponse(id="msg_abc123")
        assert resp.id == "msg_abc123"

    def test_model_stored(self):
        resp = MessageResponse(model="mlx-community/Llama-3.2-1B-4bit")
        assert resp.model == "mlx-community/Llama-3.2-1B-4bit"

    def test_content_list_independence(self):
        a = MessageResponse()
        b = MessageResponse()
        a.content.append(ContentBlock(text="hi"))
        assert len(b.content) == 0

    def test_stop_reason_end_turn(self):
        resp = MessageResponse(stop_reason="end_turn")
        assert resp.stop_reason == "end_turn"

    def test_stop_reason_max_tokens(self):
        resp = MessageResponse(stop_reason="max_tokens")
        assert resp.stop_reason == "max_tokens"

    def test_content_block_accessible(self):
        block = ContentBlock(text="Some text")
        resp = MessageResponse(content=[block])
        assert resp.content[0].text == "Some text"

    def test_usage_tokens(self):
        resp = MessageResponse(usage=Usage(input_tokens=12, output_tokens=34))
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 34

    def test_is_dataclass(self):
        names = {f.name for f in fields(MessageResponse)}
        for expected in ("id", "type", "role", "content", "model", "stop_reason", "usage"):
            assert expected in names


# ---------------------------------------------------------------------------
# Streaming event dataclasses
# ---------------------------------------------------------------------------


class TestStreamEventTypes:
    def test_message_start_event_defaults(self):
        evt = MessageStartEvent()
        assert evt.type == "message_start"
        assert evt.message is None

    def test_message_start_event_with_message(self):
        msg = MessageResponse(id="msg_x")
        evt = MessageStartEvent(message=msg)
        assert evt.message.id == "msg_x"

    def test_content_block_start_event(self):
        evt = ContentBlockStartEvent(index=0)
        assert evt.type == "content_block_start"
        assert evt.index == 0
        assert isinstance(evt.content_block, ContentBlock)

    def test_content_block_delta_event(self):
        delta = ContentBlock(type="text_delta", text="tok")
        evt = ContentBlockDeltaEvent(index=0, delta=delta)
        assert evt.type == "content_block_delta"
        assert evt.delta.text == "tok"

    def test_content_block_stop_event(self):
        evt = ContentBlockStopEvent(index=0)
        assert evt.type == "content_block_stop"
        assert evt.index == 0

    def test_message_delta_event(self):
        evt = MessageDeltaEvent(
            delta={"stop_reason": "end_turn"},
            usage=Usage(output_tokens=42),
        )
        assert evt.type == "message_delta"
        assert evt.delta["stop_reason"] == "end_turn"
        assert evt.usage.output_tokens == 42

    def test_message_stop_event(self):
        evt = MessageStopEvent()
        assert evt.type == "message_stop"


# ---------------------------------------------------------------------------
# _build_messages helper
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_no_system(self):
        msgs = _build_messages(None, [{"role": "user", "content": "Hi"}])
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hi"

    def test_with_system_prepended(self):
        msgs = _build_messages(
            "You are helpful.",
            [{"role": "user", "content": "Hello"}],
        )
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "You are helpful."
        assert msgs[1].role == "user"

    def test_empty_messages_with_system(self):
        msgs = _build_messages("sys", [])
        assert len(msgs) == 1
        assert msgs[0].role == "system"

    def test_returns_chat_message_instances(self):
        msgs = _build_messages(None, [{"role": "user", "content": "x"}])
        assert isinstance(msgs[0], ChatMessage)

    def test_multi_turn_order_preserved(self):
        msgs = _build_messages(
            None,
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
                {"role": "user", "content": "C"},
            ],
        )
        roles = [m.role for m in msgs]
        assert roles == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Messages.create (non-streaming)
# ---------------------------------------------------------------------------


class TestMessagesCreate:
    @patch("mlx_container.compat.anthropic.get_client")
    def test_returns_message_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result(
            text="4.", prompt_tokens=8, completion_tokens=2
        )

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert isinstance(resp, MessageResponse)

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_text_content(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result(text="4.")

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        assert resp.content[0].text == "4."
        assert resp.content[0].type == "text"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_usage_tokens(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result(
            prompt_tokens=10, completion_tokens=5
        )

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    @patch("mlx_container.compat.anthropic.get_client")
    def test_stop_reason_end_turn(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # completion_tokens (3) < max_tokens (256) → end_turn
        mock_client.generate.return_value = _make_generate_result(completion_tokens=3)

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.stop_reason == "end_turn"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_stop_reason_max_tokens(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # completion_tokens equals max_tokens → max_tokens stop reason
        mock_client.generate.return_value = _make_generate_result(completion_tokens=16)

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=16,
            messages=[{"role": "user", "content": "Go on..."}],
        )

        assert resp.stop_reason == "max_tokens"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_system_prompt_forwarded(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result()

        Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
            system="Be concise.",
        )

        call_kwargs = mock_client.generate.call_args
        sent_messages = call_kwargs.kwargs.get(
            "messages", call_kwargs.args[0] if call_kwargs.args else []
        )
        # Accept both positional and keyword call styles
        if not sent_messages:
            sent_messages = call_kwargs[1].get("messages", call_kwargs[0][0])
        roles = [m.role for m in sent_messages]
        assert "system" in roles
        assert roles[0] == "system"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_id_has_msg_prefix(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result()

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.id.startswith("msg_")

    @patch("mlx_container.compat.anthropic.get_client")
    def test_model_echoed_in_response(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result()
        model = "mlx-community/Llama-3.2-1B-4bit"

        resp = Messages.create(
            model=model,
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.model == model

    @patch("mlx_container.compat.anthropic.get_client")
    def test_role_is_assistant(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result()

        resp = Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert resp.role == "assistant"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_stream_false_passed_to_client(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = _make_generate_result()

        Messages.create(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
        )

        _, kwargs = mock_client.generate.call_args
        assert kwargs.get("stream") is False


# ---------------------------------------------------------------------------
# MessageStream
# ---------------------------------------------------------------------------


class TestMessageStream:
    def _make_stream(
        self,
        tokens: list[str],
        max_tokens: int = 256,
    ) -> tuple[MessageStream, MagicMock]:
        mock_client = MagicMock()
        mock_client.generate.return_value = iter(tokens)

        stream = MessageStream(
            model="mlx-community/Llama-3.2-1B-4bit",
            messages=[ChatMessage(role="user", content="Count to 3.")],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1.0,
            msg_id="msg_testid",
        )
        return stream, mock_client

    @patch("mlx_container.compat.anthropic.get_client")
    def test_text_stream_yields_tokens(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        tokens = ["1", ", ", "2", ", ", "3"]
        mock_client.generate.return_value = iter(tokens)

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Count to 3."}],
        ) as stream:
            collected = list(stream.text_stream)

        assert collected == tokens

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_populated_after_context(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = iter(["Hello", "!"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response is not None
        assert isinstance(stream.response, MessageResponse)

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_full_text(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = iter(["Hel", "lo", "!"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response.content[0].text == "Hello!"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_stop_reason_end_turn(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # 3 tokens, max_tokens=256 → end_turn
        mock_client.generate.return_value = iter(["a", "b", "c"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response.stop_reason == "end_turn"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_stop_reason_max_tokens(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        # 3 tokens, max_tokens=3 → max_tokens
        mock_client.generate.return_value = iter(["a", "b", "c"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=3,
            messages=[{"role": "user", "content": "Go"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response.stop_reason == "max_tokens"

    @patch("mlx_container.compat.anthropic.get_client")
    def test_response_output_tokens_counted(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        tokens = ["one", " ", "two", " ", "three"]
        mock_client.generate.return_value = iter(tokens)

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response.usage.output_tokens == len(tokens)

    @patch("mlx_container.compat.anthropic.get_client")
    def test_stream_used_as_context_manager_exits_cleanly(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = iter(["hello"])

        # Must not raise even if text_stream is not iterated manually
        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ):
            pass  # __exit__ should drain the stream

    @patch("mlx_container.compat.anthropic.get_client")
    def test_stream_id_has_msg_prefix(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = iter(["tok"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass

        assert stream.response.id.startswith("msg_")

    def test_text_stream_raises_outside_context(self):
        stream = MessageStream(
            model="mlx-community/Llama-3.2-1B-4bit",
            messages=[ChatMessage(role="user", content="Hi")],
            max_tokens=64,
            temperature=0.7,
            top_p=1.0,
            msg_id="msg_test",
        )
        with pytest.raises(RuntimeError, match="context manager"):
            # Calling next() on the generator triggers the RuntimeError
            next(stream.text_stream)

    @patch("mlx_container.compat.anthropic.get_client")
    def test_system_prompt_forwarded_to_stream(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.generate.return_value = iter(["ok"])

        with Messages.stream(
            model="mlx-community/Llama-3.2-1B-4bit",
            max_tokens=64,
            messages=[{"role": "user", "content": "Hi"}],
            system="Be helpful.",
        ) as stream:
            for _ in stream.text_stream:
                pass

        _, kwargs = mock_client.generate.call_args
        sent = kwargs["messages"]
        assert sent[0].role == "system"
        assert sent[0].content == "Be helpful."
