"""Shared fixtures for the mlx_container test suite."""

from __future__ import annotations

import pytest

from mlx_container.proto import mlx_container_pb2 as pb2


@pytest.fixture()
def sample_model_info() -> pb2.ModelInfo:
    """A reusable ModelInfo message fixture."""
    return pb2.ModelInfo(
        model_id="mlx-community/Llama-3.2-1B-4bit",
        alias="llama-small",
        memory_used_bytes=800_000_000,
        is_loaded=True,
        model_type="llm",
    )


@pytest.fixture()
def sample_generate_parameters() -> pb2.GenerateParameters:
    """A reusable GenerateParameters message fixture."""
    return pb2.GenerateParameters(
        max_tokens=512,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.0,
        repetition_context_size=20,
    )


@pytest.fixture()
def sample_chat_messages() -> list[pb2.ChatMessage]:
    """A reusable list of ChatMessage fixtures."""
    return [
        pb2.ChatMessage(role="system", content="You are a helpful assistant."),
        pb2.ChatMessage(role="user", content="What is 2+2?"),
    ]


@pytest.fixture()
def sample_generate_complete() -> pb2.GenerateComplete:
    """A reusable GenerateComplete message fixture."""
    return pb2.GenerateComplete(
        full_text="The answer is 4.",
        prompt_tokens=10,
        completion_tokens=5,
        prompt_time_seconds=0.05,
        generation_time_seconds=0.3,
        tokens_per_second=16.7,
    )
