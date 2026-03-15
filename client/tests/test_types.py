"""Tests for mlx_container.types — dataclass types."""

from __future__ import annotations

from dataclasses import fields

import pytest

from mlx_container.types import ChatMessage, GPUStatus, GenerateResult, ModelInfo


# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class TestGenerateResult:
    def test_required_field_text(self):
        result = GenerateResult(text="hello")
        assert result.text == "hello"

    def test_default_prompt_tokens(self):
        result = GenerateResult(text="x")
        assert result.prompt_tokens == 0

    def test_default_completion_tokens(self):
        result = GenerateResult(text="x")
        assert result.completion_tokens == 0

    def test_default_prompt_time_seconds(self):
        result = GenerateResult(text="x")
        assert result.prompt_time_seconds == 0.0

    def test_default_generation_time_seconds(self):
        result = GenerateResult(text="x")
        assert result.generation_time_seconds == 0.0

    def test_default_tokens_per_second(self):
        result = GenerateResult(text="x")
        assert result.tokens_per_second == 0.0

    def test_custom_values_are_stored(self):
        result = GenerateResult(
            text="The answer",
            prompt_tokens=8,
            completion_tokens=4,
            prompt_time_seconds=0.01,
            generation_time_seconds=0.5,
            tokens_per_second=8.0,
        )
        assert result.text == "The answer"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4
        assert result.prompt_time_seconds == pytest.approx(0.01)
        assert result.generation_time_seconds == pytest.approx(0.5)
        assert result.tokens_per_second == pytest.approx(8.0)

    def test_empty_text_is_valid(self):
        result = GenerateResult(text="")
        assert result.text == ""

    def test_is_dataclass(self):
        # Confirm it is indeed a dataclass with the expected field names
        field_names = {f.name for f in fields(GenerateResult)}
        assert "text" in field_names
        assert "prompt_tokens" in field_names
        assert "completion_tokens" in field_names
        assert "tokens_per_second" in field_names


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_required_field_model_id(self):
        info = ModelInfo(model_id="mlx-community/Llama-3.2-1B-4bit")
        assert info.model_id == "mlx-community/Llama-3.2-1B-4bit"

    def test_default_alias_is_empty_string(self):
        info = ModelInfo(model_id="some-model")
        assert info.alias == ""

    def test_default_memory_used_bytes(self):
        info = ModelInfo(model_id="some-model")
        assert info.memory_used_bytes == 0

    def test_default_is_loaded(self):
        info = ModelInfo(model_id="some-model")
        assert info.is_loaded is False

    def test_default_model_type(self):
        info = ModelInfo(model_id="some-model")
        assert info.model_type == "llm"

    def test_custom_values_stored(self):
        info = ModelInfo(
            model_id="mlx-community/Qwen2.5-1.5B-4bit",
            alias="qwen",
            memory_used_bytes=750_000_000,
            is_loaded=True,
            model_type="llm",
        )
        assert info.alias == "qwen"
        assert info.memory_used_bytes == 750_000_000
        assert info.is_loaded is True

    def test_is_dataclass(self):
        field_names = {f.name for f in fields(ModelInfo)}
        assert "model_id" in field_names
        assert "alias" in field_names
        assert "memory_used_bytes" in field_names
        assert "is_loaded" in field_names
        assert "model_type" in field_names


# ---------------------------------------------------------------------------
# GPUStatus
# ---------------------------------------------------------------------------


class TestGPUStatus:
    def test_required_field_device_name(self):
        status = GPUStatus(device_name="Apple M3 Max")
        assert status.device_name == "Apple M3 Max"

    def test_default_total_memory_bytes(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.total_memory_bytes == 0

    def test_default_used_memory_bytes(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.used_memory_bytes == 0

    def test_default_available_memory_bytes(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.available_memory_bytes == 0

    def test_default_gpu_family_is_empty_string(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.gpu_family == ""

    def test_default_loaded_models_count(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.loaded_models_count == 0

    def test_default_loaded_models_is_empty_list(self):
        status = GPUStatus(device_name="Apple M3")
        assert status.loaded_models == []

    def test_loaded_models_list_independence(self):
        # Two separate instances should not share the same list
        a = GPUStatus(device_name="A")
        b = GPUStatus(device_name="B")
        a.loaded_models.append(ModelInfo(model_id="x"))
        assert len(b.loaded_models) == 0

    def test_custom_values_stored(self):
        models = [ModelInfo(model_id="llama", is_loaded=True)]
        status = GPUStatus(
            device_name="Apple M4",
            total_memory_bytes=32_000_000_000,
            used_memory_bytes=1_000_000_000,
            available_memory_bytes=31_000_000_000,
            gpu_family="metal3",
            loaded_models_count=1,
            loaded_models=models,
        )
        assert status.gpu_family == "metal3"
        assert status.loaded_models_count == 1
        assert len(status.loaded_models) == 1

    def test_is_dataclass(self):
        field_names = {f.name for f in fields(GPUStatus)}
        assert "device_name" in field_names
        assert "total_memory_bytes" in field_names
        assert "gpu_family" in field_names
        assert "loaded_models" in field_names


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_role_and_content_stored(self):
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_system_role(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_assistant_role(self):
        msg = ChatMessage(role="assistant", content="4.")
        assert msg.role == "assistant"

    def test_empty_content_is_valid(self):
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""

    def test_is_dataclass(self):
        field_names = {f.name for f in fields(ChatMessage)}
        assert "role" in field_names
        assert "content" in field_names
