"""
JSON-serializable message classes matching the Swift server's Codable structs.

The Swift gRPC server uses JSONEncoder/JSONDecoder (via JSONMessageSerializer /
JSONMessageDeserializer in MLXContainerProtocol/Service.swift).  Swift's
JSONEncoder encodes property names verbatim — so all field names on the wire
are camelCase Swift identifiers, NOT the snake_case proto field names.

Mapping (proto field  →  Swift property  →  JSON key):
  model_id             → modelID           → "modelID"
  memory_budget_bytes  → memoryBudgetBytes → "memoryBudgetBytes"
  load_time_seconds    → loadTimeSeconds   → "loadTimeSeconds"
  … etc.

These classes carry that camelCase knowledge inside ``_to_json_dict()`` so that
``mlx_container_pb2_grpc._serialize`` always produces the correct payload.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

class _JSONMessage:
    """
    Base for all hand-written message classes.

    Subclasses declare ``_JSON_FIELDS``: a list of ``(attr_name, json_key)``
    pairs that drive serialisation.  Nested _JSONMessage objects and lists
    of _JSONMessage objects are handled recursively.
    """

    _JSON_FIELDS: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Serialisation / deserialisation
    # ------------------------------------------------------------------

    def _to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for attr, key in self._JSON_FIELDS:
            val = getattr(self, attr)
            if isinstance(val, _JSONMessage):
                out[key] = val._to_json_dict()
            elif isinstance(val, list):
                out[key] = [
                    v._to_json_dict() if isinstance(v, _JSONMessage) else v
                    for v in val
                ]
            else:
                out[key] = val
        return out

    def SerializeToString(self) -> bytes:
        return json.dumps(self._to_json_dict()).encode("utf-8")

    @classmethod
    def _from_json_dict(cls, d: dict[str, Any]) -> "_JSONMessage":
        raise NotImplementedError(f"{cls.__name__}._from_json_dict not implemented")

    @classmethod
    def FromString(cls, data: bytes) -> "_JSONMessage":
        d = json.loads(data.decode("utf-8"))
        return cls._from_json_dict(d)

    # ------------------------------------------------------------------
    # oneof helper (used by GenerateResponse)
    # ------------------------------------------------------------------

    def HasField(self, field_name: str) -> bool:
        return bool(getattr(self, field_name, None) is not None)


# ---------------------------------------------------------------------------
# Model Management
# ---------------------------------------------------------------------------

class LoadModelRequest(_JSONMessage):
    _JSON_FIELDS = [
        ("model_id", "modelID"),
        ("alias", "alias"),
        ("memory_budget_bytes", "memoryBudgetBytes"),
    ]

    def __init__(self, model_id: str = "", alias: str = "",
                 memory_budget_bytes: int = 0, **_: Any) -> None:
        self.model_id = model_id
        self.alias = alias
        self.memory_budget_bytes = memory_budget_bytes

    @classmethod
    def _from_json_dict(cls, d: dict) -> "LoadModelRequest":
        return cls(
            model_id=d.get("modelID", ""),
            alias=d.get("alias", ""),
            memory_budget_bytes=d.get("memoryBudgetBytes", 0),
        )


class LoadModelResponse(_JSONMessage):
    _JSON_FIELDS = [
        ("success", "success"),
        ("model_id", "modelID"),
        ("error", "error"),
        ("memory_used_bytes", "memoryUsedBytes"),
        ("load_time_seconds", "loadTimeSeconds"),
    ]

    def __init__(self, success: bool = False, model_id: str = "", error: str = "",
                 memory_used_bytes: int = 0, load_time_seconds: float = 0.0,
                 **_: Any) -> None:
        self.success = success
        self.model_id = model_id
        self.error = error
        self.memory_used_bytes = memory_used_bytes
        self.load_time_seconds = load_time_seconds

    @classmethod
    def _from_json_dict(cls, d: dict) -> "LoadModelResponse":
        return cls(
            success=d.get("success", False),
            model_id=d.get("modelID", ""),
            error=d.get("error", ""),
            memory_used_bytes=d.get("memoryUsedBytes", 0),
            load_time_seconds=d.get("loadTimeSeconds", 0.0),
        )


class UnloadModelRequest(_JSONMessage):
    _JSON_FIELDS = [("model_id", "modelID")]

    def __init__(self, model_id: str = "", **_: Any) -> None:
        self.model_id = model_id

    @classmethod
    def _from_json_dict(cls, d: dict) -> "UnloadModelRequest":
        return cls(model_id=d.get("modelID", ""))


class UnloadModelResponse(_JSONMessage):
    _JSON_FIELDS = [
        ("success", "success"),
        ("error", "error"),
        ("memory_freed_bytes", "memoryFreedBytes"),
    ]

    def __init__(self, success: bool = False, error: str = "",
                 memory_freed_bytes: int = 0, **_: Any) -> None:
        self.success = success
        self.error = error
        self.memory_freed_bytes = memory_freed_bytes

    @classmethod
    def _from_json_dict(cls, d: dict) -> "UnloadModelResponse":
        return cls(
            success=d.get("success", False),
            error=d.get("error", ""),
            memory_freed_bytes=d.get("memoryFreedBytes", 0),
        )


class ListModelsRequest(_JSONMessage):
    _JSON_FIELDS: list[tuple[str, str]] = []

    def __init__(self, **_: Any) -> None:
        pass

    @classmethod
    def _from_json_dict(cls, d: dict) -> "ListModelsRequest":
        return cls()


class ListModelsResponse(_JSONMessage):
    _JSON_FIELDS = [("models", "models")]

    def __init__(self, models: list | None = None, **_: Any) -> None:
        self.models: list[ModelInfo] = models or []

    @classmethod
    def _from_json_dict(cls, d: dict) -> "ListModelsResponse":
        return cls(
            models=[ModelInfo._from_json_dict(m) for m in d.get("models", [])]
        )


class ModelInfo(_JSONMessage):
    _JSON_FIELDS = [
        ("model_id", "modelID"),
        ("alias", "alias"),
        ("memory_used_bytes", "memoryUsedBytes"),
        ("is_loaded", "isLoaded"),
        ("model_type", "modelType"),
    ]

    def __init__(self, model_id: str = "", alias: str = "",
                 memory_used_bytes: int = 0, is_loaded: bool = False,
                 model_type: str = "", **_: Any) -> None:
        self.model_id = model_id
        self.alias = alias
        self.memory_used_bytes = memory_used_bytes
        self.is_loaded = is_loaded
        self.model_type = model_type

    @classmethod
    def _from_json_dict(cls, d: dict) -> "ModelInfo":
        return cls(
            model_id=d.get("modelID", ""),
            alias=d.get("alias", ""),
            memory_used_bytes=d.get("memoryUsedBytes", 0),
            is_loaded=d.get("isLoaded", False),
            model_type=d.get("modelType", ""),
        )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ChatMessage(_JSONMessage):
    _JSON_FIELDS = [("role", "role"), ("content", "content")]

    def __init__(self, role: str = "", content: str = "", **_: Any) -> None:
        self.role = role
        self.content = content

    @classmethod
    def _from_json_dict(cls, d: dict) -> "ChatMessage":
        return cls(role=d.get("role", ""), content=d.get("content", ""))


class GenerateParameters(_JSONMessage):
    _JSON_FIELDS = [
        ("max_tokens", "maxTokens"),
        ("temperature", "temperature"),
        ("top_p", "topP"),
        ("repetition_penalty", "repetitionPenalty"),
        ("repetition_context_size", "repetitionContextSize"),
    ]

    def __init__(self, max_tokens: int = 0, temperature: float = 0.0,
                 top_p: float = 0.0, repetition_penalty: float = 0.0,
                 repetition_context_size: int = 0, **_: Any) -> None:
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repetition_context_size = repetition_context_size

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GenerateParameters":
        return cls(
            max_tokens=d.get("maxTokens", 0),
            temperature=d.get("temperature", 0.0),
            top_p=d.get("topP", 0.0),
            repetition_penalty=d.get("repetitionPenalty", 0.0),
            repetition_context_size=d.get("repetitionContextSize", 0),
        )


class GenerateRequest(_JSONMessage):
    _JSON_FIELDS = [
        ("model_id", "modelID"),
        ("prompt", "prompt"),
        ("messages", "messages"),
        ("parameters", "parameters"),
        ("container_id", "containerID"),
    ]

    def __init__(self, model_id: str = "", prompt: str = "",
                 messages: list | None = None,
                 parameters: GenerateParameters | None = None,
                 container_id: str = "", **_: Any) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.messages: list[ChatMessage] = messages or []
        self.parameters: GenerateParameters = parameters or GenerateParameters()
        self.container_id = container_id

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GenerateRequest":
        return cls(
            model_id=d.get("modelID", ""),
            prompt=d.get("prompt", ""),
            messages=[ChatMessage._from_json_dict(m) for m in d.get("messages", [])],
            parameters=GenerateParameters._from_json_dict(d.get("parameters", {})),
            container_id=d.get("containerID", ""),
        )


class GenerateComplete(_JSONMessage):
    _JSON_FIELDS = [
        ("full_text", "fullText"),
        ("prompt_tokens", "promptTokens"),
        ("completion_tokens", "completionTokens"),
        ("prompt_time_seconds", "promptTimeSeconds"),
        ("generation_time_seconds", "generationTimeSeconds"),
        ("tokens_per_second", "tokensPerSecond"),
    ]

    def __init__(self, full_text: str = "", prompt_tokens: int = 0,
                 completion_tokens: int = 0, prompt_time_seconds: float = 0.0,
                 generation_time_seconds: float = 0.0,
                 tokens_per_second: float = 0.0, **_: Any) -> None:
        self.full_text = full_text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_time_seconds = prompt_time_seconds
        self.generation_time_seconds = generation_time_seconds
        self.tokens_per_second = tokens_per_second

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GenerateComplete":
        return cls(
            full_text=d.get("fullText", ""),
            prompt_tokens=d.get("promptTokens", 0),
            completion_tokens=d.get("completionTokens", 0),
            prompt_time_seconds=d.get("promptTimeSeconds", 0.0),
            generation_time_seconds=d.get("generationTimeSeconds", 0.0),
            tokens_per_second=d.get("tokensPerSecond", 0.0),
        )


class GenerateResponse(_JSONMessage):
    """
    Represents the server-streaming response from Generate.

    The Swift server sends either:
      {"token": "<text>"}         — streaming token
      {"complete": {…}}           — final stats message

    Both fields are optional (only one is set per message).
    ``HasField`` returns True when the field is not None.
    """

    # No _JSON_FIELDS needed — serialisation is custom (client never sends this)
    _JSON_FIELDS: list[tuple[str, str]] = []

    def __init__(self, token: str | None = None,
                 complete: GenerateComplete | None = None, **_: Any) -> None:
        self.token: str | None = token
        self.complete: GenerateComplete | None = complete

    def HasField(self, field_name: str) -> bool:
        if field_name == "token":
            return self.token is not None
        if field_name == "complete":
            return self.complete is not None
        return False

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GenerateResponse":
        token: str | None = d.get("token")  # may be None or absent
        complete_dict: dict | None = d.get("complete")
        complete = GenerateComplete._from_json_dict(complete_dict) if complete_dict is not None else None
        return cls(token=token, complete=complete)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbedRequest(_JSONMessage):
    _JSON_FIELDS = [
        ("model_id", "modelID"),
        ("texts", "texts"),
        ("container_id", "containerID"),
    ]

    def __init__(self, model_id: str = "", texts: list[str] | None = None,
                 container_id: str = "", **_: Any) -> None:
        self.model_id = model_id
        self.texts: list[str] = texts or []
        self.container_id = container_id

    @classmethod
    def _from_json_dict(cls, d: dict) -> "EmbedRequest":
        return cls(
            model_id=d.get("modelID", ""),
            texts=d.get("texts", []),
            container_id=d.get("containerID", ""),
        )


class Embedding(_JSONMessage):
    _JSON_FIELDS = [("values", "values")]

    def __init__(self, values: list[float] | None = None, **_: Any) -> None:
        self.values: list[float] = values or []

    @classmethod
    def _from_json_dict(cls, d: dict) -> "Embedding":
        return cls(values=d.get("values", []))


class EmbedResponse(_JSONMessage):
    _JSON_FIELDS = [("embeddings", "embeddings"), ("error", "error")]

    def __init__(self, embeddings: list | None = None, error: str = "",
                 **_: Any) -> None:
        self.embeddings: list[Embedding] = embeddings or []
        self.error = error

    @classmethod
    def _from_json_dict(cls, d: dict) -> "EmbedResponse":
        return cls(
            embeddings=[Embedding._from_json_dict(e) for e in d.get("embeddings", [])],
            error=d.get("error", ""),
        )


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

class GetGPUStatusRequest(_JSONMessage):
    _JSON_FIELDS: list[tuple[str, str]] = []

    def __init__(self, **_: Any) -> None:
        pass

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GetGPUStatusRequest":
        return cls()


class GetGPUStatusResponse(_JSONMessage):
    _JSON_FIELDS = [
        ("device_name", "deviceName"),
        ("total_memory_bytes", "totalMemoryBytes"),
        ("used_memory_bytes", "usedMemoryBytes"),
        ("available_memory_bytes", "availableMemoryBytes"),
        ("gpu_family", "gpuFamily"),
        ("loaded_models_count", "loadedModelsCount"),
        ("loaded_models", "loadedModels"),
    ]

    def __init__(self, device_name: str = "", total_memory_bytes: int = 0,
                 used_memory_bytes: int = 0, available_memory_bytes: int = 0,
                 gpu_family: str = "", loaded_models_count: int = 0,
                 loaded_models: list | None = None, **_: Any) -> None:
        self.device_name = device_name
        self.total_memory_bytes = total_memory_bytes
        self.used_memory_bytes = used_memory_bytes
        self.available_memory_bytes = available_memory_bytes
        self.gpu_family = gpu_family
        self.loaded_models_count = loaded_models_count
        self.loaded_models: list[ModelInfo] = loaded_models or []

    @classmethod
    def _from_json_dict(cls, d: dict) -> "GetGPUStatusResponse":
        return cls(
            device_name=d.get("deviceName", ""),
            total_memory_bytes=d.get("totalMemoryBytes", 0),
            used_memory_bytes=d.get("usedMemoryBytes", 0),
            available_memory_bytes=d.get("availableMemoryBytes", 0),
            gpu_family=d.get("gpuFamily", ""),
            loaded_models_count=d.get("loadedModelsCount", 0),
            loaded_models=[ModelInfo._from_json_dict(m) for m in d.get("loadedModels", [])],
        )


class PingRequest(_JSONMessage):
    _JSON_FIELDS: list[tuple[str, str]] = []

    def __init__(self, **_: Any) -> None:
        pass

    @classmethod
    def _from_json_dict(cls, d: dict) -> "PingRequest":
        return cls()


class PingResponse(_JSONMessage):
    _JSON_FIELDS = [
        ("status", "status"),
        ("version", "version"),
        ("uptime_seconds", "uptimeSeconds"),
    ]

    def __init__(self, status: str = "", version: str = "",
                 uptime_seconds: float = 0.0, **_: Any) -> None:
        self.status = status
        self.version = version
        self.uptime_seconds = uptime_seconds

    @classmethod
    def _from_json_dict(cls, d: dict) -> "PingResponse":
        return cls(
            status=d.get("status", ""),
            version=d.get("version", ""),
            uptime_seconds=d.get("uptimeSeconds", 0.0),
        )
