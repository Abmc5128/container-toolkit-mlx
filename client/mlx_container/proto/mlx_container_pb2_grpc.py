"""
gRPC client stub for MLXContainerService.

Serialisation strategy: JSON-over-gRPC to match the Swift server's
JSONMessageSerializer / JSONMessageDeserializer (see
MLXContainerProtocol/Service.swift).

Every request message is serialised by calling its ``SerializeToString()``
which produces UTF-8 JSON with camelCase keys (matching Swift's JSONEncoder
defaults).  Every response frame is deserialised by calling the matching
class's ``FromString(data)`` which parses camelCase keys back to Python attrs.
"""

from __future__ import annotations

import grpc

from mlx_container.proto import mlx_container_pb2 as pb2


# ---------------------------------------------------------------------------
# Serialiser / deserialiser helpers
# ---------------------------------------------------------------------------

def _serialize(msg: pb2._JSONMessage) -> bytes:
    """Serialise a request message to JSON bytes."""
    return msg.SerializeToString()


def _make_deserializer(cls: type[pb2._JSONMessage]):
    """Return a deserializer callable for the given message class."""
    def _deserialize(data: bytes) -> pb2._JSONMessage:
        return cls.FromString(data)
    return _deserialize


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------

class MLXContainerServiceStub:
    """
    Client stub for the MLX Container Service.

    Wraps a ``grpc.Channel`` and exposes one callable per RPC method.
    Each callable accepts the matching ``pb2.*Request`` object and returns
    the matching ``pb2.*Response`` object (or an iterator for server-streaming
    RPCs).
    """

    def __init__(self, channel: grpc.Channel) -> None:
        self.LoadModel = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/LoadModel",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.LoadModelResponse),
        )
        self.UnloadModel = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/UnloadModel",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.UnloadModelResponse),
        )
        self.ListModels = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/ListModels",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.ListModelsResponse),
        )
        self.Generate = channel.unary_stream(
            "/mlx_container.v1.MLXContainerService/Generate",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.GenerateResponse),
        )
        self.Embed = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/Embed",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.EmbedResponse),
        )
        self.GetGPUStatus = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/GetGPUStatus",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.GetGPUStatusResponse),
        )
        self.Ping = channel.unary_unary(
            "/mlx_container.v1.MLXContainerService/Ping",
            request_serializer=_serialize,
            response_deserializer=_make_deserializer(pb2.PingResponse),
        )
