"""
JSON-over-gRPC message classes and client stub for MLXContainerService.

These are hand-written to match the Swift server's Codable structs
(MLXContainerProtocol/Messages.swift) which use Swift's JSONEncoder —
camelCase keys by default.

To regenerate from proto/mlx_container.proto instead (requires protoc):
    python -m grpc_tools.protoc -Iproto --python_out=client/mlx_container/proto \\
        --grpc_python_out=client/mlx_container/proto proto/mlx_container.proto

Note: standard protoc output would require the Swift server to switch to
binary protobuf encoding, which it currently does not use.
"""
