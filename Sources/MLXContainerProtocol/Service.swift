import Foundation
import GRPCCore

// MARK: - JSON Serialization for gRPC

/// JSON serializer for Codable messages over gRPC.
public struct JSONMessageSerializer<Message: Codable & Sendable>: MessageSerializer, Sendable {
    public init() {}

    public func serialize<Bytes: GRPCContiguousBytes>(_ message: Message) throws -> Bytes {
        let data = try JSONEncoder().encode(message)
        return Bytes(data)
    }
}

/// JSON deserializer for Codable messages over gRPC.
public struct JSONMessageDeserializer<Message: Codable & Sendable>: MessageDeserializer, Sendable {
    public init() {}

    public func deserialize<Bytes: GRPCContiguousBytes>(_ serializedMessageBytes: Bytes) throws -> Message {
        let data = serializedMessageBytes.withUnsafeBytes { Data($0) }
        return try JSONDecoder().decode(Message.self, from: data)
    }
}

// MARK: - Service Descriptor

/// The MLX Container Service namespace.
public enum MLXContainerService {
    public static let descriptor = ServiceDescriptor(
        fullyQualifiedService: "mlx_container.v1.MLXContainerService"
    )

    public enum Method {
        public static let loadModel = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "LoadModel"
        )
        public static let unloadModel = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "UnloadModel"
        )
        public static let listModels = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "ListModels"
        )
        public static let generate = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "Generate"
        )
        public static let embed = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "Embed"
        )
        public static let getGPUStatus = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "GetGPUStatus"
        )
        public static let ping = MethodDescriptor(
            service: MLXContainerService.descriptor,
            method: "Ping"
        )
    }
}

// MARK: - Server Protocol

/// Protocol for implementing the MLX Container Service server.
public protocol MLXContainerServiceProtocol: RegistrableRPCService {
    func loadModel(
        request: MLXContainer_LoadModelRequest,
        context: ServerContext
    ) async throws -> MLXContainer_LoadModelResponse

    func unloadModel(
        request: MLXContainer_UnloadModelRequest,
        context: ServerContext
    ) async throws -> MLXContainer_UnloadModelResponse

    func listModels(
        request: MLXContainer_ListModelsRequest,
        context: ServerContext
    ) async throws -> MLXContainer_ListModelsResponse

    func generate(
        request: MLXContainer_GenerateRequest,
        context: ServerContext,
        responseWriter: RPCWriter<MLXContainer_GenerateResponse>
    ) async throws

    func embed(
        request: MLXContainer_EmbedRequest,
        context: ServerContext
    ) async throws -> MLXContainer_EmbedResponse

    func getGPUStatus(
        request: MLXContainer_GetGPUStatusRequest,
        context: ServerContext
    ) async throws -> MLXContainer_GetGPUStatusResponse

    func ping(
        request: MLXContainer_PingRequest,
        context: ServerContext
    ) async throws -> MLXContainer_PingResponse
}

/// Extract the first message from a streaming request (for unary RPCs).
private func firstMessage<M: Sendable>(from request: StreamingServerRequest<M>) async throws -> M {
    var iterator = request.messages.makeAsyncIterator()
    guard let message = try await iterator.next() else {
        throw RPCError(code: .invalidArgument, message: "No message received")
    }
    return message
}

extension MLXContainerServiceProtocol {
    public func registerMethods<Transport: ServerTransport>(with router: inout RPCRouter<Transport>) {
        // LoadModel - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.loadModel,
            deserializer: JSONMessageDeserializer<MLXContainer_LoadModelRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_LoadModelResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.loadModel(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }

        // UnloadModel - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.unloadModel,
            deserializer: JSONMessageDeserializer<MLXContainer_UnloadModelRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_UnloadModelResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.unloadModel(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }

        // ListModels - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.listModels,
            deserializer: JSONMessageDeserializer<MLXContainer_ListModelsRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_ListModelsResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.listModels(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }

        // Generate - server streaming
        router.registerHandler(
            forMethod: MLXContainerService.Method.generate,
            deserializer: JSONMessageDeserializer<MLXContainer_GenerateRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_GenerateResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            return StreamingServerResponse<MLXContainer_GenerateResponse>(
                metadata: [:],
                producer: { writer in
                    try await self.generate(request: message, context: context, responseWriter: writer)
                    return [:]
                }
            )
        }

        // Embed - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.embed,
            deserializer: JSONMessageDeserializer<MLXContainer_EmbedRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_EmbedResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.embed(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }

        // GetGPUStatus - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.getGPUStatus,
            deserializer: JSONMessageDeserializer<MLXContainer_GetGPUStatusRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_GetGPUStatusResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.getGPUStatus(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }

        // Ping - unary
        router.registerHandler(
            forMethod: MLXContainerService.Method.ping,
            deserializer: JSONMessageDeserializer<MLXContainer_PingRequest>(),
            serializer: JSONMessageSerializer<MLXContainer_PingResponse>()
        ) { request, context in
            let message = try await firstMessage(from: request)
            let response = try await self.ping(request: message, context: context)
            return StreamingServerResponse(single: .init(message: response))
        }
    }
}
