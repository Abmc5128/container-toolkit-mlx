import Testing
import Foundation
@testable import MLXContainerProtocol

// MARK: - Helpers

private func roundtrip<T: Codable & Equatable>(_ value: T) throws -> T {
    let data = try JSONEncoder().encode(value)
    return try JSONDecoder().decode(T.self, from: data)
}

private func roundtrip<T: Codable>(_ value: T) throws -> T {
    let data = try JSONEncoder().encode(value)
    return try JSONDecoder().decode(T.self, from: data)
}

// MARK: - Model Management Protocol Tests

@Suite("MLXContainer Model Management Protocol Tests")
struct ModelManagementProtocolTests {

    @Test("LoadModelRequest Codable roundtrip preserves all fields")
    func loadModelRequestRoundtrip() throws {
        let original = MLXContainer_LoadModelRequest(
            modelID: "mlx-community/Llama-3.2-1B-4bit",
            alias: "llama-small",
            memoryBudgetBytes: 4_000_000_000
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(MLXContainer_LoadModelRequest.self, from: data)

        #expect(decoded.modelID == original.modelID)
        #expect(decoded.alias == original.alias)
        #expect(decoded.memoryBudgetBytes == original.memoryBudgetBytes)
    }

    @Test("LoadModelRequest default values roundtrip correctly")
    func loadModelRequestDefaults() throws {
        let original = MLXContainer_LoadModelRequest()
        let decoded: MLXContainer_LoadModelRequest = try roundtrip(original)
        #expect(decoded.modelID == "")
        #expect(decoded.alias == "")
        #expect(decoded.memoryBudgetBytes == 0)
    }

    @Test("LoadModelResponse Codable roundtrip preserves all fields")
    func loadModelResponseRoundtrip() throws {
        let original = MLXContainer_LoadModelResponse(
            success: true,
            modelID: "mlx-community/Llama-3.2-1B-4bit",
            error: "",
            memoryUsedBytes: 1_500_000_000,
            loadTimeSeconds: 3.14
        )
        let decoded: MLXContainer_LoadModelResponse = try roundtrip(original)

        #expect(decoded.success == true)
        #expect(decoded.modelID == original.modelID)
        #expect(decoded.error == "")
        #expect(decoded.memoryUsedBytes == original.memoryUsedBytes)
        #expect(abs(decoded.loadTimeSeconds - original.loadTimeSeconds) < 0.001)
    }

    @Test("LoadModelResponse with error field roundtrips correctly")
    func loadModelResponseWithError() throws {
        let original = MLXContainer_LoadModelResponse(
            success: false,
            modelID: "bad-model",
            error: "Model not found",
            memoryUsedBytes: 0,
            loadTimeSeconds: 0
        )
        let decoded: MLXContainer_LoadModelResponse = try roundtrip(original)
        #expect(decoded.success == false)
        #expect(decoded.error == "Model not found")
    }

    @Test("UnloadModelRequest Codable roundtrip preserves modelID")
    func unloadModelRequestRoundtrip() throws {
        let original = MLXContainer_UnloadModelRequest(modelID: "mlx-community/Llama-3.2-1B-4bit")
        let decoded: MLXContainer_UnloadModelRequest = try roundtrip(original)
        #expect(decoded.modelID == original.modelID)
    }

    @Test("UnloadModelResponse Codable roundtrip preserves all fields")
    func unloadModelResponseRoundtrip() throws {
        let original = MLXContainer_UnloadModelResponse(
            success: true,
            error: "",
            memoryFreedBytes: 1_200_000_000
        )
        let decoded: MLXContainer_UnloadModelResponse = try roundtrip(original)
        #expect(decoded.success == true)
        #expect(decoded.memoryFreedBytes == original.memoryFreedBytes)
    }

    @Test("ListModelsRequest encodes and decodes as empty object")
    func listModelsRequestRoundtrip() throws {
        let original = MLXContainer_ListModelsRequest()
        let data = try JSONEncoder().encode(original)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?.isEmpty == true, "ListModelsRequest should encode as an empty JSON object")
        // Decode should not throw
        _ = try JSONDecoder().decode(MLXContainer_ListModelsRequest.self, from: data)
    }

    @Test("ListModelsResponse Codable roundtrip with multiple models")
    func listModelsResponseRoundtrip() throws {
        let models = [
            MLXContainer_ModelInfo(modelID: "model-a", alias: "a", memoryUsedBytes: 100, isLoaded: true, modelType: "llm"),
            MLXContainer_ModelInfo(modelID: "model-b", alias: "b", memoryUsedBytes: 200, isLoaded: false, modelType: "llm"),
        ]
        let original = MLXContainer_ListModelsResponse(models: models)
        let decoded: MLXContainer_ListModelsResponse = try roundtrip(original)
        #expect(decoded.models.count == 2)
        #expect(decoded.models[0].modelID == "model-a")
        #expect(decoded.models[1].modelID == "model-b")
    }

    @Test("ListModelsResponse with empty models array roundtrips correctly")
    func listModelsResponseEmpty() throws {
        let original = MLXContainer_ListModelsResponse(models: [])
        let decoded: MLXContainer_ListModelsResponse = try roundtrip(original)
        #expect(decoded.models.isEmpty)
    }

    @Test("ModelInfo Codable roundtrip preserves all fields")
    func modelInfoRoundtrip() throws {
        let original = MLXContainer_ModelInfo(
            modelID: "mlx-community/Qwen2.5-1.5B-4bit",
            alias: "qwen-small",
            memoryUsedBytes: 750_000_000,
            isLoaded: true,
            modelType: "llm"
        )
        let decoded: MLXContainer_ModelInfo = try roundtrip(original)
        #expect(decoded.modelID == original.modelID)
        #expect(decoded.alias == original.alias)
        #expect(decoded.memoryUsedBytes == original.memoryUsedBytes)
        #expect(decoded.isLoaded == original.isLoaded)
        #expect(decoded.modelType == original.modelType)
    }
}

// MARK: - Inference Protocol Tests

@Suite("MLXContainer Inference Protocol Tests")
struct InferenceProtocolTests {

    @Test("ChatMessage Codable roundtrip preserves role and content")
    func chatMessageRoundtrip() throws {
        let original = MLXContainer_ChatMessage(role: "user", content: "Hello, world!")
        let decoded: MLXContainer_ChatMessage = try roundtrip(original)
        #expect(decoded.role == original.role)
        #expect(decoded.content == original.content)
    }

    @Test("ChatMessage default init has empty strings")
    func chatMessageDefaults() throws {
        let original = MLXContainer_ChatMessage()
        let decoded: MLXContainer_ChatMessage = try roundtrip(original)
        #expect(decoded.role == "")
        #expect(decoded.content == "")
    }

    @Test("GenerateParameters Codable roundtrip preserves all fields")
    func generateParametersRoundtrip() throws {
        let original = MLXContainer_GenerateParameters(
            maxTokens: 512,
            temperature: 0.7,
            topP: 0.9,
            repetitionPenalty: 1.1,
            repetitionContextSize: 20
        )
        let decoded: MLXContainer_GenerateParameters = try roundtrip(original)
        #expect(decoded.maxTokens == original.maxTokens)
        #expect(abs(decoded.temperature - original.temperature) < 0.001)
        #expect(abs(decoded.topP - original.topP) < 0.001)
        #expect(abs(decoded.repetitionPenalty - original.repetitionPenalty) < 0.001)
        #expect(decoded.repetitionContextSize == original.repetitionContextSize)
    }

    @Test("GenerateRequest Codable roundtrip preserves nested messages and parameters")
    func generateRequestRoundtrip() throws {
        let messages = [
            MLXContainer_ChatMessage(role: "system", content: "You are a helpful assistant."),
            MLXContainer_ChatMessage(role: "user", content: "What is 2+2?"),
        ]
        let params = MLXContainer_GenerateParameters(
            maxTokens: 256,
            temperature: 0.5,
            topP: 1.0,
            repetitionPenalty: 1.0,
            repetitionContextSize: 64
        )
        let original = MLXContainer_GenerateRequest(
            modelID: "mlx-community/Llama-3.2-3B-4bit",
            prompt: "",
            messages: messages,
            parameters: params,
            containerID: "container-007"
        )
        let decoded: MLXContainer_GenerateRequest = try roundtrip(original)

        #expect(decoded.modelID == original.modelID)
        #expect(decoded.containerID == original.containerID)
        #expect(decoded.messages.count == 2)
        #expect(decoded.messages[0].role == "system")
        #expect(decoded.messages[1].content == "What is 2+2?")
        #expect(decoded.parameters.maxTokens == 256)
        #expect(abs(decoded.parameters.temperature - 0.5) < 0.001)
    }

    @Test("GenerateRequest with prompt (no messages) roundtrips correctly")
    func generateRequestWithPrompt() throws {
        let original = MLXContainer_GenerateRequest(
            modelID: "mlx-community/SmolLM2-135M-4bit",
            prompt: "Once upon a time",
            messages: [],
            parameters: .init(),
            containerID: ""
        )
        let decoded: MLXContainer_GenerateRequest = try roundtrip(original)
        #expect(decoded.prompt == "Once upon a time")
        #expect(decoded.messages.isEmpty)
    }

    @Test("GenerateResponse with token field roundtrips correctly")
    func generateResponseTokenVariant() throws {
        let original = MLXContainer_GenerateResponse(token: "Hello")
        let decoded: MLXContainer_GenerateResponse = try roundtrip(original)
        #expect(decoded.token == "Hello")
        #expect(decoded.complete == nil)
    }

    @Test("GenerateResponse with complete field roundtrips correctly")
    func generateResponseCompleteVariant() throws {
        let complete = MLXContainer_GenerateComplete(
            fullText: "Hello, world!",
            promptTokens: 10,
            completionTokens: 5,
            promptTimeSeconds: 0.05,
            generationTimeSeconds: 1.2,
            tokensPerSecond: 42.0
        )
        let original = MLXContainer_GenerateResponse(complete: complete)
        let decoded: MLXContainer_GenerateResponse = try roundtrip(original)

        #expect(decoded.token == nil)
        #expect(decoded.complete != nil)
        #expect(decoded.complete?.fullText == "Hello, world!")
        #expect(decoded.complete?.promptTokens == 10)
        #expect(decoded.complete?.completionTokens == 5)
        #expect(abs((decoded.complete?.tokensPerSecond ?? 0) - 42.0) < 0.001)
    }

    @Test("GenerateResponse empty init has nil token and nil complete")
    func generateResponseEmpty() throws {
        let original = MLXContainer_GenerateResponse()
        let decoded: MLXContainer_GenerateResponse = try roundtrip(original)
        #expect(decoded.token == nil)
        #expect(decoded.complete == nil)
    }

    @Test("GenerateComplete Codable roundtrip preserves timing fields")
    func generateCompleteRoundtrip() throws {
        let original = MLXContainer_GenerateComplete(
            fullText: "The answer is 42.",
            promptTokens: 8,
            completionTokens: 4,
            promptTimeSeconds: 0.01,
            generationTimeSeconds: 0.5,
            tokensPerSecond: 8.0
        )
        let decoded: MLXContainer_GenerateComplete = try roundtrip(original)
        #expect(decoded.fullText == original.fullText)
        #expect(decoded.promptTokens == original.promptTokens)
        #expect(decoded.completionTokens == original.completionTokens)
        #expect(abs(decoded.promptTimeSeconds - original.promptTimeSeconds) < 0.0001)
        #expect(abs(decoded.generationTimeSeconds - original.generationTimeSeconds) < 0.0001)
        #expect(abs(decoded.tokensPerSecond - original.tokensPerSecond) < 0.001)
    }

    @Test("EmbedRequest Codable roundtrip preserves texts array")
    func embedRequestRoundtrip() throws {
        let original = MLXContainer_EmbedRequest(
            modelID: "mlx-community/bge-small-en-v1.5",
            texts: ["Hello", "World", "Embeddings"],
            containerID: "ctr-embed"
        )
        let decoded: MLXContainer_EmbedRequest = try roundtrip(original)
        #expect(decoded.modelID == original.modelID)
        #expect(decoded.texts == original.texts)
        #expect(decoded.containerID == original.containerID)
    }

    @Test("Embedding Codable roundtrip preserves float values")
    func embeddingRoundtrip() throws {
        let original = MLXContainer_Embedding(values: [0.1, 0.2, 0.3, -0.5, 1.0])
        let decoded: MLXContainer_Embedding = try roundtrip(original)
        #expect(decoded.values.count == 5)
        for (a, b) in zip(decoded.values, original.values) {
            #expect(abs(a - b) < 0.0001)
        }
    }

    @Test("EmbedResponse Codable roundtrip preserves nested embeddings")
    func embedResponseRoundtrip() throws {
        let original = MLXContainer_EmbedResponse(
            embeddings: [
                MLXContainer_Embedding(values: [0.1, 0.2]),
                MLXContainer_Embedding(values: [0.3, 0.4]),
            ],
            error: ""
        )
        let decoded: MLXContainer_EmbedResponse = try roundtrip(original)
        #expect(decoded.embeddings.count == 2)
        #expect(decoded.error == "")
    }
}

// MARK: - Health & Status Protocol Tests

@Suite("MLXContainer Health Protocol Tests")
struct HealthProtocolTests {

    @Test("PingRequest encodes and decodes as empty object")
    func pingRequestRoundtrip() throws {
        let original = MLXContainer_PingRequest()
        let data = try JSONEncoder().encode(original)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?.isEmpty == true, "PingRequest should encode as an empty JSON object")
        _ = try JSONDecoder().decode(MLXContainer_PingRequest.self, from: data)
    }

    @Test("PingResponse Codable roundtrip preserves all fields")
    func pingResponseRoundtrip() throws {
        let original = MLXContainer_PingResponse(
            status: "ok",
            version: "0.1.0",
            uptimeSeconds: 3600.5
        )
        let decoded: MLXContainer_PingResponse = try roundtrip(original)
        #expect(decoded.status == "ok")
        #expect(decoded.version == "0.1.0")
        #expect(abs(decoded.uptimeSeconds - 3600.5) < 0.001)
    }

    @Test("GetGPUStatusRequest encodes and decodes as empty object")
    func getGPUStatusRequestRoundtrip() throws {
        let original = MLXContainer_GetGPUStatusRequest()
        let data = try JSONEncoder().encode(original)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?.isEmpty == true, "GetGPUStatusRequest should encode as an empty JSON object")
        _ = try JSONDecoder().decode(MLXContainer_GetGPUStatusRequest.self, from: data)
    }

    @Test("GetGPUStatusResponse Codable roundtrip preserves all fields including nested models")
    func getGPUStatusResponseRoundtrip() throws {
        let loadedModels = [
            MLXContainer_ModelInfo(
                modelID: "mlx-community/Llama-3.2-1B-4bit",
                alias: "llama",
                memoryUsedBytes: 800_000_000,
                isLoaded: true,
                modelType: "llm"
            )
        ]
        let original = MLXContainer_GetGPUStatusResponse(
            deviceName: "Apple M3 Pro",
            totalMemoryBytes: 18_000_000_000,
            usedMemoryBytes: 800_000_000,
            availableMemoryBytes: 17_200_000_000,
            gpuFamily: "metal3",
            loadedModelsCount: 1,
            loadedModels: loadedModels
        )
        let decoded: MLXContainer_GetGPUStatusResponse = try roundtrip(original)
        #expect(decoded.deviceName == "Apple M3 Pro")
        #expect(decoded.totalMemoryBytes == original.totalMemoryBytes)
        #expect(decoded.usedMemoryBytes == original.usedMemoryBytes)
        #expect(decoded.availableMemoryBytes == original.availableMemoryBytes)
        #expect(decoded.gpuFamily == "metal3")
        #expect(decoded.loadedModelsCount == 1)
        #expect(decoded.loadedModels.count == 1)
        #expect(decoded.loadedModels[0].modelID == "mlx-community/Llama-3.2-1B-4bit")
    }
}

// MARK: - JSON Serializer Tests

@Suite("JSON Message Serializer / Deserializer Tests")
struct JSONSerializerTests {

    @Test("JSONMessageSerializer serializes a Codable message to valid JSON bytes")
    func serializerProducesValidJSON() throws {
        let serializer = JSONMessageSerializer<MLXContainer_PingRequest>()
        let msg = MLXContainer_PingRequest()
        let bytes: Data = try serializer.serialize(msg)
        // Should be valid JSON
        _ = try JSONSerialization.jsonObject(with: bytes)
    }

    @Test("JSONMessageDeserializer deserializes bytes back to a message")
    func deserializerRestoresMessage() throws {
        let serializer = JSONMessageSerializer<MLXContainer_PingResponse>()
        let deserializer = JSONMessageDeserializer<MLXContainer_PingResponse>()

        let original = MLXContainer_PingResponse(status: "ok", version: "1.0", uptimeSeconds: 99.0)
        let bytes: Data = try serializer.serialize(original)
        let restored: MLXContainer_PingResponse = try deserializer.deserialize(bytes)

        #expect(restored.status == "ok")
        #expect(restored.version == "1.0")
        #expect(abs(restored.uptimeSeconds - 99.0) < 0.001)
    }

    @Test("JSONMessageSerializer + JSONMessageDeserializer roundtrip for LoadModelRequest")
    func serializerDeserializerRoundtrip() throws {
        let serializer = JSONMessageSerializer<MLXContainer_LoadModelRequest>()
        let deserializer = JSONMessageDeserializer<MLXContainer_LoadModelRequest>()

        let original = MLXContainer_LoadModelRequest(
            modelID: "mlx-community/SmolLM2-360M-4bit",
            alias: "smollm",
            memoryBudgetBytes: 500_000_000
        )
        let bytes: Data = try serializer.serialize(original)
        let restored: MLXContainer_LoadModelRequest = try deserializer.deserialize(bytes)

        #expect(restored.modelID == original.modelID)
        #expect(restored.alias == original.alias)
        #expect(restored.memoryBudgetBytes == original.memoryBudgetBytes)
    }
}
