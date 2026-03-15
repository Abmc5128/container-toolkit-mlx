import Testing
import Foundation
@testable import MLXContainerConfig
@testable import MLXDeviceDiscovery

@Suite("ToolkitConfiguration Tests")
struct ToolkitConfigurationTests {

    // MARK: - Default values

    @Test("ToolkitConfiguration default vsockPort is 2048")
    func defaultVsockPort() {
        let config = ToolkitConfiguration()
        #expect(config.vsockPort == 2048)
        #expect(ToolkitConfiguration.defaultVsockPort == 2048)
    }

    @Test("ToolkitConfiguration default modelsDirectory is the expected path")
    func defaultModelsDirectory() {
        let config = ToolkitConfiguration()
        #expect(config.modelsDirectory == "~/.mlx-container/models")
        #expect(ToolkitConfiguration.defaultModelsDirectory == "~/.mlx-container/models")
    }

    @Test("ToolkitConfiguration default maxGPUMemoryBytes is 0 (unlimited)")
    func defaultMaxGPUMemoryBytes() {
        let config = ToolkitConfiguration()
        #expect(config.maxGPUMemoryBytes == 0)
    }

    @Test("ToolkitConfiguration default maxLoadedModels is 3")
    func defaultMaxLoadedModels() {
        let config = ToolkitConfiguration()
        #expect(config.maxLoadedModels == 3)
    }

    @Test("ToolkitConfiguration default logLevel is info")
    func defaultLogLevel() {
        let config = ToolkitConfiguration()
        #expect(config.logLevel == "info")
    }

    @Test("ToolkitConfiguration default enableStreaming is true")
    func defaultEnableStreaming() {
        let config = ToolkitConfiguration()
        #expect(config.enableStreaming == true)
    }

    @Test("ToolkitConfiguration default defaultMaxTokens is 512")
    func defaultMaxTokens() {
        let config = ToolkitConfiguration()
        #expect(config.defaultMaxTokens == 512)
    }

    @Test("ToolkitConfiguration default defaultTemperature is 0.7")
    func defaultTemperature() {
        let config = ToolkitConfiguration()
        #expect(abs(config.defaultTemperature - 0.7) < 0.001)
    }

    // MARK: - Custom initialisation

    @Test("ToolkitConfiguration custom init stores all provided values")
    func customInit() {
        let config = ToolkitConfiguration(
            vsockPort: 9090,
            modelsDirectory: "/tmp/models",
            maxGPUMemoryBytes: 8_000_000_000,
            maxLoadedModels: 5,
            logLevel: "debug",
            enableStreaming: false,
            defaultMaxTokens: 1024,
            defaultTemperature: 0.3
        )
        #expect(config.vsockPort == 9090)
        #expect(config.modelsDirectory == "/tmp/models")
        #expect(config.maxGPUMemoryBytes == 8_000_000_000)
        #expect(config.maxLoadedModels == 5)
        #expect(config.logLevel == "debug")
        #expect(config.enableStreaming == false)
        #expect(config.defaultMaxTokens == 1024)
        #expect(abs(config.defaultTemperature - 0.3) < 0.001)
    }

    // MARK: - resolvedModelsDirectory tilde expansion

    @Test("resolvedModelsDirectory expands tilde to home directory")
    func resolvedModelsDirectoryExpandsTilde() {
        let config = ToolkitConfiguration()
        let resolved = config.resolvedModelsDirectory
        let home = NSString(string: "~").expandingTildeInPath
        #expect(resolved.path.hasPrefix(home), "resolvedModelsDirectory should start with the home directory")
        #expect(!resolved.path.contains("~"), "resolvedModelsDirectory must not contain a literal tilde")
    }

    @Test("resolvedModelsDirectory with absolute path is returned as-is")
    func resolvedModelsDirectoryAbsolutePath() {
        var config = ToolkitConfiguration()
        config.modelsDirectory = "/var/lib/mlx/models"
        let resolved = config.resolvedModelsDirectory
        #expect(resolved.path == "/var/lib/mlx/models")
    }

    @Test("resolvedModelsDirectory ends with the relative path after the home directory")
    func resolvedModelsDirectoryPathSuffix() {
        let config = ToolkitConfiguration()
        let resolved = config.resolvedModelsDirectory
        #expect(resolved.path.hasSuffix(".mlx-container/models"))
    }

    // MARK: - Save / Load roundtrip

    @Test("ToolkitConfiguration saves to and loads from a temp file with identical values")
    func saveLoadRoundtrip() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let configURL = tmpDir.appendingPathComponent("test-config-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: configURL) }

        let original = ToolkitConfiguration(
            vsockPort: 3000,
            modelsDirectory: "/custom/models",
            maxGPUMemoryBytes: 4_000_000_000,
            maxLoadedModels: 2,
            logLevel: "warning",
            enableStreaming: false,
            defaultMaxTokens: 256,
            defaultTemperature: 0.5
        )

        try original.save(to: configURL)
        #expect(FileManager.default.fileExists(atPath: configURL.path), "Config file should exist after save")

        let loaded = try ToolkitConfiguration.load(from: configURL)

        #expect(loaded.vsockPort == original.vsockPort)
        #expect(loaded.modelsDirectory == original.modelsDirectory)
        #expect(loaded.maxGPUMemoryBytes == original.maxGPUMemoryBytes)
        #expect(loaded.maxLoadedModels == original.maxLoadedModels)
        #expect(loaded.logLevel == original.logLevel)
        #expect(loaded.enableStreaming == original.enableStreaming)
        #expect(loaded.defaultMaxTokens == original.defaultMaxTokens)
        #expect(abs(loaded.defaultTemperature - original.defaultTemperature) < 0.001)
    }

    @Test("ToolkitConfiguration load returns defaults when file does not exist")
    func loadNonExistentReturnsDefaults() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent-\(UUID().uuidString).json")
        let config = try ToolkitConfiguration.load(from: tmpURL)
        #expect(config.vsockPort == ToolkitConfiguration.defaultVsockPort)
        #expect(config.modelsDirectory == ToolkitConfiguration.defaultModelsDirectory)
        #expect(config.maxLoadedModels == 3)
    }

    @Test("ToolkitConfiguration save produces valid JSON")
    func savesValidJSON() throws {
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("valid-json-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        let config = ToolkitConfiguration()
        try config.save(to: tmpURL)

        let data = try Data(contentsOf: tmpURL)
        let json = try JSONSerialization.jsonObject(with: data, options: [])
        #expect(json is [String: Any], "Saved config must be a JSON object")
    }

    @Test("ToolkitConfiguration save creates intermediate directories")
    func saveCreatesIntermediateDirectories() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let nestedURL = tmpDir
            .appendingPathComponent("nested-\(UUID().uuidString)")
            .appendingPathComponent("deep")
            .appendingPathComponent("config.json")
        defer {
            let parent = nestedURL.deletingLastPathComponent().deletingLastPathComponent()
            try? FileManager.default.removeItem(at: parent)
        }

        let config = ToolkitConfiguration()
        try config.save(to: nestedURL)
        #expect(FileManager.default.fileExists(atPath: nestedURL.path))
    }
}

@Suite("ContainerGPUConfig Tests")
struct ContainerGPUConfigTests {

    // MARK: - Default values

    @Test("ContainerGPUConfig default init has enabled = true")
    func defaultEnabled() {
        let config = ContainerGPUConfig()
        #expect(config.enabled == true)
    }

    @Test("ContainerGPUConfig default memoryBudgetBytes is 0")
    func defaultMemoryBudget() {
        let config = ContainerGPUConfig()
        #expect(config.memoryBudgetBytes == 0)
    }

    @Test("ContainerGPUConfig default preloadModel is nil")
    func defaultPreloadModel() {
        let config = ContainerGPUConfig()
        #expect(config.preloadModel == nil)
    }

    @Test("ContainerGPUConfig default maxTokensPerRequest is 2048")
    func defaultMaxTokensPerRequest() {
        let config = ContainerGPUConfig()
        #expect(config.maxTokensPerRequest == 2048)
    }

    @Test("ContainerGPUConfig default allowModelManagement is true")
    func defaultAllowModelManagement() {
        let config = ContainerGPUConfig()
        #expect(config.allowModelManagement == true)
    }

    @Test("ContainerGPUConfig default containerID is nil")
    func defaultContainerID() {
        let config = ContainerGPUConfig()
        #expect(config.containerID == nil)
    }

    // MARK: - .disabled static

    @Test("ContainerGPUConfig.disabled has enabled = false")
    func disabledHasEnabledFalse() {
        let disabled = ContainerGPUConfig.disabled
        #expect(disabled.enabled == false)
    }

    @Test("ContainerGPUConfig.disabled has default memory budget of 0")
    func disabledHasZeroMemoryBudget() {
        let disabled = ContainerGPUConfig.disabled
        #expect(disabled.memoryBudgetBytes == 0)
    }

    @Test("ContainerGPUConfig.disabled has nil preloadModel")
    func disabledHasNilPreloadModel() {
        let disabled = ContainerGPUConfig.disabled
        #expect(disabled.preloadModel == nil)
    }

    // MARK: - Custom values

    @Test("ContainerGPUConfig stores custom containerID")
    func customContainerID() {
        let config = ContainerGPUConfig(containerID: "container-abc-123")
        #expect(config.containerID == "container-abc-123")
    }

    @Test("ContainerGPUConfig stores preloadModel when provided")
    func storesPreloadModel() {
        let config = ContainerGPUConfig(preloadModel: "mlx-community/Llama-3.2-1B-4bit")
        #expect(config.preloadModel == "mlx-community/Llama-3.2-1B-4bit")
    }

    // MARK: - Codable roundtrip

    @Test("ContainerGPUConfig encodes and decodes via JSON without data loss")
    func codableRoundtrip() throws {
        let original = ContainerGPUConfig(
            enabled: true,
            memoryBudgetBytes: 2_000_000_000,
            preloadModel: "mlx-community/Qwen2.5-1.5B-4bit",
            maxTokensPerRequest: 4096,
            allowModelManagement: false,
            containerID: "ctr-0042"
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ContainerGPUConfig.self, from: data)

        #expect(decoded.enabled == original.enabled)
        #expect(decoded.memoryBudgetBytes == original.memoryBudgetBytes)
        #expect(decoded.preloadModel == original.preloadModel)
        #expect(decoded.maxTokensPerRequest == original.maxTokensPerRequest)
        #expect(decoded.allowModelManagement == original.allowModelManagement)
        #expect(decoded.containerID == original.containerID)
    }

    @Test("ContainerGPUConfig.disabled Codable roundtrip preserves enabled=false")
    func disabledCodableRoundtrip() throws {
        let data = try JSONEncoder().encode(ContainerGPUConfig.disabled)
        let decoded = try JSONDecoder().decode(ContainerGPUConfig.self, from: data)
        #expect(decoded.enabled == false)
    }
}
