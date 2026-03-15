import ArgumentParser
import Foundation
import GRPCCore
import GRPCNIOTransportHTTP2
import MLXContainerProtocol

struct StatusCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Show GPU and model status (mlx-smi equivalent)",
        aliases: ["smi"]
    )

    @Option(name: .long, help: "Daemon TCP port (default: 50051)")
    var port: Int = 50051

    @Option(name: .long, help: "Daemon host (default: 127.0.0.1)")
    var host: String = "127.0.0.1"

    @Flag(name: .long, help: "Output as JSON")
    var json: Bool = false

    func run() async throws {
        do {
            let (gpuStatus, ping) = try await queryDaemon(host: host, port: port)

            if json {
                printJSON(gpuStatus: gpuStatus, ping: ping)
            } else {
                printTable(gpuStatus: gpuStatus, ping: ping)
            }
        } catch {
            if json {
                let errorObj: [String: Any] = [
                    "error": "Daemon not reachable",
                    "detail": error.localizedDescription,
                    "hint": "Run 'mlx-ctk service start' to start the daemon",
                ]
                if let data = try? JSONSerialization.data(withJSONObject: errorObj, options: .prettyPrinted),
                   let str = String(data: data, encoding: .utf8)
                {
                    print(str)
                }
            } else {
                print("ERROR: Cannot connect to MLX container daemon on \(host):\(port)")
                print("       \(error.localizedDescription)")
                print("")
                print("Is the daemon running?  mlx-ctk service status")
                print("Start the daemon:       mlx-ctk service start")
            }
            throw ExitCode.failure
        }
    }

    // MARK: - gRPC client

    private func queryDaemon(
        host: String,
        port: Int
    ) async throws -> (MLXContainer_GetGPUStatusResponse, MLXContainer_PingResponse) {
        try await withGRPCClient(
            transport: .http2NIOPosix(
                target: .ipv4(address: host, port: port),
                transportSecurity: .plaintext
            )
        ) { grpcClient in
            async let gpuStatus: MLXContainer_GetGPUStatusResponse = grpcClient.unary(
                request: ClientRequest(message: MLXContainer_GetGPUStatusRequest()),
                descriptor: MLXContainerService.Method.getGPUStatus,
                serializer: JSONMessageSerializer<MLXContainer_GetGPUStatusRequest>(),
                deserializer: JSONMessageDeserializer<MLXContainer_GetGPUStatusResponse>(),
                options: .defaults
            ) { try $0.message }
            async let ping: MLXContainer_PingResponse = grpcClient.unary(
                request: ClientRequest(message: MLXContainer_PingRequest()),
                descriptor: MLXContainerService.Method.ping,
                serializer: JSONMessageSerializer<MLXContainer_PingRequest>(),
                deserializer: JSONMessageDeserializer<MLXContainer_PingResponse>(),
                options: .defaults
            ) { try $0.message }
            return try await (gpuStatus, ping)
        }
    }

    // MARK: - Formatted table output

    private func printTable(
        gpuStatus: MLXContainer_GetGPUStatusResponse,
        ping: MLXContainer_PingResponse
    ) {
        let totalGB = formatGB(gpuStatus.totalMemoryBytes)
        let usedGB = formatGB(gpuStatus.usedMemoryBytes)
        let modelCount = gpuStatus.loadedModelsCount

        let metalLabel: String
        switch gpuStatus.gpuFamily {
        case "metal3":  metalLabel = "Metal 3"
        case "apple9":  metalLabel = "Metal 3"
        case "apple8":  metalLabel = "Metal 2"
        default:        metalLabel = "Metal"
        }

        let width = 43
        let rule = String(repeating: "\u{2550}", count: width)  // ═══
        let divider = String(repeating: "\u{2500}", count: width)  // ───

        print("MLX Container Toolkit v\(ping.version)  |  uptime \(formatUptime(ping.uptimeSeconds))")
        print(rule)
        print("GPU:     \(gpuStatus.deviceName)  |  \(metalLabel)")
        print("Memory:  \(totalGB) total  |  \(usedGB) used")
        print("Models:  \(modelCount) loaded")

        let loadedModels = gpuStatus.loadedModels.filter { $0.isLoaded }
        if !loadedModels.isEmpty {
            print(divider)
            for model in loadedModels {
                let memLabel = formatGB(model.memoryUsedBytes)
                let name = model.alias.isEmpty ? model.modelID : model.alias
                // Truncate name to fit in 43 chars with right-aligned memory
                let maxNameLen = width - memLabel.count - 2
                let truncated = name.count > maxNameLen
                    ? String(name.prefix(maxNameLen - 1)) + "\u{2026}"
                    : name
                let padding = String(repeating: " ", count: max(0, width - truncated.count - memLabel.count))
                print("  \(truncated)\(padding)\(memLabel)")
            }
        }

        print(rule)
    }

    // MARK: - JSON output

    private func printJSON(
        gpuStatus: MLXContainer_GetGPUStatusResponse,
        ping: MLXContainer_PingResponse
    ) {
        let models = gpuStatus.loadedModels.map { m -> [String: Any] in
            [
                "modelID": m.modelID,
                "alias": m.alias,
                "isLoaded": m.isLoaded,
                "memoryUsedBytes": m.memoryUsedBytes,
                "modelType": m.modelType,
            ]
        }
        let output: [String: Any] = [
            "daemonVersion": ping.version,
            "daemonStatus": ping.status,
            "uptimeSeconds": ping.uptimeSeconds,
            "gpu": [
                "deviceName": gpuStatus.deviceName,
                "gpuFamily": gpuStatus.gpuFamily,
                "totalMemoryBytes": gpuStatus.totalMemoryBytes,
                "usedMemoryBytes": gpuStatus.usedMemoryBytes,
                "availableMemoryBytes": gpuStatus.availableMemoryBytes,
                "loadedModelsCount": gpuStatus.loadedModelsCount,
            ] as [String: Any],
            "models": models,
        ]

        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8)
        {
            print(str)
        }
    }

    // MARK: - Formatting helpers

    private func formatGB(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / (1024 * 1024 * 1024)
        return String(format: "%.1f GB", gb)
    }

    private func formatUptime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        return "\(s / 3600)h \(s % 3600 / 60)m"
    }
}
