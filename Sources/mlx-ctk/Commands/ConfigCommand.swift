import ArgumentParser
import Foundation
import MLXContainerConfig

struct ConfigCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "Manage toolkit configuration",
        subcommands: [ShowConfig.self, SetConfig.self, ResetConfig.self],
        defaultSubcommand: ShowConfig.self
    )

    struct ShowConfig: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "show",
            abstract: "Show current configuration"
        )

        func run() async throws {
            let config = try ToolkitConfiguration.load()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(config)
            if let output = String(data: data, encoding: .utf8) {
                print(output)
            }
        }
    }

    struct SetConfig: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "set",
            abstract: "Set a configuration value"
        )

        @Option(name: .long, help: "vsock port for daemon")
        var vsockPort: UInt32?

        @Option(name: .long, help: "Directory for MLX models")
        var modelsDir: String?

        @Option(name: .long, help: "Max GPU memory in GB (0 = unlimited)")
        var maxGPUMemoryGB: UInt64?

        @Option(name: .long, help: "Max number of loaded models")
        var maxModels: Int?

        @Option(name: .long, help: "Default max tokens for generation")
        var defaultMaxTokens: Int?

        @Option(name: .long, help: "Default temperature for generation")
        var defaultTemperature: Float?

        func run() async throws {
            // Validate inputs before touching the stored config.
            if let port = vsockPort {
                guard port > 0, port < 65536 else {
                    fputs("error: vsock-port must be between 1 and 65535 (got \(port))\n", stderr)
                    throw ExitCode.failure
                }
            }
            if let mem = maxGPUMemoryGB {
                // UInt64 is always >= 0; guard kept for documentation clarity.
                _ = mem  // no further constraint beyond the type
            }
            if let max = maxModels {
                guard max > 0 else {
                    fputs("error: max-models must be greater than 0 (got \(max))\n", stderr)
                    throw ExitCode.failure
                }
            }
            if let tokens = defaultMaxTokens {
                guard tokens > 0, tokens <= 8192 else {
                    fputs("error: default-max-tokens must be between 1 and 8192 (got \(tokens))\n", stderr)
                    throw ExitCode.failure
                }
            }
            if let temp = defaultTemperature {
                guard temp >= 0, temp <= 2.0 else {
                    fputs("error: default-temperature must be between 0.0 and 2.0 (got \(temp))\n", stderr)
                    throw ExitCode.failure
                }
            }

            var config = try ToolkitConfiguration.load()

            if let port = vsockPort {
                config.vsockPort = port
            }
            if let dir = modelsDir {
                config.modelsDirectory = dir
            }
            if let mem = maxGPUMemoryGB {
                config.maxGPUMemoryBytes = mem * 1024 * 1024 * 1024
            }
            if let max = maxModels {
                config.maxLoadedModels = max
            }
            if let tokens = defaultMaxTokens {
                config.defaultMaxTokens = tokens
            }
            if let temp = defaultTemperature {
                config.defaultTemperature = temp
            }

            try config.save()
            print("Configuration updated.")
        }
    }

    struct ResetConfig: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "reset",
            abstract: "Reset configuration to defaults"
        )

        func run() async throws {
            let config = ToolkitConfiguration()
            try config.save()
            print("Configuration reset to defaults.")
        }
    }
}
