import ArgumentParser
import Foundation
import Logging
import MLXContainerConfig
import MLXDeviceDiscovery

@main
struct DaemonMain: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-container-daemon",
        abstract: "MLX Container GPU Daemon — serves inference over vsock to Linux containers"
    )

    @Option(name: .long, help: "vsock port to listen on")
    var port: UInt32?

    @Option(name: .long, help: "Path to config file")
    var config: String?

    @Option(name: .long, help: "Log level (trace, debug, info, warning, error)")
    var logLevel: String = "info"

    @Option(name: .long, help: "Model to pre-load on startup")
    var preloadModel: String?

    @Flag(name: .long, help: "Use TCP instead of vsock (for local development)")
    var tcp: Bool = false

    @Option(name: .long, help: "TCP port when using --tcp (default: 50051)")
    var tcpPort: Int = 50051

    func run() async throws {
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardError(label: label)
            handler.logLevel = Logger.Level(rawValue: logLevel) ?? .info
            return handler
        }
        let logger = Logger(label: "com.aiflowlabs.mlx-container-daemon")

        // Load config
        let configPath = config.map { URL(fileURLWithPath: $0) }
        var toolkitConfig = try ToolkitConfiguration.load(from: configPath)
        if let p = port {
            toolkitConfig.vsockPort = p
        }

        // Discover GPU
        let devices = DeviceDiscovery.discover()
        guard let gpu = devices.first else {
            logger.critical("No Apple GPU found. Cannot start daemon.")
            throw ExitCode.failure
        }
        logger.info("GPU: \(gpu.name), Memory: \(gpu.unifiedMemoryBytes / (1024*1024*1024)) GB")

        // Create and start the inference server
        let server = MLXInferenceServer(
            config: toolkitConfig,
            gpu: gpu,
            logger: logger
        )

        // Pre-load model if requested
        if let modelID = preloadModel {
            logger.info("Pre-loading model: \(modelID)")
            try await server.modelManager.loadModel(id: modelID)
        }

        // Set up graceful shutdown on SIGTERM / SIGINT.
        // DispatchSource signal handlers must run on a serial queue; we use the
        // main queue and capture a Task to cancel the serve work.
        let serveTask: Task<Void, Error>
        if tcp {
            logger.info("Starting gRPC server on TCP localhost:\(tcpPort)")
            serveTask = Task { try await server.serveTCP(port: tcpPort) }
        } else {
            logger.info("Starting gRPC server on vsock port \(toolkitConfig.vsockPort)")
            serveTask = Task { try await server.serve() }
        }

        // Ignore default signal disposition so DispatchSource can intercept them.
        signal(SIGTERM, SIG_IGN)
        signal(SIGINT, SIG_IGN)

        let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        let intSource  = DispatchSource.makeSignalSource(signal: SIGINT,  queue: .main)

        let shutdownHandler: @Sendable () -> Void = {
            logger.info("Shutting down...")
            // Unload all models before cancelling the serve task so MLX
            // releases GPU memory cleanly before the process exits.
            Task {
                await server.modelManager.unloadAll()
                serveTask.cancel()
            }
        }

        termSource.setEventHandler(handler: shutdownHandler)
        intSource.setEventHandler(handler: shutdownHandler)
        termSource.resume()
        intSource.resume()

        // Block until the serve task finishes (either naturally or after cancellation).
        do {
            try await serveTask.value
        } catch is CancellationError {
            logger.info("Daemon stopped gracefully")
        }

        termSource.cancel()
        intSource.cancel()
    }
}
