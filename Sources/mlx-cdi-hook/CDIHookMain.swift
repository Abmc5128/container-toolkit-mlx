import ArgumentParser
import Foundation
import Logging
import MLXContainerConfig
import Network

// MARK: - Constants

private enum HookConstants {
    static let pidFilePath = "~/.mlx-container/daemon.pid"
    static let daemonBinaryName = "mlx-container-daemon"
    static let daemonSearchPaths = [
        "/usr/local/bin/mlx-container-daemon",
        "/usr/bin/mlx-container-daemon",
        "/opt/homebrew/bin/mlx-container-daemon",
    ]
    static let vsockCID: UInt32 = 2
    /// Default TCP health-check port (gRPC TCP mode). Not the vsock port.
    static let defaultTCPPort: UInt32 = 50051
    static let daemonStartTimeoutSeconds: Double = 5.0
    static let healthCheckRetries = 3

    /// Resolve the port to probe in priority order:
    ///   1. Explicit CLI override (passed in)
    ///   2. `MLX_DAEMON_PORT` env var
    ///   3. Config file `vsockPort` when TCP mode is detected (port != defaultVsockPort)
    ///   4. Hard default: 50051 (TCP gRPC)
    static func resolveProbePort(cliOverride: UInt32?) -> UInt32 {
        if let p = cliOverride, p > 0 { return p }
        if let envStr = ProcessInfo.processInfo.environment["MLX_DAEMON_PORT"],
           let p = UInt32(envStr), p > 0 { return p }
        let config = (try? ToolkitConfiguration.load()) ?? ToolkitConfiguration()
        // If the config port equals the vsock default it is likely a vsock-only deployment;
        // we still probe via TCP on the gRPC default so the hook works without vsock kernel support.
        return config.vsockPort == ToolkitConfiguration.defaultVsockPort ? defaultTCPPort : config.vsockPort
    }
}

// MARK: - Entry Point

@main
struct CDIHook: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-cdi-hook",
        abstract: "OCI prestart hook for Apple MLX GPU access in containers",
        version: "0.1.0",
        subcommands: [StartDaemon.self, HealthCheck.self],
        defaultSubcommand: nil
    )
}

// MARK: - start-daemon

struct StartDaemon: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "start-daemon",
        abstract: "Ensure the MLX container daemon is running (called by container runtime)"
    )

    @Flag(name: .long, help: "Verbose output")
    var verbose = false

    @Option(name: .long, help: "TCP port to probe for daemon health (overrides config / MLX_DAEMON_PORT env var)")
    var port: UInt32?

    func run() async throws {
        var logger = Logger(label: "com.aiflowlabs.mlx-cdi-hook.start-daemon")
        logger.logLevel = verbose ? .debug : .info

        let probePort = HookConstants.resolveProbePort(cliOverride: port)
        logger.debug("Using probe port \(probePort)")

        let pidPath = NSString(string: HookConstants.pidFilePath).expandingTildeInPath

        // 1. Check if daemon is already running via PID file
        if let pid = readPID(at: pidPath) {
            if isProcessAlive(pid: pid) {
                logger.info("Daemon already running (PID \(pid))")
                // Still verify it is reachable
                if await verifyDaemonResponding(port: probePort, logger: logger) {
                    logger.info("Daemon health check passed")
                    return
                }
                logger.warning("Daemon PID \(pid) is alive but not responding — restarting")
                killProcess(pid: pid)
            } else {
                logger.info("Stale PID file found (PID \(pid) not alive) — removing")
            }
            try? FileManager.default.removeItem(atPath: pidPath)
        }

        // 2. Locate daemon binary
        guard let daemonURL = locateDaemon(logger: logger) else {
            logger.error("mlx-container-daemon not found in PATH or standard locations")
            fputs("mlx-cdi-hook: daemon binary not found\n", stderr)
            throw ExitCode.failure
        }

        logger.debug("Found daemon at \(daemonURL.path)")

        // 3. Start daemon in background
        let process = Process()
        process.executableURL = daemonURL
        // Daemon runs in background; redirect its output to a log file
        let logDir = NSString(string: "~/.mlx-container/logs").expandingTildeInPath
        try FileManager.default.createDirectory(
            atPath: logDir,
            withIntermediateDirectories: true
        )
        let logPath = (logDir as NSString).appendingPathComponent("daemon.log")
        FileManager.default.createFile(atPath: logPath, contents: nil)
        let logHandle = FileHandle(forWritingAtPath: logPath)
        process.standardOutput = logHandle
        process.standardError = logHandle

        do {
            try process.run()
        } catch {
            logger.error("Failed to launch daemon: \(error.localizedDescription)")
            fputs("mlx-cdi-hook: failed to launch daemon: \(error)\n", stderr)
            throw ExitCode.failure
        }

        let newPID = process.processIdentifier
        logger.info("Daemon started with PID \(newPID)")

        // 4. Write PID file
        let pidDir = (pidPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: pidDir, withIntermediateDirectories: true)
        try "\(newPID)\n".write(toFile: pidPath, atomically: true, encoding: .utf8)

        // 5. Wait for daemon to become ready
        let deadline = Date().addingTimeInterval(HookConstants.daemonStartTimeoutSeconds)
        var ready = false
        while Date() < deadline {
            if await verifyDaemonResponding(port: probePort, logger: logger) {
                ready = true
                break
            }
            try await Task.sleep(nanoseconds: 250_000_000) // 250ms
        }

        if ready {
            logger.info("Daemon is ready (PID \(newPID))")
        } else {
            logger.warning("Daemon started (PID \(newPID)) but did not respond within \(Int(HookConstants.daemonStartTimeoutSeconds))s — it may still be loading")
        }
    }
}

// MARK: - health-check

struct HealthCheck: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "health-check",
        abstract: "Check whether the MLX container daemon is running and healthy"
    )

    @Flag(name: .long, help: "Verbose output")
    var verbose = false

    @Option(name: .long, help: "TCP port to probe for daemon health (overrides config / MLX_DAEMON_PORT env var)")
    var port: UInt32?

    func run() async throws {
        var logger = Logger(label: "com.aiflowlabs.mlx-cdi-hook.health-check")
        logger.logLevel = verbose ? .debug : .info

        let probePort = HookConstants.resolveProbePort(cliOverride: port)
        logger.debug("Using probe port \(probePort)")

        let pidPath = NSString(string: HookConstants.pidFilePath).expandingTildeInPath

        guard let pid = readPID(at: pidPath) else {
            logger.error("No PID file found at \(pidPath)")
            fputs("mlx-cdi-hook: daemon not running (no PID file)\n", stderr)
            throw ExitCode.failure
        }

        guard isProcessAlive(pid: pid) else {
            logger.error("Daemon PID \(pid) is not alive")
            fputs("mlx-cdi-hook: daemon process \(pid) is dead\n", stderr)
            throw ExitCode.failure
        }

        logger.debug("PID \(pid) is alive — pinging daemon")

        var passed = false
        for attempt in 1...HookConstants.healthCheckRetries {
            if await verifyDaemonResponding(port: probePort, logger: logger) {
                passed = true
                break
            }
            logger.debug("Ping attempt \(attempt)/\(HookConstants.healthCheckRetries) failed")
            if attempt < HookConstants.healthCheckRetries {
                try await Task.sleep(nanoseconds: 200_000_000) // 200ms
            }
        }

        if passed {
            print("Daemon healthy (PID \(pid))")
            logger.info("Health check passed")
        } else {
            logger.error("Daemon PID \(pid) is alive but not responding")
            fputs("mlx-cdi-hook: daemon not responding\n", stderr)
            throw ExitCode.failure
        }
    }
}

// MARK: - Helpers

/// Read an integer PID from a file, returning nil if missing or unparseable.
private func readPID(at path: String) -> Int32? {
    guard let raw = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let pid = Int32(trimmed), pid > 0 else { return nil }
    return pid
}

/// Returns true if a process with the given PID exists (signal 0 check).
private func isProcessAlive(pid: Int32) -> Bool {
    kill(pid, 0) == 0
}

/// Send SIGTERM to a process.
private func killProcess(pid: Int32) {
    kill(pid, SIGTERM)
}

/// Try to connect to the daemon on localhost TCP (fallback from vsock for probe).
/// A real vsock ping would require kernel extensions unavailable to user binaries,
/// so we probe the gRPC TCP port that the daemon also listens on.
private func verifyDaemonResponding(port: UInt32, logger: Logger) async -> Bool {
    guard let nwPort = NWEndpoint.Port(rawValue: UInt16(clamping: port)) else { return false }

    let probe = TCPProbe(port: nwPort, logger: logger)
    return await probe.check()
}

/// Encapsulates a single non-blocking TCP reachability probe using NWConnection.
/// Wrapping state in an actor satisfies Swift 6 Sendable requirements.
private actor TCPProbe {
    private let port: NWEndpoint.Port
    private let logger: Logger
    private var continuation: CheckedContinuation<Bool, Never>?
    private var connection: NWConnection?
    private var settled = false

    init(port: NWEndpoint.Port, logger: Logger) {
        self.port = port
        self.logger = logger
    }

    func check() async -> Bool {
        await withCheckedContinuation { cont in
            self.continuation = cont
            let endpoint = NWEndpoint.hostPort(host: .ipv4(.loopback), port: self.port)
            let conn = NWConnection(to: endpoint, using: .tcp)
            self.connection = conn

            conn.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                Task { await self.handleState(state) }
            }
            conn.start(queue: .global())

            // Schedule a timeout via a detached Task
            let portValue = self.port.rawValue
            Task.detached {
                try? await Task.sleep(nanoseconds: 1_500_000_000) // 1.5s
                await self.settle(result: false, reason: "timeout on port \(portValue)")
            }
        }
    }

    private func handleState(_ state: NWConnection.State) {
        switch state {
        case .ready:
            settle(result: true, reason: "connected")
        case .failed(let error):
            settle(result: false, reason: "failed: \(error)")
        case .cancelled:
            settle(result: false, reason: "cancelled")
        default:
            break
        }
    }

    private func settle(result: Bool, reason: String) {
        guard !settled else { return }
        settled = true
        connection?.cancel()
        connection = nil
        logger.debug("TCP probe \(result ? "succeeded" : "failed") — \(reason)")
        continuation?.resume(returning: result)
        continuation = nil
    }
}

/// Search for the daemon binary in PATH and standard locations.
private func locateDaemon(logger: Logger) -> URL? {
    // 1. Standard fixed locations
    for path in HookConstants.daemonSearchPaths {
        if FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }
    }

    // 2. PATH search
    guard let pathEnv = ProcessInfo.processInfo.environment["PATH"] else { return nil }
    for dir in pathEnv.split(separator: ":") {
        let fullPath = "\(dir)/\(HookConstants.daemonBinaryName)"
        if FileManager.default.isExecutableFile(atPath: fullPath) {
            return URL(fileURLWithPath: fullPath)
        }
    }

    return nil
}
