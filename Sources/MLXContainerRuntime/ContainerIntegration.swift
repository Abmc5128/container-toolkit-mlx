import Foundation
import Logging
import MLXContainerConfig

// MARK: - Public API

/// Entry point for apple/container to activate GPU support when `--gpu` is passed.
///
/// `SandboxService` calls `configureGPU` once per container, right after reading
/// the container configuration from its bundle and before building the
/// `LinuxContainer.Configuration` closure.  The returned `GPUSetupResult` carries:
///
/// - environment variables to splice into the container process
/// - the vsock port the container should dial
/// - the PID of the host daemon (for logging / health checks)
/// - a reference to the shared `GPUDaemonLifecycle` actor, so `SandboxService`
///   can call `stopForContainer` when the container exits
///
/// Usage in SandboxService.swift:
///
/// ```swift
/// let gpuSetup = try await MLXContainerIntegration.configureGPU(
///     containerID: config.id,
///     gpuConfig: gpuConfig,
///     toolkitConfig: toolkitConfig,
///     vsockPort: vsockPort
/// )
/// // Inject env vars into czConfig.process.environmentVariables
/// // Store gpuSetup.lifecycle to call stopForContainer on exit
/// ```
public enum MLXContainerIntegration {

    // MARK: - Shared lifecycle actor

    /// Process-wide singleton so multiple containers share a single daemon.
    ///
    /// `SandboxService` is an `actor`; accessing this nonisolated property from
    /// within an actor context is safe because `GPUDaemonLifecycle` is itself
    /// an actor with its own internal isolation.
    private static let sharedLifecycle: GPUDaemonLifecycle = {
        let config = (try? ToolkitConfiguration.load()) ?? ToolkitConfiguration()
        var logger = Logger(label: "com.aiflowlabs.mlx-ctk.gpu-lifecycle")
        logger.logLevel = .info
        return GPUDaemonLifecycle(config: config, logger: logger)
    }()

    // MARK: - configureGPU

    /// Set up GPU support for a container that is about to launch.
    ///
    /// This function is the single call-site for apple/container.  It:
    ///
    /// 1. Validates that vsock is available on this macOS host.
    /// 2. Starts (or reuses) the MLX Container Daemon process.
    /// 3. Optionally waits for the daemon to be ready when a pre-load model is
    ///    specified, giving the daemon up to `daemonReadyTimeout` seconds.
    /// 4. Returns a `GPUSetupResult` containing environment variables and
    ///    metadata the caller needs to finish container configuration.
    ///
    /// - Parameters:
    ///   - containerID: Unique identifier for the container (used for lifecycle tracking).
    ///   - gpuConfig: Per-container GPU settings decoded from the `mlx.container.gpu` label.
    ///   - toolkitConfig: Global daemon settings (vsock port, model cache directory, etc.).
    ///   - vsockPort: Override vsock port; takes precedence over `toolkitConfig.vsockPort`
    ///     when non-zero.  Defaults to `ToolkitConfiguration.defaultVsockPort`.
    ///   - daemonReadyTimeout: Seconds to wait for daemon readiness after starting it.
    ///
    /// - Returns: A `GPUSetupResult` with env vars, the vsock port, the daemon PID,
    ///   and a reference to the shared `GPUDaemonLifecycle` actor.
    ///
    /// - Throws: `GPUDaemonError.daemonNotFound` if the daemon binary cannot be located.
    ///   `GPUIntegrationError.vsockUnavailable` if the host does not support vsock.
    ///   `GPUIntegrationError.daemonTimeout` if the daemon does not become ready in time.
    public static func configureGPU(
        containerID: String,
        gpuConfig: ContainerGPUConfig,
        toolkitConfig: ToolkitConfiguration,
        vsockPort: UInt32 = ToolkitConfiguration.defaultVsockPort,
        daemonReadyTimeout: TimeInterval = 10
    ) async throws -> GPUSetupResult {
        guard gpuConfig.enabled else {
            return GPUSetupResult(
                environmentVariables: [:],
                vsockPort: vsockPort,
                daemonPID: -1,
                lifecycle: sharedLifecycle
            )
        }

        guard GPUVsockRelay.validateVsockSupport() else {
            throw GPUIntegrationError.vsockUnavailable
        }

        // Resolve effective port: caller override > gpuConfig label > toolkitConfig default
        let effectivePort: UInt32 = vsockPort != 0 ? vsockPort : toolkitConfig.vsockPort

        // Reconcile per-container GPU config with global toolkit config.
        // Memory: respect per-container budget; fall back to global max.
        var resolvedConfig = gpuConfig
        resolvedConfig.containerID = containerID
        if resolvedConfig.memoryBudgetBytes == 0 && toolkitConfig.maxGPUMemoryBytes > 0 {
            resolvedConfig.memoryBudgetBytes = toolkitConfig.maxGPUMemoryBytes
        }

        // Start daemon (idempotent — lifecycle actor guards double-start).
        try await sharedLifecycle.startForContainer(
            containerID: containerID,
            gpuConfig: resolvedConfig
        )

        // If a model was requested, give the daemon time to load it before the
        // container's init process runs and tries to make inference calls.
        if resolvedConfig.preloadModel != nil {
            try await waitForDaemonReady(
                lifecycle: sharedLifecycle,
                timeout: daemonReadyTimeout,
                containerID: containerID
            )
        }

        let daemonPID = await sharedLifecycle.daemonPID

        let envVars = buildEnvironmentVariables(
            gpuConfig: resolvedConfig,
            vsockPort: effectivePort,
            toolkitConfig: toolkitConfig
        )

        return GPUSetupResult(
            environmentVariables: envVars,
            vsockPort: effectivePort,
            daemonPID: daemonPID,
            lifecycle: sharedLifecycle
        )
    }

    // MARK: - Result type

    /// Output of `configureGPU`.  Consumed by `SandboxService` to finish
    /// building the container configuration.
    public struct GPUSetupResult: Sendable {
        /// Environment variables to inject into `czConfig.process.environmentVariables`.
        ///
        /// Keys and their semantics:
        /// - `MLX_GPU_ENABLED`    — `"1"` when GPU is active
        /// - `MLX_VSOCK_PORT`     — port the in-container SDK dials
        /// - `MLX_VSOCK_CID`      — host vsock CID (`"2"`)
        /// - `MLX_GPU_MEMORY_GB`  — memory budget in whole GB (`"0"` = unlimited)
        /// - `MLX_MAX_TOKENS`     — default token cap per request
        public let environmentVariables: [String: String]

        /// The vsock port the container should dial to reach the daemon.
        public let vsockPort: UInt32

        /// PID of the host daemon process, or `-1` if the daemon was not started
        /// (e.g. GPU is disabled).
        public let daemonPID: Int32

        /// Reference to the shared lifecycle actor.  `SandboxService` should
        /// call `lifecycle.stopForContainer(containerID:)` when the container exits.
        public let lifecycle: GPUDaemonLifecycle
    }
}

// MARK: - Private helpers

extension MLXContainerIntegration {

    /// Build the set of environment variables to inject into the container.
    private static func buildEnvironmentVariables(
        gpuConfig: ContainerGPUConfig,
        vsockPort: UInt32,
        toolkitConfig: ToolkitConfiguration
    ) -> [String: String] {
        var env: [String: String] = [:]

        env["MLX_GPU_ENABLED"] = "1"
        env["MLX_VSOCK_PORT"] = String(vsockPort)
        env["MLX_VSOCK_CID"] = String(GPUVsockRelay.hostCID)

        // Memory budget as whole GB (round down; 0 means unlimited)
        let memGB = gpuConfig.memoryBudgetBytes / 1_073_741_824
        env["MLX_GPU_MEMORY_GB"] = String(memGB)

        // Token limit: prefer per-container setting, fall back to toolkit default
        let maxTokens = gpuConfig.maxTokensPerRequest > 0
            ? gpuConfig.maxTokensPerRequest
            : toolkitConfig.defaultMaxTokens
        env["MLX_MAX_TOKENS"] = String(maxTokens)

        // Optional: advertise the pre-loaded model ID so the in-container SDK
        // can skip an explicit model-load call.
        if let model = gpuConfig.preloadModel {
            env["MLX_DEFAULT_MODEL"] = model
        }

        return env
    }

    /// Poll `GPUDaemonLifecycle.isRunning` until true or timeout expires.
    ///
    /// A real production implementation would connect to the daemon's health
    /// endpoint over vsock.  This polling approach is safe for the current
    /// `Process`-based daemon because `isRunning` reflects OS process state.
    private static func waitForDaemonReady(
        lifecycle: GPUDaemonLifecycle,
        timeout: TimeInterval,
        containerID: String
    ) async throws {
        let deadline = Date(timeIntervalSinceNow: timeout)
        let pollInterval: UInt64 = 200_000_000  // 200 ms in nanoseconds

        while Date() < deadline {
            let running = await lifecycle.isRunning
            if running { return }
            try await Task.sleep(nanoseconds: pollInterval)
        }

        // Final check — daemon might have become ready in the last slice
        let finalCheck = await lifecycle.isRunning
        guard finalCheck else {
            throw GPUIntegrationError.daemonTimeout(seconds: timeout, containerID: containerID)
        }
    }
}

// MARK: - Errors

/// Errors specific to the apple/container integration layer.
public enum GPUIntegrationError: Error, LocalizedError {
    /// The host OS does not support vsock (should not happen on macOS 13+).
    case vsockUnavailable

    /// The daemon did not become ready within the specified timeout.
    case daemonTimeout(seconds: TimeInterval, containerID: String)

    /// The `mlx.container.gpu` label exists but could not be decoded.
    case invalidGPULabel(String)

    public var errorDescription: String? {
        switch self {
        case .vsockUnavailable:
            return "vsock is not available on this host. GPU support requires macOS 13+ with Apple Virtualization framework."
        case let .daemonTimeout(seconds, containerID):
            return "MLX Container Daemon did not become ready within \(Int(seconds))s while starting container \(containerID)."
        case let .invalidGPULabel(detail):
            return "Failed to decode mlx.container.gpu container label: \(detail)"
        }
    }
}
