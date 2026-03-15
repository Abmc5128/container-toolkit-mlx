import ArgumentParser
import Foundation

struct ServiceCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "service",
        abstract: "Manage the MLX container daemon as a macOS launchd service",
        subcommands: [
            InstallService.self,
            StartService.self,
            StopService.self,
            ServiceStatus.self,
            ServiceLogs.self,
        ],
        defaultSubcommand: ServiceStatus.self
    )

    // MARK: - Shared helpers

    static let plistLabel = "com.aiflowlabs.mlx-container-daemon"
    static var plistPath: String {
        let home = NSString(string: "~").expandingTildeInPath
        return "\(home)/Library/LaunchAgents/\(plistLabel).plist"
    }
    static var logPath: String {
        let home = NSString(string: "~").expandingTildeInPath
        return "\(home)/.mlx-container/logs/daemon.log"
    }

    // MARK: - install

    struct InstallService: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "install",
            abstract: "Generate (or overwrite) the launchd plist for the daemon"
        )

        @Flag(name: .long, help: "Configure for vsock mode (production, inside Apple VM)")
        var vsock: Bool = false

        @Option(name: .long, help: "TCP port (default: 50051)")
        var tcpPort: Int = 50051

        @Option(name: .long, help: "vsock port (default: 2048)")
        var vsockPort: UInt32 = 2048

        func run() async throws {
            // Delegate to RuntimeCommand.ConfigureRuntime for plist generation
            var configureCmd = RuntimeCommand.ConfigureRuntime()
            configureCmd.vsock = vsock
            configureCmd.tcpPort = tcpPort
            configureCmd.vsockPort = vsockPort
            try await configureCmd.run()
        }
    }

    // MARK: - start

    struct StartService: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "start",
            abstract: "Load and start the daemon via launchctl"
        )

        func run() async throws {
            let plistPath = ServiceCommand.plistPath
            guard FileManager.default.fileExists(atPath: plistPath) else {
                print("ERROR: plist not found at \(plistPath)")
                print("Run 'mlx-ctk service install' first.")
                throw ExitCode.failure
            }

            print("Starting MLX container daemon...")

            // Unload first in case it's already registered (ignore errors)
            _ = try? runLaunchctl(["unload", plistPath])

            let (output, status) = try runLaunchctl(["load", plistPath])
            if status == 0 {
                print("Daemon started.")
                print("Check status: mlx-ctk service status")
                if !output.isEmpty { print(output) }
            } else {
                print("ERROR: launchctl load failed (exit \(status))")
                if !output.isEmpty { print(output) }
                throw ExitCode.failure
            }
        }
    }

    // MARK: - stop

    struct StopService: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "stop",
            abstract: "Stop and unload the daemon via launchctl"
        )

        func run() async throws {
            let plistPath = ServiceCommand.plistPath
            guard FileManager.default.fileExists(atPath: plistPath) else {
                print("ERROR: plist not found at \(plistPath)")
                print("Run 'mlx-ctk service install' first.")
                throw ExitCode.failure
            }

            print("Stopping MLX container daemon...")
            let (output, status) = try runLaunchctl(["unload", plistPath])
            if status == 0 {
                print("Daemon stopped.")
                if !output.isEmpty { print(output) }
            } else {
                print("WARN: launchctl unload returned exit \(status) (daemon may not have been running)")
                if !output.isEmpty { print(output) }
            }
        }
    }

    // MARK: - status

    struct ServiceStatus: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "status",
            abstract: "Show daemon status: launchctl state, PID, and TCP reachability"
        )

        func run() async throws {
            let label = ServiceCommand.plistLabel
            let plistPath = ServiceCommand.plistPath

            print("MLX Container Daemon — Status")
            print(String(repeating: "─", count: 44))

            // 1. Plist on disk
            let plistExists = FileManager.default.fileExists(atPath: plistPath)
            print("Plist installed : \(plistExists ? "yes" : "no")  (\(plistPath))")

            // 2. launchctl list
            let (listOutput, listStatus) = (try? runLaunchctl(["list", label])) ?? ("", -1)
            if listStatus == 0 && !listOutput.isEmpty {
                // Parse PID and last exit status from launchctl output
                // Format: PID  LastExitStatus  Label
                let lines = listOutput.split(separator: "\n")
                for line in lines where line.contains(label) {
                    let parts = line.split(separator: "\t").map(String.init)
                    if parts.count >= 3 {
                        let pid = parts[0].trimmingCharacters(in: .whitespaces)
                        let exitStatus = parts[1].trimmingCharacters(in: .whitespaces)
                        if pid == "-" {
                            print("launchd state  : registered, not running (last exit: \(exitStatus))")
                        } else {
                            print("launchd state  : running (PID \(pid))")
                        }
                    }
                }
            } else {
                print("launchd state  : not registered")
            }

            // 3. TCP connectivity check on port 50051
            let tcpReachable = await checkTCPPort(host: "127.0.0.1", port: 50051)
            print("TCP :50051     : \(tcpReachable ? "reachable (daemon responding)" : "not reachable")")

            print(String(repeating: "─", count: 44))
            if !plistExists {
                print("Run 'mlx-ctk service install' to install, then 'mlx-ctk service start'.")
            } else if !tcpReachable {
                print("Run 'mlx-ctk service start' to start the daemon.")
            } else {
                print("Daemon is running. Use 'mlx-ctk device status' for GPU/model details.")
            }
        }

        private func checkTCPPort(host: String, port: Int) async -> Bool {
            await withCheckedContinuation { continuation in
                let task = Task {
                    do {
                        let (didConnect, _) = try await connectTCP(host: host, port: port, timeoutSeconds: 2)
                        continuation.resume(returning: didConnect)
                    } catch {
                        continuation.resume(returning: false)
                    }
                }
                _ = task
            }
        }

        private func connectTCP(host: String, port: Int, timeoutSeconds: Double) async throws -> (Bool, String) {
            // Use URLSession to send a minimal HTTP/1.1 request — if we get any response the port is open
            let url = URL(string: "http://\(host):\(port)/")!
            var request = URLRequest(url: url, timeoutInterval: timeoutSeconds)
            request.httpMethod = "GET"
            let session = URLSession(configuration: .ephemeral)
            do {
                let (_, response) = try await session.data(for: request)
                _ = response
                return (true, "")
            } catch let error as URLError {
                // Connection refused = port closed; other errors (bad server response) = port open
                if error.code == .cannotConnectToHost || error.code == .networkConnectionLost {
                    return (false, error.localizedDescription)
                }
                // The gRPC server will reject plain HTTP with a protocol error — that still means it's up
                return (true, "")
            }
        }
    }

    // MARK: - logs

    struct ServiceLogs: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "logs",
            abstract: "Tail the daemon log file"
        )

        @Option(name: .shortAndLong, help: "Number of lines to show (default: 50)")
        var lines: Int = 50

        @Flag(name: .shortAndLong, help: "Follow the log (like tail -f)")
        var follow: Bool = false

        func run() async throws {
            let logPath = ServiceCommand.logPath
            guard FileManager.default.fileExists(atPath: logPath) else {
                print("No log file found at \(logPath)")
                print("Start the daemon first: mlx-ctk service start")
                throw ExitCode.failure
            }

            var args = ["-n", "\(lines)", logPath]
            if follow { args.insert("-f", at: 0) }

            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/tail")
            process.arguments = args
            process.standardOutput = FileHandle.standardOutput
            process.standardError = FileHandle.standardError
            try process.run()
            process.waitUntilExit()

            if process.terminationStatus != 0 {
                throw ExitCode(process.terminationStatus)
            }
        }
    }
}

// MARK: - launchctl helper (file-private)

private func runLaunchctl(_ args: [String]) throws -> (output: String, exitCode: Int32) {
    let process = Process()
    let pipe = Pipe()
    let errPipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
    process.arguments = args
    process.standardOutput = pipe
    process.standardError = errPipe
    try process.run()
    process.waitUntilExit()

    let outData = pipe.fileHandleForReading.readDataToEndOfFile()
    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
    let combined = [
        String(data: outData, encoding: .utf8) ?? "",
        String(data: errData, encoding: .utf8) ?? "",
    ]
    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    .filter { !$0.isEmpty }
    .joined(separator: "\n")

    return (combined, process.terminationStatus)
}
