import ArgumentParser
import Foundation

struct RuntimeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "runtime",
        abstract: "Manage the MLX container runtime and daemon configuration",
        subcommands: [ConfigureRuntime.self],
        defaultSubcommand: ConfigureRuntime.self
    )

    struct ConfigureRuntime: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "configure",
            abstract: "Generate a launchd plist for the MLX container daemon"
        )

        @Flag(name: .long, help: "Configure for vsock mode (production, inside Apple VM)")
        var vsock: Bool = false

        @Option(name: .long, help: "TCP port to use (default: 50051, only for TCP mode)")
        var tcpPort: Int = 50051

        @Option(name: .long, help: "vsock port to use (default: 2048, only for vsock mode)")
        var vsockPort: UInt32 = 2048

        func run() async throws {
            let fm = FileManager.default
            let home = NSString(string: "~").expandingTildeInPath

            // Paths
            let plistDir = "\(home)/Library/LaunchAgents"
            let plistPath = "\(plistDir)/com.aiflowlabs.mlx-container-daemon.plist"
            let logsDir = "\(home)/.mlx-container/logs"
            let workingDir = "\(home)/.mlx-container"

            // Ensure directories exist
            try fm.createDirectory(atPath: plistDir, withIntermediateDirectories: true)
            try fm.createDirectory(atPath: logsDir, withIntermediateDirectories: true)
            try fm.createDirectory(atPath: workingDir, withIntermediateDirectories: true)

            // Locate the daemon binary
            let daemonPath = resolveDaemonPath()

            // Build program arguments
            let programArguments: [String]
            if vsock {
                programArguments = [daemonPath, "--port", "\(vsockPort)"]
            } else {
                programArguments = [daemonPath, "--tcp", "--tcp-port", "\(tcpPort)"]
            }

            let modeLabel = vsock ? "vsock port \(vsockPort)" : "TCP port \(tcpPort)"

            // Build plist content
            let plistContent = buildPlist(
                label: "com.aiflowlabs.mlx-container-daemon",
                programArguments: programArguments,
                workingDir: workingDir,
                stdoutLog: "\(logsDir)/daemon.log",
                stderrLog: "\(logsDir)/daemon-error.log"
            )

            // Write plist
            try plistContent.write(toFile: plistPath, atomically: true, encoding: .utf8)

            print("MLX Container Daemon — launchd configuration")
            print(String(repeating: "=", count: 50))
            print("Plist written:  \(plistPath)")
            print("Mode:           \(vsock ? "vsock (production)" : "TCP (development)")")
            print("Listening on:   \(modeLabel)")
            print("Working dir:    \(workingDir)")
            print("Logs:           \(logsDir)/")
            print(String(repeating: "=", count: 50))
            print("")
            print("Next steps:")
            print("  Load agent:   launchctl load \(plistPath)")
            print("  Start daemon: mlx-ctk service start")
            print("  Stop daemon:  mlx-ctk service stop")
            print("  Check status: mlx-ctk service status")
            print("  View logs:    mlx-ctk service logs")
            print("")
            print("NOTE: RunAtLoad is false — start manually with 'mlx-ctk service start'.")
            if vsock {
                print("NOTE: vsock mode requires the daemon to run on the Apple VM host.")
            }
        }

        private func resolveDaemonPath() -> String {
            // Try to find the daemon binary relative to the CLI binary
            if let execPath = Bundle.main.executableURL?.deletingLastPathComponent() {
                let candidate = execPath.appendingPathComponent("mlx-container-daemon").path
                if FileManager.default.isExecutableFile(atPath: candidate) {
                    return candidate
                }
            }
            // Fallback to a well-known install location
            let installPath = "/usr/local/bin/mlx-container-daemon"
            if FileManager.default.isExecutableFile(atPath: installPath) {
                return installPath
            }
            // Last resort: assume it's on PATH
            return "mlx-container-daemon"
        }

        private func buildPlist(
            label: String,
            programArguments: [String],
            workingDir: String,
            stdoutLog: String,
            stderrLog: String
        ) -> String {
            let argsXML = programArguments
                .map { "        <string>\($0)</string>" }
                .joined(separator: "\n")

            return """
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
              "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0">
            <dict>
                <key>Label</key>
                <string>\(label)</string>

                <key>ProgramArguments</key>
                <array>
            \(argsXML)
                </array>

                <key>KeepAlive</key>
                <true/>

                <key>RunAtLoad</key>
                <false/>

                <key>WorkingDirectory</key>
                <string>\(workingDir)</string>

                <key>StandardOutPath</key>
                <string>\(stdoutLog)</string>

                <key>StandardErrorPath</key>
                <string>\(stderrLog)</string>

                <key>EnvironmentVariables</key>
                <dict>
                    <key>HOME</key>
                    <string>\(NSString(string: "~").expandingTildeInPath)</string>
                </dict>
            </dict>
            </plist>
            """
        }
    }
}
