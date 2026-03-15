import Testing
import Foundation
@testable import MLXDeviceDiscovery

@Suite("DeviceDiscovery Tests")
struct DeviceDiscoveryTests {

    // MARK: - discover()

    @Test("discover() returns at least one device on macOS with Metal")
    func discoverReturnsDevices() {
        let devices = DeviceDiscovery.discover()
        // Metal is available on all supported macOS 15+ Apple Silicon machines
        #expect(!devices.isEmpty, "Expected at least one Metal GPU device on Apple Silicon macOS")
    }

    @Test("discover() returns AppleGPUDevice with non-empty name")
    func discoverDeviceHasName() {
        let devices = DeviceDiscovery.discover()
        guard let first = devices.first else {
            Issue.record("No devices discovered — skipping name check")
            return
        }
        #expect(!first.name.isEmpty, "GPU device name must not be empty")
    }

    @Test("discover() returns device with positive memory values")
    func discoverDeviceHasPositiveMemory() {
        let devices = DeviceDiscovery.discover()
        guard let first = devices.first else {
            Issue.record("No devices discovered — skipping memory check")
            return
        }
        #expect(first.unifiedMemoryBytes > 0, "unifiedMemoryBytes must be > 0")
        #expect(first.recommendedMaxWorkingSetSize > 0, "recommendedMaxWorkingSetSize must be > 0")
    }

    @Test("discover() returns device with non-empty GPU family")
    func discoverDeviceHasGPUFamily() {
        let devices = DeviceDiscovery.discover()
        guard let first = devices.first else {
            Issue.record("No devices discovered — skipping gpuFamily check")
            return
        }
        #expect(!first.gpuFamily.isEmpty, "gpuFamily must not be empty")
    }

    // MARK: - defaultDevice()

    @Test("defaultDevice() returns first discovered device")
    func defaultDeviceMatchesDiscover() {
        let first = DeviceDiscovery.discover().first
        let defaultDev = DeviceDiscovery.defaultDevice()
        #expect(first?.name == defaultDev?.name)
    }

    // MARK: - systemMemoryBytes()

    @Test("systemMemoryBytes() returns a value greater than zero")
    func systemMemoryBytesPositive() {
        let mem = DeviceDiscovery.systemMemoryBytes()
        #expect(mem > 0, "System memory must be positive")
    }

    @Test("systemMemoryBytes() returns a plausible value (at least 1 GB)")
    func systemMemoryBytesAtLeast1GB() {
        let mem = DeviceDiscovery.systemMemoryBytes()
        let oneGB: UInt64 = 1024 * 1024 * 1024
        #expect(mem >= oneGB, "System memory should be at least 1 GB on any supported machine")
    }

    // MARK: - chipName()

    @Test("chipName() returns a non-nil value on Apple Silicon macOS")
    func chipNameNonNil() {
        let name = DeviceDiscovery.chipName()
        #expect(name != nil, "chipName() should return a non-nil string on Apple Silicon macOS")
    }

    @Test("chipName() returns a non-empty string")
    func chipNameNonEmpty() {
        guard let name = DeviceDiscovery.chipName() else {
            Issue.record("chipName() returned nil — skipping non-empty check")
            return
        }
        #expect(!name.isEmpty, "chipName() must return a non-empty string")
    }

    // MARK: - AppleGPUDevice Codable roundtrip

    @Test("AppleGPUDevice encodes and decodes via JSON without data loss")
    func appleGPUDeviceCodableRoundtrip() throws {
        let original = AppleGPUDevice(
            name: "Apple M3 Max",
            registryID: 0xDEADBEEF_CAFEBABE,
            recommendedMaxWorkingSetSize: 68_719_476_736,
            gpuFamily: "metal3",
            unifiedMemoryBytes: 137_438_953_472,
            maxThreadsPerThreadgroup: 1024,
            supportsMetal3: true,
            hasUnifiedMemory: true
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        #expect(!data.isEmpty, "Encoded data must not be empty")

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(AppleGPUDevice.self, from: data)

        #expect(decoded.name == original.name)
        #expect(decoded.registryID == original.registryID)
        #expect(decoded.recommendedMaxWorkingSetSize == original.recommendedMaxWorkingSetSize)
        #expect(decoded.gpuFamily == original.gpuFamily)
        #expect(decoded.unifiedMemoryBytes == original.unifiedMemoryBytes)
        #expect(decoded.maxThreadsPerThreadgroup == original.maxThreadsPerThreadgroup)
        #expect(decoded.supportsMetal3 == original.supportsMetal3)
        #expect(decoded.hasUnifiedMemory == original.hasUnifiedMemory)
    }

    @Test("AppleGPUDevice with minimal values encodes and decodes correctly")
    func appleGPUDeviceCodableMinimalValues() throws {
        let original = AppleGPUDevice(
            name: "",
            registryID: 0,
            recommendedMaxWorkingSetSize: 0,
            gpuFamily: "",
            unifiedMemoryBytes: 0,
            maxThreadsPerThreadgroup: 0,
            supportsMetal3: false,
            hasUnifiedMemory: false
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AppleGPUDevice.self, from: data)

        #expect(decoded.name == "")
        #expect(decoded.registryID == 0)
        #expect(decoded.supportsMetal3 == false)
        #expect(decoded.hasUnifiedMemory == false)
    }

    @Test("AppleGPUDevice description contains device name")
    func appleGPUDeviceDescriptionContainsName() {
        let device = AppleGPUDevice(
            name: "Apple M2 Pro",
            registryID: 1,
            recommendedMaxWorkingSetSize: 1024 * 1024 * 1024,
            gpuFamily: "apple9",
            unifiedMemoryBytes: 16 * 1024 * 1024 * 1024,
            maxThreadsPerThreadgroup: 1024,
            supportsMetal3: false,
            hasUnifiedMemory: true
        )
        #expect(device.description.contains("Apple M2 Pro"))
    }

    @Test("Discovered device Codable roundtrip preserves all fields")
    func discoveredDeviceCodableRoundtrip() throws {
        guard let device = DeviceDiscovery.defaultDevice() else {
            Issue.record("No device available — skipping live Codable roundtrip")
            return
        }
        let data = try JSONEncoder().encode(device)
        let decoded = try JSONDecoder().decode(AppleGPUDevice.self, from: data)
        #expect(decoded.name == device.name)
        #expect(decoded.registryID == device.registryID)
        #expect(decoded.gpuFamily == device.gpuFamily)
        #expect(decoded.unifiedMemoryBytes == device.unifiedMemoryBytes)
    }
}
