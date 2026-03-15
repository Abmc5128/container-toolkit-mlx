import Testing
import Foundation
import Logging
@testable import MLXContainerDaemon

// MARK: - Helpers

private func makeAllocator(
    totalMemory: UInt64 = 16 * 1024 * 1024 * 1024,  // 16 GB
    maxBudget: UInt64 = 8 * 1024 * 1024 * 1024       // 8 GB budget
) -> GPUMemoryAllocator {
    let logger = Logger(label: "test.gpu-memory-allocator")
    return GPUMemoryAllocator(
        totalMemoryBytes: totalMemory,
        maxBudgetBytes: maxBudget,
        logger: logger
    )
}

// MARK: - Tests

@Suite("GPUMemoryAllocator Tests")
struct GPUMemoryAllocatorTests {

    // MARK: - Successful allocation

    @Test("Allocation succeeds when requested bytes are within budget")
    func allocationSucceedsWithinBudget() async throws {
        let allocator = makeAllocator(maxBudget: 8 * 1024 * 1024 * 1024)
        let requestedBytes: UInt64 = 2 * 1024 * 1024 * 1024  // 2 GB

        let granted = try await allocator.allocate(containerID: "ctr-1", requestedBytes: requestedBytes)
        #expect(granted == requestedBytes, "Should grant exactly what was requested when within budget")
    }

    @Test("Allocation succeeds for 1 MB (minimum allocation)")
    func allocationSucceedsMinimum() async throws {
        let allocator = makeAllocator()
        let oneMB: UInt64 = 1024 * 1024

        let granted = try await allocator.allocate(containerID: "ctr-min", requestedBytes: oneMB)
        #expect(granted == oneMB)
    }

    @Test("Allocation caps granted bytes to available budget when request exceeds it")
    func allocationCapsToAvailable() async throws {
        let budgetBytes: UInt64 = 2 * 1024 * 1024 * 1024  // 2 GB budget
        let allocator = makeAllocator(maxBudget: budgetBytes)
        let overRequestBytes: UInt64 = 10 * 1024 * 1024 * 1024  // 10 GB — exceeds budget

        let granted = try await allocator.allocate(containerID: "ctr-cap", requestedBytes: overRequestBytes)
        // Budget is 2 GB, so granted should be capped at 2 GB (the available amount)
        #expect(granted == budgetBytes, "Granted bytes should be capped to the total budget")
    }

    // MARK: - Allocation failure when budget exhausted

    @Test("Allocation throws when budget is fully exhausted")
    func allocationFailsWhenBudgetExhausted() async throws {
        let budgetBytes: UInt64 = 2 * 1024 * 1024 * 1024  // 2 GB
        let allocator = makeAllocator(maxBudget: budgetBytes)

        // Exhaust the budget
        _ = try await allocator.allocate(containerID: "ctr-fill", requestedBytes: budgetBytes)

        // Second allocation should fail — no memory remains
        await #expect(throws: (any Error).self) {
            _ = try await allocator.allocate(containerID: "ctr-fail", requestedBytes: 1024 * 1024)
        }
    }

    @Test("Allocation fails with GPUMemoryError.insufficientMemory when over budget")
    func allocationThrowsCorrectErrorType() async throws {
        let budgetBytes: UInt64 = 1 * 1024 * 1024 * 1024  // 1 GB
        let allocator = makeAllocator(maxBudget: budgetBytes)
        _ = try await allocator.allocate(containerID: "ctr-full", requestedBytes: budgetBytes)

        do {
            _ = try await allocator.allocate(containerID: "ctr-extra", requestedBytes: 512 * 1024 * 1024)
            Issue.record("Expected an error to be thrown")
        } catch let error as GPUMemoryError {
            if case .insufficientMemory = error {
                // Correct error type — test passes
            } else {
                Issue.record("Unexpected GPUMemoryError case: \(error)")
            }
        }
    }

    // MARK: - Release

    @Test("Release frees memory so subsequent allocation succeeds")
    func releaseFreesMemory() async throws {
        let budgetBytes: UInt64 = 2 * 1024 * 1024 * 1024  // 2 GB
        let allocator = makeAllocator(maxBudget: budgetBytes)

        _ = try await allocator.allocate(containerID: "ctr-a", requestedBytes: budgetBytes)

        // Should fail before release
        await #expect(throws: (any Error).self) {
            _ = try await allocator.allocate(containerID: "ctr-b", requestedBytes: 1024 * 1024)
        }

        // Release and retry
        await allocator.release(containerID: "ctr-a")

        let granted = try await allocator.allocate(containerID: "ctr-b", requestedBytes: 1024 * 1024)
        #expect(granted == 1024 * 1024, "Memory should be available after releasing the previous allocation")
    }

    @Test("Release of unknown container ID is a no-op (does not throw)")
    func releaseUnknownContainerIsNoOp() async {
        let allocator = makeAllocator()
        // Should not throw
        await allocator.release(containerID: "unknown-container")
    }

    // MARK: - Snapshot

    @Test("Snapshot reflects zero allocation on fresh allocator")
    func snapshotInitialState() async throws {
        let totalBytes: UInt64 = 16 * 1024 * 1024 * 1024
        let budgetBytes: UInt64 = 8 * 1024 * 1024 * 1024
        let allocator = makeAllocator(totalMemory: totalBytes, maxBudget: budgetBytes)

        let snap = await allocator.snapshot()
        #expect(snap.totalBytes == totalBytes)
        #expect(snap.budgetBytes == budgetBytes)
        #expect(snap.allocatedBytes == 0)
        #expect(snap.availableBytes == budgetBytes)
        #expect(snap.containerAllocations.isEmpty)
    }

    @Test("Snapshot reflects allocated bytes after allocation")
    func snapshotAfterAllocation() async throws {
        let budgetBytes: UInt64 = 4 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)
        let requestBytes: UInt64 = 1 * 1024 * 1024 * 1024

        _ = try await allocator.allocate(containerID: "ctr-snap", requestedBytes: requestBytes)

        let snap = await allocator.snapshot()
        #expect(snap.allocatedBytes == requestBytes)
        #expect(snap.availableBytes == budgetBytes - requestBytes)
        #expect(snap.containerAllocations["ctr-snap"] == requestBytes)
    }

    @Test("Snapshot availableBytes returns 0 when fully allocated")
    func snapshotFullyAllocated() async throws {
        let budgetBytes: UInt64 = 2 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)

        _ = try await allocator.allocate(containerID: "ctr-full", requestedBytes: budgetBytes)

        let snap = await allocator.snapshot()
        #expect(snap.availableBytes == 0)
        #expect(snap.allocatedBytes == budgetBytes)
    }

    @Test("Snapshot shows zero allocation after release")
    func snapshotAfterRelease() async throws {
        let budgetBytes: UInt64 = 4 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)
        let requestBytes: UInt64 = 1 * 1024 * 1024 * 1024

        _ = try await allocator.allocate(containerID: "ctr-release", requestedBytes: requestBytes)
        await allocator.release(containerID: "ctr-release")

        let snap = await allocator.snapshot()
        #expect(snap.allocatedBytes == 0)
        #expect(snap.availableBytes == budgetBytes)
        #expect(snap.containerAllocations.isEmpty)
    }

    // MARK: - Re-allocation replaces, not accumulates

    @Test("Re-allocating the same container ID replaces the previous allocation")
    func reallocationReplacesPrevious() async throws {
        let budgetBytes: UInt64 = 4 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)

        let first: UInt64 = 1 * 1024 * 1024 * 1024
        let second: UInt64 = 500 * 1024 * 1024

        _ = try await allocator.allocate(containerID: "ctr-same", requestedBytes: first)
        _ = try await allocator.allocate(containerID: "ctr-same", requestedBytes: second)

        let snap = await allocator.snapshot()
        // Should reflect the second allocation only, not accumulated
        #expect(snap.containerAllocations["ctr-same"] == second,
                "Re-allocation should replace the previous entry, not accumulate")
        #expect(snap.allocatedBytes == second)
    }

    // MARK: - Multiple containers tracked independently

    @Test("Multiple containers are tracked with independent allocations")
    func multipleContainersTrackedIndependently() async throws {
        let budgetBytes: UInt64 = 8 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)

        let a: UInt64 = 1 * 1024 * 1024 * 1024
        let b: UInt64 = 2 * 1024 * 1024 * 1024
        let c: UInt64 = 500 * 1024 * 1024

        _ = try await allocator.allocate(containerID: "ctr-a", requestedBytes: a)
        _ = try await allocator.allocate(containerID: "ctr-b", requestedBytes: b)
        _ = try await allocator.allocate(containerID: "ctr-c", requestedBytes: c)

        let snap = await allocator.snapshot()
        #expect(snap.containerAllocations.count == 3)
        #expect(snap.containerAllocations["ctr-a"] == a)
        #expect(snap.containerAllocations["ctr-b"] == b)
        #expect(snap.containerAllocations["ctr-c"] == c)
        #expect(snap.allocatedBytes == a + b + c)
    }

    @Test("Releasing one container does not affect others")
    func releasingOneContainerDoesNotAffectOthers() async throws {
        let budgetBytes: UInt64 = 8 * 1024 * 1024 * 1024
        let allocator = makeAllocator(maxBudget: budgetBytes)

        let a: UInt64 = 1 * 1024 * 1024 * 1024
        let b: UInt64 = 2 * 1024 * 1024 * 1024

        _ = try await allocator.allocate(containerID: "ctr-a", requestedBytes: a)
        _ = try await allocator.allocate(containerID: "ctr-b", requestedBytes: b)

        await allocator.release(containerID: "ctr-a")

        let snap = await allocator.snapshot()
        #expect(snap.containerAllocations["ctr-a"] == nil)
        #expect(snap.containerAllocations["ctr-b"] == b)
        #expect(snap.allocatedBytes == b)
    }

    // MARK: - maxBudget = 0 uses total memory

    @Test("maxBudgetBytes = 0 treats totalMemoryBytes as the effective budget")
    func zeroBudgetUsesTotalMemory() async throws {
        let totalBytes: UInt64 = 4 * 1024 * 1024 * 1024
        let logger = Logger(label: "test.zero-budget")
        let allocator = GPUMemoryAllocator(
            totalMemoryBytes: totalBytes,
            maxBudgetBytes: 0,  // 0 = unlimited → should use totalMemory
            logger: logger
        )
        let snap = await allocator.snapshot()
        #expect(snap.budgetBytes == totalBytes, "When maxBudgetBytes is 0, budget should equal totalMemoryBytes")
    }
}
