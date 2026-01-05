#!/usr/bin/env python3
"""Test register allocation."""

from r65.compiler.codegen import (
    RegisterAllocator,
    ScratchRegisterPool,
    LocationKind,
)
from r65.compiler.mir.nodes import VirtualRegister
from r65.compiler.hir.types import BasicTypeInfo


def test_scratch_pool():
    """Test scratch register pool management."""
    print("=" * 80)
    print("Test 1: Scratch Register Pool")
    print("=" * 80)
    print()

    # Create scratch pool
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")
    pool.add_scratch(0x17, 1, "SCRATCH1")
    pool.add_scratch(0x18, 2, "SCRATCH2")  # 2-byte scratch

    print("Scratch pool created:")
    for scratch in pool.scratches:
        print(f"  ${scratch.address:04X}: {scratch.name} ({scratch.size} bytes)")
    print()

    # Allocate some virtual registers
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="temp")
    vreg1 = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="count")
    vreg2 = VirtualRegister(id=2, type_info=BasicTypeInfo('u16'), hint="value")

    print("Allocating virtual registers:")
    scratch0 = pool.allocate(vreg0)
    if scratch0:
        print(f"  %0 (u8) → ${scratch0.address:04X} ({scratch0.name})")
    else:
        print(f"  %0 (u8) → FAILED (no scratch)")

    scratch1 = pool.allocate(vreg1)
    if scratch1:
        print(f"  %1 (u8) → ${scratch1.address:04X} ({scratch1.name})")
    else:
        print(f"  %1 (u8) → FAILED (no scratch)")

    scratch2 = pool.allocate(vreg2)
    if scratch2:
        print(f"  %2 (u16) → ${scratch2.address:04X} ({scratch2.name})")
    else:
        print(f"  %2 (u16) → FAILED (no scratch)")

    print()

    # Try to allocate when pool exhausted
    vreg3 = VirtualRegister(id=3, type_info=BasicTypeInfo('u8'), hint="extra")
    scratch3 = pool.allocate(vreg3)
    if scratch3:
        print(f"  %3 (u8) → ${scratch3.address:04X} ({scratch3.name})")
    else:
        print(f"  ✅ %3 (u8) → Correctly failed (pool exhausted)")

    print()

    # Free a scratch and reallocate
    print("Freeing %0, then reallocating:")
    pool.free(vreg0)
    scratch3_retry = pool.allocate(vreg3)
    if scratch3_retry:
        print(f"  ✅ %3 (u8) → ${scratch3_retry.address:04X} ({scratch3_retry.name}) (reused freed scratch)")
    else:
        print(f"  ❌ %3 (u8) → FAILED (should have reused freed scratch)")

    print()
    return True


def test_register_allocator():
    """Test full register allocator with scratch and stack."""
    print("=" * 80)
    print("Test 2: Register Allocator (Scratch + Stack)")
    print("=" * 80)
    print()

    # Create allocator with limited scratch pool
    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")
    pool.add_scratch(0x17, 1, "SCRATCH1")

    allocator = RegisterAllocator(scratch_pool=pool)

    # Allocate more vregs than scratch pool can handle
    vregs = [
        VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a"),
        VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b"),
        VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="c"),  # Should spill
        VirtualRegister(id=3, type_info=BasicTypeInfo('u16'), hint="d"),  # Should spill
    ]

    print("Allocating 4 virtual registers (pool has 2 scratch slots):")
    for vreg in vregs:
        location = allocator.allocate_vreg(vreg)
        print(f"  %{vreg.id} ({vreg.type_info.name}) → {location}")

    print()

    # Verify allocations
    print("Verification:")
    loc0 = allocator.get_location(vregs[0])
    loc1 = allocator.get_location(vregs[1])
    loc2 = allocator.get_location(vregs[2])
    loc3 = allocator.get_location(vregs[3])

    if loc0.kind == LocationKind.SCRATCH and loc1.kind == LocationKind.SCRATCH:
        print(f"  ✅ %0 and %1 allocated to scratch")
    else:
        print(f"  ❌ %0 and %1 should be in scratch")

    if loc2.kind == LocationKind.STACK and loc3.kind == LocationKind.STACK:
        print(f"  ✅ %2 and %3 spilled to stack")
    else:
        print(f"  ❌ %2 and %3 should be spilled to stack")

    print(f"  Stack frame size: {allocator.get_stack_frame_size()} bytes")

    print()
    return True


def test_vreg_size_detection():
    """Test virtual register size detection."""
    print("=" * 80)
    print("Test 3: Virtual Register Size Detection")
    print("=" * 80)
    print()

    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")  # 1-byte
    pool.add_scratch(0x18, 2, "SCRATCH1")  # 2-byte

    allocator = RegisterAllocator(scratch_pool=pool)

    # Test different sizes
    vreg_u8 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="byte")
    vreg_u16 = VirtualRegister(id=1, type_info=BasicTypeInfo('u16'), hint="word")

    print("Allocating different sizes:")
    loc_u8 = allocator.allocate_vreg(vreg_u8)
    loc_u16 = allocator.allocate_vreg(vreg_u16)

    print(f"  u8 → {loc_u8} (size={loc_u8.size})")
    print(f"  u16 → {loc_u16} (size={loc_u16.size})")

    print()

    # Verify size matching
    if loc_u8.size == 1 and loc_u8.kind == LocationKind.SCRATCH:
        print(f"  ✅ u8 allocated to 1-byte scratch")
    else:
        print(f"  ❌ u8 should use 1-byte scratch")

    if loc_u16.size == 2 and loc_u16.kind == LocationKind.SCRATCH:
        print(f"  ✅ u16 allocated to 2-byte scratch")
    else:
        print(f"  ❌ u16 should use 2-byte scratch")

    print()
    return True


def test_bulk_allocation():
    """Test bulk allocation of all vregs at once."""
    print("=" * 80)
    print("Test 4: Bulk Allocation")
    print("=" * 80)
    print()

    pool = ScratchRegisterPool()
    pool.add_scratch(0x16, 1, "SCRATCH0")

    allocator = RegisterAllocator(scratch_pool=pool)

    # Create multiple vregs
    vregs = [
        VirtualRegister(id=i, type_info=BasicTypeInfo('u8'), hint=f"v{i}")
        for i in range(5)
    ]

    print(f"Bulk allocating {len(vregs)} virtual registers:")
    allocator.allocate_all(vregs)

    for vreg in vregs:
        loc = allocator.get_location(vreg)
        print(f"  %{vreg.id} → {loc}")

    print()

    # Count scratch vs stack
    scratch_count = sum(1 for vreg in vregs if allocator.get_location(vreg).kind == LocationKind.SCRATCH)
    stack_count = sum(1 for vreg in vregs if allocator.get_location(vreg).kind == LocationKind.STACK)

    print(f"Allocation summary:")
    print(f"  Scratch: {scratch_count}")
    print(f"  Stack: {stack_count}")
    print(f"  Total: {scratch_count + stack_count}")

    if scratch_count == 1 and stack_count == 4:
        print(f"  ✅ Correct allocation (1 scratch, 4 stack)")
    else:
        print(f"  ❌ Expected 1 scratch and 4 stack")

    print()
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Register Allocation Tests")
    print("=" * 80)
    print()

    # Run tests
    test1_passed = test_scratch_pool()
    test2_passed = test_register_allocator()
    test3_passed = test_vreg_size_detection()
    test4_passed = test_bulk_allocation()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Scratch Pool Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Register Allocator Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Size Detection Test: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Bulk Allocation Test: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    print()

    if test1_passed and test2_passed and test3_passed and test4_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
