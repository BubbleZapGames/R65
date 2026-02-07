#!/usr/bin/env python3
"""Test memory allocation and symbol definition generation."""

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder
from r65.compiler.codegen import ProgramCodeGenerator


def test_memory_allocation():
    """Test memory allocation with various storage types."""
    source = """
// Constants
const TILE_SIZE: u8 = 8;
const BUFFER_SIZE: u16 = 256;

// Zero-page variables (explicit addresses)
#[zeropage(0x20)]
static mut TEMP: u8 = 0;

#[zeropage(0x22)]
static mut COUNTER: u16 = 0;

// Zero-page variables (auto-allocated)
#[zeropage]
static mut FLAGS: u8 = 0;

#[zeropage]
static mut PLAYER_X: u8 = 10;

// RAM variables (explicit address)
#[ram(0x7E2000)]
static mut BUFFER: [u8; 256];

// RAM variables (auto-allocated)
#[ram]
static mut SCORE: u16 = 0;

#[ram]
static mut LIVES: u8 = 3;

// Hardware registers
#[hw(0x2100)]
static mut INIDISP: u8;

#[hw(0x4212)]
static mut HVBJOY: u8;

// Entry point
#[entry]
fn main() -> ! {
    TEMP = 42;
    loop {}
}
"""

    print("=" * 80)
    print("Testing Memory Allocation and Symbol Definitions")
    print("=" * 80)
    print()

    # Parse
    ast_program = parse(source, "<test>")

    # Build HIR
    hir_builder = HIRBuilder()
    hir_program = hir_builder.build_program(ast_program)

    # Type check
    type_checker = TypeChecker(hir_program)
    type_checker.check()

    # Build MIR
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)

    # Generate assembly
    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program)

    print("Generated Assembly:")
    print("=" * 80)
    print(assembly)
    print("=" * 80)
    print()

    # Verify allocations
    print("Verification:")
    print("=" * 80)
    allocator = codegen.allocator

    # Check explicit zero-page allocations
    all_allocs = allocator.get_all_allocations()
    zp_allocs = allocator.get_allocations_by_type('zeropage')
    ram_allocs = allocator.get_allocations_by_type('ram')
    hw_allocs = allocator.get_allocations_by_type('hw')

    print(f"Total allocations: {len(all_allocs)}")
    print(f"  Zero-page: {len(zp_allocs)}")
    print(f"  RAM: {len(ram_allocs)}")
    print(f"  Hardware: {len(hw_allocs)}")
    print()

    print("Zero-Page Allocations:")
    for alloc in sorted(zp_allocs, key=lambda a: a.address):
        explicit = "explicit" if alloc.is_explicit else "auto"
        print(f"  ${alloc.address:04X}: {alloc.symbol.name} ({alloc.size} bytes, {explicit})")

    print()
    print("RAM Allocations:")
    for alloc in sorted(ram_allocs, key=lambda a: a.address):
        explicit = "explicit" if alloc.is_explicit else "auto"
        print(f"  ${alloc.address:06X}: {alloc.symbol.name} ({alloc.size} bytes, {explicit})")

    print()
    print("Hardware Register Allocations:")
    for alloc in sorted(hw_allocs, key=lambda a: a.address):
        print(f"  ${alloc.address:04X}: {alloc.symbol.name}")

    print()

    # Verify specific allocations
    checks = [
        ("TEMP", 0x20, 1, 'zeropage', True),
        ("COUNTER", 0x22, 2, 'zeropage', True),
        ("FLAGS", None, 1, 'zeropage', False),  # Auto-allocated
        ("PLAYER_X", None, 1, 'zeropage', False),  # Auto-allocated
        ("BUFFER", 0x7E2000, 256, 'ram', True),
        ("SCORE", None, 2, 'ram', False),  # Auto-allocated
        ("LIVES", None, 1, 'ram', False),  # Auto-allocated
        ("INIDISP", 0x2100, 1, 'hw', True),
        ("HVBJOY", 0x4212, 1, 'hw', True),
    ]

    print("Allocation Verification:")
    all_passed = True
    for symbol_name, expected_addr, expected_size, expected_type, is_explicit in checks:
        # Find allocation
        alloc = None
        for a in all_allocs:
            if a.symbol.name == symbol_name:
                alloc = a
                break

        if alloc is None:
            print(f"  ❌ {symbol_name}: NOT FOUND")
            all_passed = False
            continue

        # Check size
        if alloc.size != expected_size:
            print(f"  ❌ {symbol_name}: Size mismatch (expected {expected_size}, got {alloc.size})")
            all_passed = False
            continue

        # Check type
        if alloc.storage_type != expected_type:
            print(f"  ❌ {symbol_name}: Type mismatch (expected {expected_type}, got {alloc.storage_type})")
            all_passed = False
            continue

        # Check explicit
        if alloc.is_explicit != is_explicit:
            print(f"  ❌ {symbol_name}: Explicit mismatch (expected {is_explicit}, got {alloc.is_explicit})")
            all_passed = False
            continue

        # Check address (if explicit)
        if expected_addr is not None:
            if alloc.address != expected_addr:
                print(f"  ❌ {symbol_name}: Address mismatch (expected ${expected_addr:04X}, got ${alloc.address:04X})")
                all_passed = False
                continue

        print(f"  ✅ {symbol_name}: OK (${alloc.address:04X}, {alloc.size} bytes, {alloc.storage_type}, {'explicit' if alloc.is_explicit else 'auto'})")

    print()
    if all_passed:
        print("✅ All allocation checks passed!")
    else:
        print("❌ Some allocation checks failed!")

    print()

    # Check that assembly contains expected directives
    print("Assembly Content Verification:")
    expected_directives = [
        ".EQU TILE_SIZE 8",
        ".EQU BUFFER_SIZE 256",
        ".DEFINE TEMP $0020",
        ".DEFINE COUNTER $0022",
        ".DEFINE INIDISP $2100",
        ".DEFINE HVBJOY $4212",
        ".DEFINE BUFFER $7E2000",
    ]

    assembly_passed = True
    for directive in expected_directives:
        if directive in assembly:
            print(f"  ✅ Found: {directive}")
        else:
            print(f"  ❌ Missing: {directive}")
            assembly_passed = False

    print()
    if assembly_passed:
        print("✅ All assembly directives found!")
    else:
        print("❌ Some assembly directives missing!")

    assert all_passed, "Some allocation checks failed"
    assert assembly_passed, "Some assembly directives missing"


def test_address_conflict():
    """Test that address conflicts are detected."""
    source = """
// Two variables at same address - should fail
#[zeropage(0x20)]
static mut VAR1: u8 = 0;

#[zeropage(0x20)]
static mut VAR2: u8 = 0;

fn main() -> ! {
    loop {}
}
"""

    print()
    print("=" * 80)
    print("Testing Address Conflict Detection")
    print("=" * 80)
    print()

    try:
        # Parse
        ast_program = parse(source, "<test>")

        # Build HIR
        hir_builder = HIRBuilder()
        hir_program = hir_builder.build_program(ast_program)

        # Type check
        type_checker = TypeChecker(hir_program)
        type_checker.check()

        # Build MIR
        mir_builder = MIRBuilder()
        mir_program = mir_builder.build_program(hir_program)

        # Generate assembly - should fail during memory allocation
        codegen = ProgramCodeGenerator()
        assembly = codegen.generate(mir_program)

        assert False, "Expected exception for address conflict!"

    except Exception as e:
        assert "already allocated" in str(e) or "conflict" in str(e).lower(), \
            f"Unexpected exception: {e}"
        print(f"Correctly detected address conflict: {e}")


def test_auto_allocation_order():
    """Test that auto-allocation happens after explicit allocation."""
    source = """
// Explicit allocation in middle of auto range
#[zeropage(0x30)]
static mut EXPLICIT: u16 = 0;

// Auto allocations should work around explicit
#[zeropage]
static mut AUTO1: u8 = 0;

#[zeropage]
static mut AUTO2: u8 = 0;

fn main() -> ! {
    loop {}
}
"""

    print()
    print("=" * 80)
    print("Testing Auto-Allocation Order")
    print("=" * 80)
    print()

    # Parse
    ast_program = parse(source, "<test>")

    # Build HIR
    hir_builder = HIRBuilder()
    hir_program = hir_builder.build_program(ast_program)

    # Type check
    type_checker = TypeChecker(hir_program)
    type_checker.check()

    # Build MIR
    mir_builder = MIRBuilder()
    mir_program = mir_builder.build_program(hir_program)

    # Generate assembly
    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program)

    # Check allocations
    allocator = codegen.allocator
    zp_allocs = allocator.get_allocations_by_type('zeropage')

    print("Zero-Page Allocations:")
    for alloc in sorted(zp_allocs, key=lambda a: a.address):
        explicit = "explicit" if alloc.is_explicit else "auto"
        print(f"  ${alloc.address:04X}: {alloc.symbol.name} ({explicit})")

    # Verify explicit is at correct address
    explicit_alloc = None
    for alloc in zp_allocs:
        if alloc.symbol.name == "EXPLICIT":
            explicit_alloc = alloc
            break

    assert explicit_alloc is not None and explicit_alloc.address == 0x30, \
        "EXPLICIT not at correct address"

    # Verify auto allocations don't overlap with explicit
    for alloc in zp_allocs:
        if alloc.symbol.name in ("AUTO1", "AUTO2"):
            # Check that auto allocation doesn't overlap with explicit (0x30-0x31)
            assert alloc.address != 0x30 and alloc.address != 0x31, \
                f"{alloc.symbol.name} overlaps with explicit allocation!"

    print()
    print("Auto-allocation correctly avoids explicit allocations!")


if __name__ == "__main__":
    print("=" * 80)
    print("Memory Allocation and Symbol Definition Tests")
    print("=" * 80)
    print()

    # Run tests
    test_memory_allocation()
    test_address_conflict()
    test_auto_allocation_order()

    print("🎉 All tests passed!")
