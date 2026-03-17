#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test addressing mode selection."""

from r65.compiler.codegen import (
    AddressingModeSelector,
    AddressingMode,
    PhysicalLocation,
    LocationKind,
)


def test_direct_page_vs_absolute():
    """Test selection between direct page and absolute modes."""
    print("=" * 80)
    print("Test 1: Direct Page vs Absolute Selection")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # Test zero-page addresses (should use direct page)
    test_cases = [
        (0x00, AddressingMode.DIRECT_PAGE, "$00"),
        (0x10, AddressingMode.DIRECT_PAGE, "$10"),
        (0x20, AddressingMode.DIRECT_PAGE, "$20"),
        (0xFF, AddressingMode.DIRECT_PAGE, "$FF"),
    ]

    print("Zero-page addresses (should use direct page):")
    for addr, expected_mode, expected_operand in test_cases:
        loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=addr, size=1)
        mode, operand = selector.select_for_location(loc)

        status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
        print(f"  {status} ${addr:04X} → {mode.value:15} {operand}")

    print()

    # Test absolute addresses (should use absolute mode)
    test_cases = [
        (0x0100, AddressingMode.ABSOLUTE, "$0100"),
        (0x2000, AddressingMode.ABSOLUTE, "$2000"),
        (0x7E00, AddressingMode.ABSOLUTE, "$7E00"),
        (0xFFFF, AddressingMode.ABSOLUTE, "$FFFF"),
    ]

    print("Absolute addresses (should use absolute mode):")
    for addr, expected_mode, expected_operand in test_cases:
        loc = PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=addr, size=1)
        mode, operand = selector.select_for_location(loc)

        status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
        print(f"  {status} ${addr:04X} → {mode.value:15} {operand}")

    print()

    # Test long addresses (24-bit)
    test_cases = [
        (0x7E2000, AddressingMode.LONG, "$7E0000"),
        (0x7FFFFF, AddressingMode.LONG, "$7FFFFF"),
    ]

    print("Long addresses (24-bit):")
    for addr, expected_mode, expected_operand in test_cases:
        loc = PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=addr, size=1)
        mode, operand = selector.select_for_location(loc)

        status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
        print(f"  {status} ${addr:06X} → {mode.value:15} {operand}")

    print()


def test_indexed_addressing():
    """Test indexed addressing modes (,X and ,Y)."""
    print("=" * 80)
    print("Test 2: Indexed Addressing Modes")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # Direct page indexed
    print("Direct page indexed:")
    test_cases = [
        (0x20, 'X', AddressingMode.DIRECT_PAGE_X, "$20,X"),
        (0x30, 'Y', AddressingMode.DIRECT_PAGE_Y, "$30,Y"),
    ]

    for addr, index, expected_mode, expected_operand in test_cases:
        loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=addr, size=1)
        mode, operand = selector.select_for_location(loc, index_register=index)

        status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
        print(f"  {status} ${addr:02X},{index} → {mode.value:15} {operand}")

    print()

    # Absolute indexed
    print("Absolute indexed:")
    test_cases = [
        (0x2000, 'X', AddressingMode.ABSOLUTE_X, "$2000,X"),
        (0x3000, 'Y', AddressingMode.ABSOLUTE_Y, "$3000,Y"),
    ]

    for addr, index, expected_mode, expected_operand in test_cases:
        loc = PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=addr, size=1)
        mode, operand = selector.select_for_location(loc, index_register=index)

        status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
        print(f"  {status} ${addr:04X},{index} → {mode.value:15} {operand}")

    print()

    # Long indexed (only X)
    print("Long indexed (24-bit):")
    loc = PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=0x7E2000, size=1)
    mode, operand = selector.select_for_location(loc, index_register='X')

    expected_mode = AddressingMode.LONG_X
    expected_operand = "$7E0000,X"
    status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
    print(f"  {status} $7E0000,X → {mode.value:15} {operand}")

    print()


def test_indirect_addressing():
    """Test indirect addressing modes."""
    print("=" * 80)
    print("Test 3: Indirect Addressing Modes")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # Simple indirect
    print("Simple indirect (zero-page pointer):")
    loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x42, size=1)
    mode, operand = selector.select_for_location(loc, is_indirect=True)

    expected_mode = AddressingMode.INDIRECT
    expected_operand = "($42)"
    status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
    print(f"  {status} ($42) → {mode.value:15} {operand}")

    print()

    # Indirect indexed with Y
    print("Indirect indexed with Y:")
    loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x50, size=1)
    mode, operand = selector.select_for_location(loc, index_register='Y', is_indirect=True)

    expected_mode = AddressingMode.INDIRECT_Y
    expected_operand = "($50),Y"
    status = "✅" if mode == expected_mode and operand == expected_operand else "❌"
    print(f"  {status} ($50),Y → {mode.value:15} {operand}")

    print()


def test_immediate_mode():
    """Test immediate addressing mode."""
    print("=" * 80)
    print("Test 4: Immediate Addressing Mode")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # 8-bit immediate
    print("8-bit immediate values:")
    test_cases = [
        (0x00, False, "#$00"),
        (0x42, False, "#$42"),
        (0xFF, False, "#$FF"),
    ]

    for value, is_16bit, expected_operand in test_cases:
        mode, operand = selector.select_immediate(value, is_16bit)

        status = "✅" if mode == AddressingMode.IMMEDIATE and operand == expected_operand else "❌"
        print(f"  {status} #{value:02X} → {operand}")

    print()

    # 16-bit immediate
    print("16-bit immediate values:")
    test_cases = [
        (0x0000, True, "#$0000"),
        (0x1234, True, "#$1234"),
        (0xFFFF, True, "#$FFFF"),
    ]

    for value, is_16bit, expected_operand in test_cases:
        mode, operand = selector.select_immediate(value, is_16bit)

        status = "✅" if mode == AddressingMode.IMMEDIATE and operand == expected_operand else "❌"
        print(f"  {status} #{value:04X} → {operand}")

    print()


def test_optimization_helpers():
    """Test optimization helper methods."""
    print("=" * 80)
    print("Test 5: Optimization Helpers")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # Test can_use_direct_page
    print("Can use direct page:")
    test_cases = [
        (0x00, True),
        (0x50, True),
        (0xFF, True),
        (0x100, False),
        (0x2000, False),
    ]

    for addr, expected in test_cases:
        result = selector.can_use_direct_page(addr)
        status = "✅" if result == expected else "❌"
        print(f"  {status} ${addr:04X} → {result}")

    print()

    # Test should_use_stz
    print("Should use STZ (store zero optimization):")
    test_cases = [
        (0, True),
        (1, False),
        (42, False),
    ]

    for value, expected in test_cases:
        result = selector.should_use_stz(value)
        status = "✅" if result == expected else "❌"
        print(f"  {status} value={value} → {result}")

    print()

    # Test cycle counts
    print("Cycle count estimates:")
    test_modes = [
        (AddressingMode.IMMEDIATE, "LDA", 2),
        (AddressingMode.DIRECT_PAGE, "LDA", 3),
        (AddressingMode.ABSOLUTE, "LDA", 4),
        (AddressingMode.INDIRECT, "LDA", 5),
    ]

    for mode, instr, expected_cycles in test_modes:
        cycles = selector.get_cycle_count(mode, instr)
        status = "✅" if cycles == expected_cycles else "❌"
        print(f"  {status} {mode.value:15} → {cycles} cycles")

    print()

    # Test instruction sizes
    print("Instruction size estimates:")
    test_modes = [
        (AddressingMode.ACCUMULATOR, "INC", 1),
        (AddressingMode.IMMEDIATE, "LDA", 2),
        (AddressingMode.DIRECT_PAGE, "LDA", 2),
        (AddressingMode.ABSOLUTE, "LDA", 3),
        (AddressingMode.LONG, "LDA", 4),
    ]

    for mode, instr, expected_size in test_modes:
        size = selector.get_byte_size(mode, instr)
        status = "✅" if size == expected_size else "❌"
        print(f"  {status} {mode.value:15} → {size} bytes")

    print()


def test_format_helpers():
    """Test formatting helper methods."""
    print("=" * 80)
    print("Test 6: Format Helpers")
    print("=" * 80)
    print()

    selector = AddressingModeSelector()

    # Test format_operand
    print("Format operand:")
    loc = PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x20, size=1)
    operand = selector.format_operand(loc)
    print(f"  ✅ Direct page: {operand}")

    loc = PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=0x2000, size=1)
    operand = selector.format_operand(loc, index_register='X')
    print(f"  ✅ Absolute,X: {operand}")

    print()

    # Test format_immediate
    print("Format immediate:")
    operand = selector.format_immediate(0x42, is_16bit=False)
    print(f"  ✅ 8-bit: {operand}")

    operand = selector.format_immediate(0x1234, is_16bit=True)
    print(f"  ✅ 16-bit: {operand}")

    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Addressing Mode Selection Tests")
    print("=" * 80)
    print()

    # Run tests
    test_direct_page_vs_absolute()
    test_indexed_addressing()
    test_indirect_addressing()
    test_immediate_mode()
    test_optimization_helpers()
    test_format_helpers()

    print("🎉 All tests passed!")
