#!/usr/bin/env python3
"""Test AssemblyEmitter basic functionality."""

from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.asm_nodes import Immediate, Address


def test_basic_structure():
    """Test basic assembly file structure generation."""
    emitter = AssemblyEmitter(source_file="test.r65")

    # File header
    emitter.emit_file_header()

    # Processor directive
    emitter.emit_processor_directive()

    # Memory map
    emitter.emit_memory_map(rom_type="lorom", banks=1)

    # Direct page allocations
    emitter.emit_section_header("Direct Page Allocations")
    emitter.emit_define("TEMP", 0x20, "test.r65:5")
    emitter.emit_define("COUNTER", 0x22, "test.r65:8")
    emitter.emit_blank_line()

    # Constants
    emitter.emit_section_header("Constants")
    emitter.emit_equ("SCREEN_WIDTH", 256)
    emitter.emit_equ("SCREEN_HEIGHT", 224)
    emitter.emit_blank_line()

    # Code section
    emitter.emit_section_header("Bank 0 - Main Code")
    emitter.emit_bank_directive(0)

    # Simple function
    emitter.emit_subsection_header("main")
    emitter.emit_comment("Source: test.r65:10-15")
    emitter.emit_comment("")
    emitter.emit_comment("Entry point")
    emitter.emit_subsection_header("")
    emitter.emit_blank_line()

    emitter.emit_label("main")
    emitter.emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x0F), "Load brightness")
    emitter.emit_instr(Opcode.STA_ABSOLUTE, Address(0x2100), "INIDISP")
    emitter.emit_label("__L1")
    emitter.emit_instr(Opcode.JMP_ABSOLUTE, Address("__L1"), "Infinite loop")
    emitter.emit_blank_line()

    # Interrupt vectors
    emitter.emit_interrupt_vectors(reset="main")

    # Symbol exports
    emitter.emit_exports(["main", "TEMP", "COUNTER"])

    # Print output
    print(emitter.to_string())
    print()
    print("=" * 80)
    print("AssemblyEmitter test completed successfully!")
    print(f"Generated {len(emitter.to_lines())} lines of assembly")


def test_instructions():
    """Test instruction emission."""
    emitter = AssemblyEmitter()

    emitter.emit_comment("Test various instructions")
    emitter.emit_blank_line()

    # Immediate mode
    emitter.emit_instr(Opcode.LDA_IMMEDIATE, Immediate(0x42))

    # Direct page
    emitter.emit_instr(Opcode.STA_DP, Address(0x20))

    # Absolute
    emitter.emit_instr(Opcode.STA_ABSOLUTE, Address(0x7E2000))

    # Indexed
    emitter.emit_instr(Opcode.LDA_DP_X, Address(0x20))

    # No operand
    emitter.emit_instr(Opcode.RTS)

    # With comments
    emitter.emit_instr(Opcode.PHP, comment="Push processor status")
    emitter.emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x30), "16-bit mode")

    print(emitter.to_string())
    print()
    print("Instruction emission test passed!")


def test_data_directives():
    """Test data directive emission."""
    emitter = AssemblyEmitter()

    emitter.emit_comment("Test data directives")
    emitter.emit_blank_line()

    # Byte data
    emitter.emit_label("sin_table")
    emitter.emit_byte([0x80, 0x83, 0x86, 0x89, 0x8C, 0x8F, 0x92, 0x95,
                      0x98, 0x9B, 0x9E, 0xA2, 0xA5, 0xA8, 0xAB, 0xAE])
    emitter.emit_blank_line()

    # Word data
    emitter.emit_label("level_data")
    emitter.emit_word([0x0000, 0x0100, 0x0200, 0x0300])
    emitter.emit_blank_line()

    # Space reservation
    emitter.emit_label("buffer")
    emitter.emit_space(256, 0)

    print(emitter.to_string())
    print()
    print("Data directive test passed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing AssemblyEmitter")
    print("=" * 80)
    print()

    test_basic_structure()
    print()

    print("=" * 80)
    print("Testing Instructions")
    print("=" * 80)
    print()

    test_instructions()
    print()

    print("=" * 80)
    print("Testing Data Directives")
    print("=" * 80)
    print()

    test_data_directives()
