#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test AssemblyEmitter basic functionality."""

import pytest

from r65.compiler.codegen.emitter import AssemblyEmitter, _emit_instruction
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
    emitter.emit_instr(Opcode.STA_ABSOLUTE, Address(0x2000))

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


@pytest.mark.parametrize("opcode, address, text", [
    (Opcode.STA_ABSOLUTE, 0x0010, "STA.w $0010"),
    (Opcode.STA_ABSOLUTE, 0x0300, "STA $0300"),
    (Opcode.LDA_ABSOLUTE_X, 0x0010, "LDA.w $0010,X"),
    (Opcode.LDA_LONG, 0x000010, "LDA.l $000010"),
    (Opcode.LDA_LONG, 0x7E2000, "LDA $7E2000"),
])
def test_small_address_keeps_its_mode(opcode, address, text):
    """WLA-DX shrinks `$0010` and `$000010` to direct page, which under D = S
    addresses the stack frame; a sized mnemonic keeps the mode."""
    assert _emit_instruction(opcode, Address(address)) == text


def test_bool_constant_equ_is_numeric():
    """WLA-DX has no True/False: a bool const is written as 1/0."""
    emitter = AssemblyEmitter()
    emitter.emit_equ("DBG", False)
    emitter.emit_equ("T", True)
    assert ".EQU DBG 0" in emitter.to_string()
    assert ".EQU T 1" in emitter.to_string()


def test_capture_leaves_no_trace():
    """A trial emission shows its nodes inside the block and on exit leaves
    the node list and every tracked state as they were."""
    from r65.compiler.codegen.instruction_select_helpers import XBAStateManager

    emitter = AssemblyEmitter()
    emitter.emit_instr(Opcode.TYX)
    xba = XBAStateManager(emitter)
    before = (list(emitter.nodes), dict(vars(emitter)), xba.state)
    with emitter.capture(xba) as trial:
        emitter.emit_instr(Opcode.REP_IMMEDIATE, Immediate(0x20))
        emitter.emit_accu_mode(16)
        emitter.emit_instr(Opcode.PHA)
        emitter.emit_label("trial")
        xba.emit_xba()
        assert emitter.displacement == 2      # the PHA, in m16
        assert [n.opcode for n in trial.nodes if hasattr(n, 'opcode')] == [
            Opcode.REP_IMMEDIATE, Opcode.PHA, Opcode.XBA]
    # vars(emitter) covers the stack depth and its peak
    assert (emitter.nodes, dict(vars(emitter)), xba.state) == before


def test_stack_displacement():
    """The one record of how far S has moved past the frame: pushes and
    pulls counted as emitted, moves no opcode shows noted, TCS no reset."""
    emitter = AssemblyEmitter()
    emitter.emit_instr(Opcode.PHX)            # a prologue push
    emitter.start_body()
    assert emitter.displacement == 0
    emitter.emit_instr(Opcode.PHA)            # m8: one byte
    emitter.emit_instr(Opcode.PHY)
    assert emitter.displacement == 3
    emitter.emit_instr(Opcode.TCS)            # S set; the emitting code notes by how much
    assert emitter.displacement == 3
    emitter.note_stack(-2)                    # say, a callee removed two bytes
    emitter.emit_instr(Opcode.PLA)
    assert emitter.displacement == 0
    assert emitter.peak_displacement == 3
