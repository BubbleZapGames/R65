#!/usr/bin/env python3
"""Test transition=caller wrapper generation."""

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder

source = """
#[zeropage(0x20)]
static mut VALUE: u8 = 0;

#[mode(m16, x16, transition=caller)]
fn process_16bit() {
    A = 0x1234;
}

#[mode(m16, x16, transition=caller)]
#[preserves(STATUS)]
fn process_16bit_preserves() {
    A = 0xABCD;
}

#[mode(m8, x8)]
fn caller() {
    process_16bit();
    process_16bit_preserves();
}
"""

# Parse → HIR → Type Check → MIR
program = parse(source, "test.r65")
hir_program = HIRBuilder().build_program(program)
type_checker = TypeChecker(hir_program)
type_checker.check()

mir_builder = MIRBuilder()
mir_program = mir_builder.build_program(hir_program)

# Print MIR for caller function
for mir_func in mir_program.functions:
    if mir_func.name == "caller":
        print(f"Function: {mir_func.name}")
        print(f"Mode: {mir_func.mode_attr}")
        print()

        for block_id, block in mir_func.blocks.items():
            print(f"Block {block_id}:")
            for i, instr in enumerate(block.instructions):
                print(f"  {i:2d}: {instr}")
            print()

print("\nExpected wrapper for process_16bit (no preserves(STATUS)):")
print("  PHP          ; Save STATUS")
print("  REP #$30     ; Switch to m16/x16")
print("  Call process_16bit")
print("  PLP          ; Restore STATUS")
print()

print("Expected wrapper for process_16bit_preserves (has preserves(STATUS)):")
print("  REP #$30     ; Switch to m16/x16")
print("  Call process_16bit_preserves")
print("  SEP #$30     ; Restore to m8/x8")
