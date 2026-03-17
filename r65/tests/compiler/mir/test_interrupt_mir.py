#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Debug script to inspect interrupt handler MIR."""

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder

source = """
#[zeropage(0x20)]
static mut FLAG: u8 = 0;

#[interrupt(nmi)]
fn nmi_handler() {
    A = 0x42;
    FLAG = A;
    return;
}
"""

# Parse → HIR → Type Check → MIR
program = parse(source, "test.r65")
hir_program = HIRBuilder().build_program(program)
type_checker = TypeChecker(hir_program)
type_checker.check()

mir_builder = MIRBuilder()
mir_program = mir_builder.build_program(hir_program)

# Print MIR for nmi_handler
for mir_func in mir_program.functions:
    if mir_func.name == "nmi_handler":
        print(f"Function: {mir_func.name}")
        print(f"Interrupt: {mir_func.interrupt_attr}")
        print(f"Mode: {mir_func.mode_attr}")
        print()

        for block_id, block in mir_func.blocks.items():
            print(f"Block {block_id}:")
            for i, instr in enumerate(block.instructions):
                print(f"  {i:2d}: {instr}")
            print()
