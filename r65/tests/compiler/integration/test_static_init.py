#!/usr/bin/env python3
"""Test static initialization (__init_start generation)."""

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder

source = """
#[zeropage(0x20)]
static mut FLAGS: u8 = 0x80;

#[zeropage(0x21)]
static mut COUNTER: u8 = 0;  // Zero - no init

#[zeropage(0x22)]
static mut VALUE: u16 = 0x1234;

#[ram]
static mut LIVES: u8 = 3;

#[entry]
fn main() -> ! {
    loop {
        A = FLAGS;
    }
}
"""

# Parse → HIR → Type Check → MIR
program = parse(source, "test.r65")
hir_program = HIRBuilder().build_program(program)
type_checker = TypeChecker(hir_program)
type_checker.check()

mir_builder = MIRBuilder()
mir_program = mir_builder.build_program(hir_program)

print("Generated MIR functions:")
print("=" * 80)
for mir_func in mir_program.functions:
    print(f"\nFunction: {mir_func.name}")
    print(f"  is_entry: {mir_func.is_entry}")
    print(f"  Blocks: {len(mir_func.blocks)}")
    print()

    for block_id, block in mir_func.blocks.items():
        print(f"  Block {block_id}:")
        for i, instr in enumerate(block.instructions):
            print(f"    {i:2d}: {instr}")

print("\n" + "=" * 80)
print("\nExpected __init_start() to initialize:")
print("  FLAGS = 0x80")
print("  COUNTER = 0  (MUST be initialized - RAM not zeroed on SNES!)")
print("  VALUE = 0x1234")
print("  LIVES = 3")
print("\nExpected main() to start with:")
print("  Call __init_start()")
print("\nNote: SNES RAM contents are unpredictable at power-on.")
