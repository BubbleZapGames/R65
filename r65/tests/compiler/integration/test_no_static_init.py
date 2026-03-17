#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test program with no static initializers (undefined values)."""

from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.mir import MIRBuilder

def test_no_static_initializers():
    """Test that no __init_start() is generated for programs without initializers."""
    
    source = """
#[zeropage(0x20)]
static mut COUNTER: u8;  // No initializer - undefined value!

#[entry]
fn main() -> ! {
    COUNTER = 0;  // Must initialize manually in code
    loop {
        COUNTER = COUNTER + 1;
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
    
    # Should have no __init_start() function
    init_functions = [f for f in mir_program.functions if f.name == "__init_start"]
    assert len(init_functions) == 0, f"❌ UNEXPECTED: {len(init_functions)} __init_start() functions found"
    print("✅ CORRECT: No __init_start() generated (no explicit initializers)")
    
    # main() should not call __init_start() at start
    main_func = [f for f in mir_program.functions if f.name == "main"]
    assert len(main_func) == 1, "❌ UNEXPECTED: No main() function found"

    # Collect all instructions from all blocks
    main_instructions = []
    for block in main_func[0].blocks.values():
        main_instructions.extend(block.instructions)

    has_init_call = any(
        "Call" in str(type(instr)) and
        hasattr(instr, 'function') and
        instr.function == "__init_start"
        for instr in main_instructions
    )

    assert not has_init_call, "❌ UNEXPECTED: main() calls __init_start()"
    print("✅ CORRECT: main() doesn't call __init_start()")
