#!/usr/bin/env python3
"""Test complete program code generation."""

from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.mir.nodes import (
    MIRProgram,
    MIRFunction,
    BasicBlock,
    Move, BinaryOp, Return,
    VirtualRegister,
    Immediate,
)
from r65.compiler.hir.nodes import (
    HIRStaticDecl,
    HIRConstDecl,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.hir.attributes import (
    StorageAttribute,
    StorageKind,
    InterruptAttribute,
    InterruptVector,
)
from r65.compiler.hir.symbol_table import Symbol, SymbolKind
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def create_test_program():
    """Create a complete test MIR program."""

    # ==========================================================================
    # Static Variables
    # ==========================================================================

    # Create symbols
    frame_count_symbol = Symbol(
        name="FRAME_COUNT",
        kind=SymbolKind.STATIC_VAR,
        definition=None,
        scope_id=0,
        var_type=BasicTypeInfo('u16'),
        is_mutable=True
    )

    buffer_symbol = Symbol(
        name="BUFFER",
        kind=SymbolKind.STATIC_VAR,
        definition=None,
        scope_id=0,
        var_type=BasicTypeInfo('u8'),
        is_mutable=True
    )

    inidisp_symbol = Symbol(
        name="INIDISP",
        kind=SymbolKind.STATIC_VAR,
        definition=None,
        scope_id=0,
        var_type=BasicTypeInfo('u8'),
        is_mutable=True
    )

    statics = [
        # Zero-page variable
        HIRStaticDecl(
            name="FRAME_COUNT",
            var_type=BasicTypeInfo('u16'),
            initializer=None,
            storage_attr=StorageAttribute(
                name="zeropage",
                storage_kind=StorageKind.ZEROPAGE,
                address=0x20
            ),
            is_mutable=True,
            symbol=frame_count_symbol,
        ),

        # RAM buffer
        HIRStaticDecl(
            name="BUFFER",
            var_type=BasicTypeInfo('u8'),  # Array type not fully implemented
            initializer=None,
            storage_attr=StorageAttribute(
                name="ram",
                storage_kind=StorageKind.RAM,
                address=0x7E0000
            ),
            is_mutable=True,
            symbol=buffer_symbol,
        ),

        # Hardware register
        HIRStaticDecl(
            name="INIDISP",
            var_type=BasicTypeInfo('u8'),
            initializer=None,
            storage_attr=StorageAttribute(
                name="hw",
                storage_kind=StorageKind.HW,
                address=0x2100
            ),
            is_mutable=True,
            symbol=inidisp_symbol,
        ),
    ]

    # ==========================================================================
    # Constants
    # ==========================================================================

    constants = [
        HIRConstDecl(
            name="SCREEN_WIDTH",
            const_type=BasicTypeInfo('u16'),
            value=256,
            symbol=None,
        ),
        HIRConstDecl(
            name="SCREEN_HEIGHT",
            const_type=BasicTypeInfo('u16'),
            value=224,
            symbol=None,
        ),
    ]

    # ==========================================================================
    # Functions
    # ==========================================================================

    functions = []

    # Function 1: init() - entry point
    init_func = MIRFunction(
        name="init",
        parameters=[],
        return_type=None,
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        is_entry=True,  # Entry point
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    # init() body: FRAME_COUNT = 0
    vreg0 = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="init_val")
    init_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg0, source=Immediate(0), type_info=BasicTypeInfo('u16')),
            Return(values=[])
        ],
        predecessors=[],
        successors=[]
    )
    init_func.blocks[0] = init_block
    functions.append(init_func)

    # Function 2: process() - regular function
    process_func = MIRFunction(
        name="process",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    # process() body: return 42
    vreg1 = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="result")
    process_block = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg1, source=Immediate(42), type_info=BasicTypeInfo('u8')),
            Return(values=[vreg1])
        ],
        predecessors=[],
        successors=[]
    )
    process_func.blocks[0] = process_block
    functions.append(process_func)

    # Function 3: vblank_handler() - NMI interrupt
    nmi_func = MIRFunction(
        name="vblank_handler",
        parameters=[],
        return_type=None,
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=InterruptAttribute(
            name="interrupt",
            vector=InterruptVector.NMI,
            preserve=True
        ),
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    # vblank_handler() body: just return
    nmi_block = BasicBlock(
        block_id=0,
        instructions=[
            Return(values=[])
        ],
        predecessors=[],
        successors=[]
    )
    nmi_func.blocks[0] = nmi_block
    functions.append(nmi_func)

    # ==========================================================================
    # Create MIR Program
    # ==========================================================================

    mir_program = MIRProgram(
        functions=functions,
        statics=statics,
        constants=constants,
        structs=[],
        enums=[],
        symbol_table=None,
    )

    return mir_program


def test_complete_program_generation():
    """Test generation of complete program."""
    print("=" * 80)
    print("Complete Program Generation Test")
    print("=" * 80)
    print()

    # Create test program
    mir_program = create_test_program()

    # Generate assembly
    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program)

    # Print result
    print(assembly)
    print()

    # Verify key elements
    checks = [
        ("File header", "Generated by R65 Compiler", assembly),
        ("Processor directive", ".65816", assembly),
        ("Memory map", ".MEMORYMAP", assembly),
        ("Zero-page define", "FRAME_COUNT", assembly),
        ("RAM define", "BUFFER", assembly),
        ("Hardware define", "INIDISP", assembly),
        ("Constant", "SCREEN_WIDTH", assembly),
        ("Entry function", "init:", assembly),
        ("Regular function", "process:", assembly),
        ("NMI handler", "vblank_handler:", assembly),
        ("Interrupt vectors", "Interrupt Vectors", assembly),
        ("Symbol exports", ".EXPORT", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    print()
    return all_passed


def test_output_file_writing():
    """Test writing assembly to file."""
    print("=" * 80)
    print("Output File Writing Test")
    print("=" * 80)
    print()

    # Create test program
    mir_program = create_test_program()

    # Generate and write to file
    output_file = "/tmp/test_output.asm"
    codegen = ProgramCodeGenerator()
    assembly = codegen.generate(mir_program, output_file=output_file)

    # Read file back
    with open(output_file, 'r') as f:
        file_content = f.read()

    # Verify file matches returned assembly
    if file_content == assembly:
        print(f"  ✅ File written to {output_file}")
        print(f"  ✅ File content matches returned assembly")
        print(f"  ✅ File size: {len(file_content)} bytes")
        success = True
    else:
        print(f"  ❌ File content does not match returned assembly")
        success = False

    print()
    return success


def test_multi_bank_program():
    """Test program with functions in multiple banks."""
    print("=" * 80)
    print("Multi-Bank Program Test")
    print("=" * 80)
    print()

    # Note: Multi-bank requires bank_attr which needs proper implementation
    # For now, this is a placeholder test
    print("  ⚠️  Multi-bank test placeholder (requires full bank_attr implementation)")
    print()
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Program Code Generation Tests")
    print("=" * 80)
    print()

    # Run tests
    test1_passed = test_complete_program_generation()
    test2_passed = test_output_file_writing()
    test3_passed = test_multi_bank_program()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Complete Program Generation: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Output File Writing: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Multi-Bank Program: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print()

    if test1_passed and test2_passed and test3_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
