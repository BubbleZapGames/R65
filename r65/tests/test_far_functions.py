#!/usr/bin/env python3
"""Test far function calls and DBR management."""

from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.mir.nodes import (
    MIRProgram,
    MIRFunction,
    BasicBlock,
    Move, Return, Call,
    VirtualRegister,
    Immediate,
    HardwareRegister,
    Argument,
    ArgumentMechanism,
)
from r65.compiler.hir.attributes import (
    BankAttribute,
    DataBankMode,
    ModeAttribute,
    MMode,
    XMode,
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def test_basic_far_function():
    """Test basic far function with JSL/RTL."""

    # Create a simple far function that returns A
    vreg_alloc = VirtualRegisterAllocator()

    entry_block = BasicBlock(block_id=0)
    entry_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(42),
            type_info=None
        ),
        Return(values=[HardwareRegister('A')])
    ]

    far_func = MIRFunction(
        name="far_helper",
        parameters=[],
        return_type=None,
        blocks={0: entry_block},
        entry_block_id=0,
        is_far=True,  # Far function
        bank_attr=BankAttribute(
            name='bank',
            bank_number=1,
            data_bank=DataBankMode.NONE  # No DBR management
        ),
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[far_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify RTL is emitted instead of RTS
    assert 'RTL' in asm_output, "Far function should emit RTL"
    assert 'RTS' not in asm_output, "Far function should not emit RTS"

    print("✓ Basic far function test passed")


def test_far_function_with_data_bank_auto():
    """Test far function with data_bank=auto (callee manages DBR)."""

    vreg_alloc = VirtualRegisterAllocator()

    entry_block = BasicBlock(block_id=0)
    entry_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(99),
            type_info=None
        ),
        Return(values=[HardwareRegister('A')])
    ]

    far_func = MIRFunction(
        name="graphics_helper",
        parameters=[],
        return_type=None,
        blocks={0: entry_block},
        entry_block_id=0,
        is_far=True,
        bank_attr=BankAttribute(
            name='bank',
            bank_number=2,
            data_bank=DataBankMode.AUTO  # Callee manages DBR
        ),
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[far_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify DBR management prologue/epilogue
    assert 'PHB' in asm_output, "data_bank=auto should save DBR (PHB)"
    assert 'PLB' in asm_output, "data_bank=auto should restore DBR (PLB)"
    assert '#$02' in asm_output, "data_bank=auto should load bank number"
    assert 'RTL' in asm_output, "Far function should emit RTL"

    # Verify correct sequence
    lines = [line.strip() for line in asm_output.split('\n') if line.strip()]

    # Find PHB (prologue), both PLBs (set and restore), and RTL
    phb_idx = None
    plb_indices = []
    rtl_idx = None

    for i, line in enumerate(lines):
        if 'PHB' in line:
            phb_idx = i
        elif 'PLB' in line:
            plb_indices.append(i)
        elif 'RTL' in line:
            rtl_idx = i

    assert phb_idx is not None, "PHB not found in prologue"
    assert len(plb_indices) >= 2, "Should have 2 PLBs (one to set DBR, one to restore)"
    assert rtl_idx is not None, "RTL not found"

    # First PLB sets DBR (after PHB/LDA/PHA), second PLB restores DBR (before RTL)
    plb_set = plb_indices[0]
    plb_restore = plb_indices[1]

    assert phb_idx < plb_set < plb_restore < rtl_idx, \
        "DBR sequence should be: PHB ... PLB (set) ... PLB (restore) ... RTL"

    print("✓ data_bank=auto test passed")


def test_far_function_with_data_bank_caller():
    """Test far function call with data_bank=caller (caller manages DBR)."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create callee (far function with data_bank=caller)
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(123),
            type_info=None
        ),
        Return(values=[HardwareRegister('A')])
    ]

    callee_func = MIRFunction(
        name="sound_routine",
        parameters=[],
        return_type=None,
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=True,
        bank_attr=BankAttribute(
            name='bank',
            bank_number=3,
            data_bank=DataBankMode.CALLER  # Caller manages DBR
        ),
        vreg_allocator=vreg_alloc
    )

    # Create caller
    caller_block = BasicBlock(block_id=0)
    caller_block.instructions = [
        Call(
            function="sound_routine",
            args=[],
            returns=[],
            is_far=True,
            bank_attr=BankAttribute(  # Pass callee's bank_attr to Call
                name='bank',
                bank_number=3,
                data_bank=DataBankMode.CALLER
            )
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="main",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,  # Near function
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify caller-side DBR management
    # Should have: PHB, LDA #bank, PHA, PLB, JSL, PLB
    assert 'PHB' in asm_output, "data_bank=caller should save DBR (PHB) at caller"
    assert 'JSL' in asm_output, "Far call should use JSL"
    assert '#$03' in asm_output, "Caller should load callee's bank number"

    # Find the main function in output
    main_start = asm_output.find('main:')
    assert main_start != -1, "main function not found"
    main_section = asm_output[main_start:main_start+500]

    # Verify sequence: PHB ... JSL ... PLB (restore)
    lines = [line.strip() for line in main_section.split('\n') if line.strip()]

    phb_count = sum(1 for line in lines if 'PHB' in line)
    plb_count = sum(1 for line in lines if 'PLB' in line)

    # Should have 2 PLBs (one to set DBR, one to restore)
    assert phb_count >= 1, "Caller should have PHB to save DBR"
    assert plb_count >= 2, "Caller should have 2 PLBs (set and restore DBR)"

    print("✓ data_bank=caller test passed")


def test_near_function_uses_rts():
    """Test that near functions use RTS instead of RTL."""

    vreg_alloc = VirtualRegisterAllocator()

    entry_block = BasicBlock(block_id=0)
    entry_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(10),
            type_info=None
        ),
        Return(values=[HardwareRegister('A')])
    ]

    near_func = MIRFunction(
        name="near_helper",
        parameters=[],
        return_type=None,
        blocks={0: entry_block},
        entry_block_id=0,
        is_far=False,  # Near function
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[near_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify RTS is emitted, not RTL
    assert 'RTS' in asm_output, "Near function should emit RTS"
    assert 'RTL' not in asm_output, "Near function should not emit RTL"

    print("✓ Near function RTS test passed")


def test_far_call_jsl():
    """Test that far function calls use JSL instead of JSR."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create callee (far function)
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        Return(values=[])
    ]

    callee_func = MIRFunction(
        name="far_callee",
        parameters=[],
        return_type=None,
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=True,
        bank_attr=BankAttribute(
            name='bank',
            bank_number=1,
            data_bank=DataBankMode.NONE
        ),
        vreg_allocator=vreg_alloc
    )

    # Create caller
    caller_block = BasicBlock(block_id=0)
    caller_block.instructions = [
        Call(
            function="far_callee",
            args=[],
            returns=[],
            is_far=True,  # Far call
            bank_attr=None
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="caller",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify JSL is used for far calls
    assert 'JSL' in asm_output, "Far call should use JSL"
    assert 'JSR far_callee' not in asm_output, "Far call should not use JSR"

    print("✓ Far call JSL test passed")


def test_near_call_jsr():
    """Test that near function calls use JSR instead of JSL."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create callee (near function)
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        Return(values=[])
    ]

    callee_func = MIRFunction(
        name="near_callee",
        parameters=[],
        return_type=None,
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=False,  # Near function
        vreg_allocator=vreg_alloc
    )

    # Create caller
    caller_block = BasicBlock(block_id=0)
    caller_block.instructions = [
        Call(
            function="near_callee",
            args=[],
            returns=[],
            is_far=False,  # Near call
            bank_attr=None
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="caller",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify JSR is used for near calls
    assert 'JSR' in asm_output, "Near call should use JSR"
    assert 'JSL near_callee' not in asm_output, "Near call should not use JSL"

    print("✓ Near call JSR test passed")


def run_all_tests():
    """Run all far function tests."""
    print("\n=== Running Far Function Tests ===\n")

    test_basic_far_function()
    test_far_function_with_data_bank_auto()
    test_far_function_with_data_bank_caller()
    test_near_function_uses_rts()
    test_far_call_jsl()
    test_near_call_jsr()

    print("\n=== All Far Function Tests Passed ===\n")


if __name__ == '__main__':
    run_all_tests()
