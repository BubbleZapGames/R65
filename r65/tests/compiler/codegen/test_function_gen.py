#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test function code generation."""

from r65.compiler.codegen import (
    AssemblyEmitter,
    FunctionCodeGenerator,
    MemoryAllocator,
)
from r65.compiler.mir.nodes import (
    MIRFunction, BasicBlock,
    Move, BinaryOp, Jump, CondBranch, Return,
    VirtualRegister,
    Immediate,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.tests.language.common import make_mir_function


def create_simple_function():
    """Create a simple MIR function for testing."""
    func = make_mir_function("add")

    vreg_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a")
    vreg_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b")
    vreg_sum = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="sum")

    func.blocks[0] = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_a, source=Immediate(10), type_info=BasicTypeInfo('u8')),
            Move(dest=vreg_b, source=Immediate(20), type_info=BasicTypeInfo('u8')),
            BinaryOp(dest=vreg_sum, left=vreg_a, right=vreg_b, op='+',
                     type_info=BasicTypeInfo('u8')),
            Return(values=[vreg_sum])
        ],
        predecessors=[], successors=[]
    )
    return func


def create_function_with_branches():
    """Create MIR function with conditional branches."""
    func = make_mir_function("max")
    func.exit_block_ids = [1, 2]

    # Virtual registers
    vreg_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a")
    vreg_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b")
    vreg_cond = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="cond")

    # Block 0: Entry - compare a and b
    block0 = BasicBlock(
        block_id=0,
        instructions=[
            Move(dest=vreg_a, source=Immediate(10), type_info=BasicTypeInfo('u8')),
            Move(dest=vreg_b, source=Immediate(5), type_info=BasicTypeInfo('u8')),

            # Simple comparison: cond = (a != 0)
            Move(dest=vreg_cond, source=vreg_a, type_info=BasicTypeInfo('u8')),

            # Branch based on condition
            CondBranch(
                condition=vreg_cond,
                true_target=1,
                false_target=2,
                comparison='!='
            )
        ],
        predecessors=[],
        successors=[1, 2]
    )

    # Block 1: Return a
    block1 = BasicBlock(
        block_id=1,
        instructions=[
            Return(values=[vreg_a])
        ],
        predecessors=[0],
        successors=[]
    )

    # Block 2: Return b
    block2 = BasicBlock(
        block_id=2,
        instructions=[
            Return(values=[vreg_b])
        ],
        predecessors=[0],
        successors=[]
    )

    func.blocks[0] = block0
    func.blocks[1] = block1
    func.blocks[2] = block2

    return func


def test_simple_function_generation():
    """Test generation of simple function."""
    print("=" * 80)
    print("Test 1: Simple Function Generation")
    print("=" * 80)
    print()

    # Create function
    mir_func = create_simple_function()

    # Setup code generator
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    func_gen = FunctionCodeGenerator(emitter, mem_alloc)

    # Generate function
    print("Generating function 'add':")
    func_gen.generate_function(mir_func)

    # Print result
    assembly = emitter.to_string()
    print(assembly)
    print()

    # Verify key elements
    checks = [
        ("Function label", "add:", assembly),
        ("Move instruction", "LDA", assembly),
        ("Store instruction", "STA", assembly),
        ("Add with carry", "ADC", assembly),
        ("Return", "RTS", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    assert all_passed, "Some verification checks failed"
    print()


def test_function_with_branches():
    """Test generation of function with control flow."""
    print("=" * 80)
    print("Test 2: Function with Branches")
    print("=" * 80)
    print()

    # Create function
    mir_func = create_function_with_branches()

    # Setup code generator
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    func_gen = FunctionCodeGenerator(emitter, mem_alloc)

    # Generate function
    print("Generating function 'max' with conditional branches:")
    func_gen.generate_function(mir_func)

    # Print result
    assembly = emitter.to_string()
    print(assembly)
    print()

    # Verify key elements
    checks = [
        ("Function label", "max:", assembly),
        ("Block label 1", "__L1:", assembly),
        ("Block label 2", "__L2:", assembly),
        ("Branch instruction", "BEQ", assembly),
        ("Return", "RTS", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    assert all_passed, "Some verification checks failed"
    print()


def test_function_header():
    """Test function header generation."""
    print("=" * 80)
    print("Test 3: Function Header Generation")
    print("=" * 80)
    print()

    # Create function with metadata
    func = MIRFunction(
        name="test_function",
        parameters=[],
        return_type=BasicTypeInfo('u16'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[0],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        is_entry=True,  # Entry point
        is_far=True,    # Far function
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

    # Single block with return
    block0 = BasicBlock(
        block_id=0,
        instructions=[Return(values=[])],
        predecessors=[],
        successors=[]
    )
    func.blocks[0] = block0

    # Generate
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    func_gen = FunctionCodeGenerator(emitter, mem_alloc)

    print("Generating function with metadata:")
    func_gen.generate_function(func)

    assembly = emitter.to_string()
    print(assembly)
    print()

    # Verify header elements
    checks = [
        ("Function name in header", "test_function", assembly),
        ("Return type", "Returns:", assembly),
        ("Entry marker", "Entry: true", assembly),
        ("Far marker", "Far: true", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    assert all_passed, "Some verification checks failed"
    print()


def create_function_with_stack_params():
    """Create MIR function with stack parameters."""
    # Function: add(a: u8, b: u8) -> u8
    # Both a and b are stack parameters

    func = MIRFunction(
        name="add_stack",
        parameters=[],  # HIR parameters - not used in MIR
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

    # Virtual registers for parameters
    vreg_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="param_a")
    vreg_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="param_b")
    vreg_sum = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="sum")

    # Set up stack parameter tracking
    # Near function: return address is 2 bytes, so first param at SP+3
    func.stack_param_offsets = {
        0: 3,  # param 0 (a) at SP+3
        1: 4,  # param 1 (b) at SP+4
    }
    func.param_to_vreg = {
        0: vreg_a,
        1: vreg_b,
    }

    # Entry block: add params and return
    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            # sum = a + b
            BinaryOp(
                dest=vreg_sum,
                left=vreg_a,
                right=vreg_b,
                op='+',
                type_info=BasicTypeInfo('u8')
            ),
            # return sum
            Return(values=[vreg_sum])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_function_with_16bit_stack_param():
    """Create MIR function with 16-bit stack parameter."""
    func = MIRFunction(
        name="double_word",
        parameters=[],
        return_type=BasicTypeInfo('u16'),
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

    # Virtual register for 16-bit parameter
    vreg_val = VirtualRegister(id=0, type_info=BasicTypeInfo('u16'), hint="param_val")
    vreg_result = VirtualRegister(id=1, type_info=BasicTypeInfo('u16'), hint="result")

    # Stack parameter tracking for 16-bit param
    func.stack_param_offsets = {
        0: 3,  # param 0 at SP+3 (low byte), SP+4 (high byte)
    }
    func.param_to_vreg = {
        0: vreg_val,
    }

    # Entry block: double the value and return
    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            BinaryOp(
                dest=vreg_result,
                left=vreg_val,
                right=vreg_val,
                op='+',
                type_info=BasicTypeInfo('u16')
            ),
            Return(values=[vreg_result])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def test_stack_parameters():
    """Test generation of stack parameter loading."""
    print("=" * 80)
    print("Test 4: Stack Parameter Loading")
    print("=" * 80)
    print()

    # Create function with stack params
    mir_func = create_function_with_stack_params()

    # Setup code generator
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    func_gen = FunctionCodeGenerator(emitter, mem_alloc)

    # Generate function
    print("Generating function with stack parameters:")
    func_gen.generate_function(mir_func)

    # Print result
    assembly = emitter.to_string()
    print(assembly)
    print()

    # Verify stack parameter loading
    # BinaryOp result coalesces to A, eliminating the frame.
    # Without frame, params are at $03,S and $04,S (return addr at $01-02,S).
    checks = [
        ("Function label", "add_stack:", assembly),
        ("Load param from stack", ",S", assembly),
        ("Add with carry", "ADC", assembly),
        ("Return", "RTS", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    assert all_passed, "Some verification checks failed"
    print()


def test_16bit_stack_parameter():
    """Test generation of 16-bit stack parameter loading."""
    print("=" * 80)
    print("Test 5: 16-bit Stack Parameter Loading")
    print("=" * 80)
    print()

    # Create function with 16-bit stack param
    mir_func = create_function_with_16bit_stack_param()

    # Setup code generator
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    func_gen = FunctionCodeGenerator(emitter, mem_alloc)

    # Generate function
    print("Generating function with 16-bit stack parameter:")
    func_gen.generate_function(mir_func)

    # Print result
    assembly = emitter.to_string()
    print(assembly)
    print()

    # Verify 16-bit stack parameter loading (low and high bytes)
    # BinaryOp result coalesces to A, eliminating the frame.
    checks = [
        ("Function label", "double_word:", assembly),
        ("Load param from stack", ",S", assembly),
        ("Add with carry", "ADC", assembly),
        ("Return", "RTS", assembly),
    ]

    print("Verification:")
    all_passed = True
    for check_name, pattern, text in checks:
        if pattern in text:
            print(f"  ✅ {check_name}: Found '{pattern}'")
        else:
            print(f"  ❌ {check_name}: Missing '{pattern}'")
            all_passed = False

    assert all_passed, "Some verification checks failed"
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Function Code Generation Tests")
    print("=" * 80)
    print()

    # Run tests
    test_simple_function_generation()
    test_function_with_branches()
    test_function_header()
    test_stack_parameters()
    test_16bit_stack_parameter()

    print("🎉 All tests passed!")
