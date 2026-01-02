#!/usr/bin/env python3
"""Test function code generation."""

from r65.compiler.codegen import (
    AssemblyEmitter,
    FunctionCodeGenerator,
    MemoryAllocator,
)
from r65.compiler.mir.nodes import (
    MIRFunction,
    BasicBlock,
    Move, BinaryOp, Jump, CondBranch, Return,
    VirtualRegister,
    Immediate,
)
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def create_simple_function():
    """Create a simple MIR function for testing."""
    # Function: add(a: u8, b: u8) -> u8
    #   let sum @ %0 = a + b
    #   return sum

    func = MIRFunction(
        name="add",
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

    # Create virtual registers
    vreg_a = VirtualRegister(id=0, type_info=BasicTypeInfo('u8'), hint="a")
    vreg_b = VirtualRegister(id=1, type_info=BasicTypeInfo('u8'), hint="b")
    vreg_sum = VirtualRegister(id=2, type_info=BasicTypeInfo('u8'), hint="sum")

    # Entry block
    entry_block = BasicBlock(
        block_id=0,
        instructions=[
            # Initialize a and b
            Move(dest=vreg_a, source=Immediate(10), type_info=BasicTypeInfo('u8')),
            Move(dest=vreg_b, source=Immediate(20), type_info=BasicTypeInfo('u8')),

            # sum = a + b
            BinaryOp(
                dest=vreg_sum,
                left=vreg_a,
                right=vreg_b,
                op='+',
                type_info=BasicTypeInfo('u8')
            ),

            # return
            Return(values=[vreg_sum])
        ],
        predecessors=[],
        successors=[]
    )

    func.blocks[0] = entry_block
    return func


def create_function_with_branches():
    """Create MIR function with conditional branches."""
    # Function: max(a: u8, b: u8) -> u8
    #   if a > b:
    #       return a
    #   else:
    #       return b

    func = MIRFunction(
        name="max",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={},
        entry_block_id=0,
        exit_block_ids=[1, 2],
        mode_attr=None,
        preserves_attr=None,
        bank_attr=None,
        interrupt_attr=None,
        is_entry=False,
        is_far=False,
        vreg_allocator=VirtualRegisterAllocator(),
        alias_tracker=None,
    )

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

    print()
    return all_passed


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
        ("Jump instruction", "JMP", assembly),
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

    print()
    return all_passed


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

    print()
    return all_passed


if __name__ == "__main__":
    print("=" * 80)
    print("Function Code Generation Tests")
    print("=" * 80)
    print()

    # Run tests
    test1_passed = test_simple_function_generation()
    test2_passed = test_function_with_branches()
    test3_passed = test_function_header()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Simple Function: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Function with Branches: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Function Header: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print()

    if test1_passed and test2_passed and test3_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
