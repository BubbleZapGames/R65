# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Test increment/decrement optimization for hardware registers.

Verifies that X++, Y++, A++, X--, Y--, A-- generate efficient
INX, INY, INC A, DEX, DEY, DEC A instructions.
"""
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.register_alloc import RegisterAllocator, ScratchRegisterPool
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.mir.nodes import (
    BinaryOp, HardwareRegister, Immediate
)
from r65.compiler.hir.types import BasicTypeInfo


def test_x_increment():
    """Test that X++ (X = X + 1) generates INX."""
    # Setup
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create X = X + 1 instruction
    x_reg = HardwareRegister('X')
    instr = BinaryOp(
        dest=x_reg,
        left=x_reg,
        op='+',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    # Select instruction
    selector.select_binary_op(instr)
    asm = emitter.to_string()

    # Verify INX was generated
    assert 'INX' in asm, f"X++ should generate INX, got: {asm}"
    assert 'TXA' not in asm, f"X++ should not use TXA, got: {asm}"
    assert 'ADC' not in asm, f"X++ should not use ADC, got: {asm}"

    print("✓ X increment (INX) test passed")


def test_y_increment():
    """Test that Y++ (Y = Y + 1) generates INY."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create Y = Y + 1 instruction
    y_reg = HardwareRegister('Y')
    instr = BinaryOp(
        dest=y_reg,
        left=y_reg,
        op='+',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'INY' in asm, f"Y++ should generate INY, got: {asm}"
    assert 'TYA' not in asm, f"Y++ should not use TYA, got: {asm}"

    print("✓ Y increment (INY) test passed")


def test_a_increment():
    """Test that A++ (A = A + 1) generates INC A."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create A = A + 1 instruction
    a_reg = HardwareRegister('A')
    instr = BinaryOp(
        dest=a_reg,
        left=a_reg,
        op='+',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'INC A' in asm or 'INC\tA' in asm, f"A++ should generate INC A, got: {asm}"

    print("✓ A increment (INC A) test passed")


def test_x_decrement():
    """Test that X-- (X = X - 1) generates DEX."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create X = X - 1 instruction
    x_reg = HardwareRegister('X')
    instr = BinaryOp(
        dest=x_reg,
        left=x_reg,
        op='-',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'DEX' in asm, f"X-- should generate DEX, got: {asm}"
    assert 'TXA' not in asm, f"X-- should not use TXA, got: {asm}"

    print("✓ X decrement (DEX) test passed")


def test_y_decrement():
    """Test that Y-- (Y = Y - 1) generates DEY."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create Y = Y - 1 instruction
    y_reg = HardwareRegister('Y')
    instr = BinaryOp(
        dest=y_reg,
        left=y_reg,
        op='-',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'DEY' in asm, f"Y-- should generate DEY, got: {asm}"

    print("✓ Y decrement (DEY) test passed")


def test_a_decrement():
    """Test that A-- (A = A - 1) generates DEC A."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create A = A - 1 instruction
    a_reg = HardwareRegister('A')
    instr = BinaryOp(
        dest=a_reg,
        left=a_reg,
        op='-',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'DEC A' in asm or 'DEC\tA' in asm, f"A-- should generate DEC A, got: {asm}"

    print("✓ A decrement (DEC A) test passed")


if __name__ == '__main__':
    print("Running increment/decrement optimization tests...\n")

    test_x_increment()
    test_y_increment()
    test_a_increment()
    test_x_decrement()
    test_y_decrement()
    test_a_decrement()

    print("\n✅ All increment/decrement optimization tests passed!")
