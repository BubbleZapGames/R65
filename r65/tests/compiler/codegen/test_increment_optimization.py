# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Test increment/decrement optimization for hardware registers.

Verifies that X++, Y++, A++, X--, Y--, A-- generate efficient
INX, INY, INC A, DEX, DEY, DEC A instructions.
"""
from r65.compiler.mir.nodes import (
    BinaryOp, HardwareRegister, Immediate
)
from r65.compiler.hir.types import BasicTypeInfo


def test_x_increment(codegen):
    """Test that X++ (X = X + 1) generates INX."""
    x_reg = HardwareRegister('X')
    instr = BinaryOp(
        dest=x_reg, left=x_reg, op='+',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'INX' in asm, f"X++ should generate INX, got: {asm}"
    assert 'TXA' not in asm, f"X++ should not use TXA, got: {asm}"
    assert 'ADC' not in asm, f"X++ should not use ADC, got: {asm}"


def test_y_increment(codegen):
    """Test that Y++ (Y = Y + 1) generates INY."""
    y_reg = HardwareRegister('Y')
    instr = BinaryOp(
        dest=y_reg, left=y_reg, op='+',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'INY' in asm, f"Y++ should generate INY, got: {asm}"
    assert 'TYA' not in asm, f"Y++ should not use TYA, got: {asm}"


def test_a_increment(codegen):
    """Test that A++ (A = A + 1) generates INC A."""
    a_reg = HardwareRegister('A')
    instr = BinaryOp(
        dest=a_reg, left=a_reg, op='+',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'INC A' in asm or 'INC\tA' in asm, f"A++ should generate INC A, got: {asm}"


def test_x_decrement(codegen):
    """Test that X-- (X = X - 1) generates DEX."""
    x_reg = HardwareRegister('X')
    instr = BinaryOp(
        dest=x_reg, left=x_reg, op='-',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'DEX' in asm, f"X-- should generate DEX, got: {asm}"
    assert 'TXA' not in asm, f"X-- should not use TXA, got: {asm}"


def test_y_decrement(codegen):
    """Test that Y-- (Y = Y - 1) generates DEY."""
    y_reg = HardwareRegister('Y')
    instr = BinaryOp(
        dest=y_reg, left=y_reg, op='-',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'DEY' in asm, f"Y-- should generate DEY, got: {asm}"


def test_a_decrement(codegen):
    """Test that A-- (A = A - 1) generates DEC A."""
    a_reg = HardwareRegister('A')
    instr = BinaryOp(
        dest=a_reg, left=a_reg, op='-',
        right=Immediate(1), type_info=BasicTypeInfo('u8')
    )

    codegen.selector.select_binary_op(instr)
    asm = codegen.emitter.to_string()

    assert 'DEC A' in asm or 'DEC\tA' in asm, f"A-- should generate DEC A, got: {asm}"
