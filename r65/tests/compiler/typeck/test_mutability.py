# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for mutability enforcement.

Validates:
- `let x: T = val; x = new_val;` is rejected (immutable)
- `let mut x: T = val; x = new_val;` is allowed
- Compound assignments (`+=`, `-=`) require mut
- Increment/decrement (`x++`, `x--`) require mut
- Struct field and array element mutations always allowed
- Pointer dereference writes always allowed
- Register assignments (A, X, Y) always allowed
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError


def compile_source(source: str):
    """Parse, build HIR, and type check."""
    program = parse(source, '<test>')
    program = expand_macros(program)
    builder = HIRBuilder()
    hir = builder.build_program(program)
    tc = TypeChecker(hir)
    tc.check()
    return hir


class TestLetMutability:
    """Tests for let binding mutability."""

    def test_immutable_let_reassign_rejected(self):
        """let x = 5; x = 10 should fail."""
        with pytest.raises(TypeCheckError, match="cannot assign to immutable"):
            compile_source('fn test() { let x: u8 = 5; x = 10; }')

    def test_immutable_let_compound_assign_rejected(self):
        """let x = 5; x += 1 should fail."""
        with pytest.raises(TypeCheckError, match="cannot assign to immutable"):
            compile_source('fn test() { let x: u8 = 5; x += 1; }')

    def test_immutable_let_increment_rejected(self):
        """let x = 5; x++ should fail."""
        with pytest.raises(TypeCheckError, match="cannot assign to immutable"):
            compile_source('fn test() { let x: u8 = 5; x++; }')

    def test_immutable_let_decrement_rejected(self):
        """let x = 5; x-- should fail."""
        with pytest.raises(TypeCheckError, match="cannot assign to immutable"):
            compile_source('fn test() { let x: u8 = 5; x--; }')

    def test_mut_let_reassign_allowed(self):
        """let mut x = 5; x = 10 is valid."""
        compile_source('fn test() { let mut x: u8 = 5; x = 10; }')

    def test_mut_let_compound_assign_allowed(self):
        """let mut x = 5; x += 1 is valid."""
        compile_source('fn test() { let mut x: u8 = 5; x += 1; }')

    def test_mut_let_increment_allowed(self):
        """let mut x = 5; x++ is valid."""
        compile_source('fn test() { let mut x: u8 = 5; x++; }')


class TestStaticMutability:
    """Tests for static variable mutability."""

    def test_static_reassign_rejected(self):
        """Immutable static cannot be reassigned."""
        with pytest.raises(TypeCheckError, match="cannot assign to immutable"):
            compile_source('static VAR: u8 = 5; fn test() { VAR = 10; }')

    def test_static_mut_reassign_allowed(self):
        """static mut can be reassigned."""
        compile_source('#[zeropage] static mut VAR: u8 = 5; fn test() { VAR = 10; }')


class TestRegisterMutability:
    """Register assignments are always allowed (registers are implicitly mutable)."""

    def test_register_a_assign_allowed(self):
        compile_source('fn test() { A = 5; }')

    def test_register_x_assign_allowed(self):
        compile_source('fn test() { X = 5; }')

    def test_register_y_increment_allowed(self):
        compile_source('fn test() { Y = 5; Y++; }')


class TestPointerDeref:
    """Writes through pointer dereferences are always allowed."""

    def test_deref_write_immutable_ptr(self):
        """Even immutable pointer vars allow *ptr = val (mutating pointee)."""
        compile_source('fn test() { let p: *u8 = 0x1000 as *u8; *p = 5; }')


class TestAggregateMembers:
    """Struct field and array element writes always allowed (R65 idiom)."""

    def test_struct_field_write_immutable_struct(self):
        """let p: Point; p.x = 5 is allowed (R65 lacks let struct literals)."""
        source = '''
        struct Point { x: u8, y: u8 }
        fn test() { let p: Point; p.x = 5; }
        '''
        compile_source(source)

    def test_array_element_write_static_mut_array(self):
        """Writing to static mut array element."""
        source = '''
        #[ram] static mut ARR: [u8; 4] = [0, 0, 0, 0];
        fn test() { ARR[0] = 5; }
        '''
        compile_source(source)


class TestParameterMutability:
    """Parameters are mutable by default (incoming values can be modified)."""

    def test_param_mutation_allowed(self):
        source = '''
        fn test(x: u8) { x = 10; }
        '''
        compile_source(source)
