# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for integer literal overflow detection and pointer coercion safety.

Validates:
- Integer literals outside the declared type's range are rejected
- Register assignments (A/X/Y) allow wide literals (auto-mode-widening)
- Explicit `as` casts bypass overflow errors
- Far→near pointer coercion in function args is rejected
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


class TestLiteralOverflow:
    """Tests for integer literal range checking on typed variables."""

    def test_u8_overflow_rejected(self):
        """let x: u8 = 256 should fail (max 255)."""
        with pytest.raises(TypeCheckError, match="does not fit in type u8"):
            compile_source('fn test() { let x: u8 = 256; }')

    def test_u8_negative_rejected(self):
        """let x: u8 = -1 should fail (u8 is unsigned)."""
        with pytest.raises(TypeCheckError, match="does not fit in type u8"):
            compile_source('fn test() { let x: u8 = -1; }')

    def test_u16_overflow_rejected(self):
        """let x: u16 = 0x10000 should fail (max 0xFFFF)."""
        with pytest.raises(TypeCheckError, match="does not fit in type u16"):
            compile_source('fn test() { let x: u16 = 0x10000; }')

    def test_i8_positive_overflow_rejected(self):
        """let x: i8 = 128 should fail (max 127)."""
        with pytest.raises(TypeCheckError, match="does not fit in type i8"):
            compile_source('fn test() { let x: i8 = 128; }')

    def test_i8_negative_overflow_rejected(self):
        """let x: i8 = -129 should fail (min -128)."""
        with pytest.raises(TypeCheckError, match="does not fit in type i8"):
            compile_source('fn test() { let x: i8 = -129; }')

    def test_u8_max_allowed(self):
        """let x: u8 = 255 is valid."""
        compile_source('fn test() { let x: u8 = 255; }')

    def test_u8_min_allowed(self):
        """let x: u8 = 0 is valid."""
        compile_source('fn test() { let x: u8 = 0; }')

    def test_i8_min_allowed(self):
        """let x: i8 = -128 is valid (edge case: -min)."""
        compile_source('fn test() { let x: i8 = -128; }')

    def test_i8_max_allowed(self):
        """let x: i8 = 127 is valid."""
        compile_source('fn test() { let x: i8 = 127; }')

    def test_i16_min_allowed(self):
        """let x: i16 = -32768 is valid (edge case: -min)."""
        compile_source('fn test() { let x: i16 = -32768; }')

    def test_explicit_cast_bypass(self):
        """let x: u8 = 256 as u8 compiles (explicit truncation)."""
        compile_source('fn test() { let x: u8 = 256 as u8; }')

    def test_register_wide_literal_allowed(self):
        """A = 0x1234 compiles (register auto-widens to m16 mode)."""
        compile_source('fn test() { A = 0x1234; }')

    def test_x_wide_literal_allowed(self):
        """X = 0xFFFF compiles (X is always u16)."""
        compile_source('fn test() { X = 0xFFFF; }')

    def test_static_overflow_rejected(self):
        """static mut VAR: u8 = 256 should fail."""
        with pytest.raises(TypeCheckError, match="does not fit in type u8"):
            compile_source('#[zeropage] static mut VAR: u8 = 256;')


class TestPointerCoercion:
    """Tests for far↔near pointer coercion safety."""

    def test_far_to_near_param_rejected(self):
        """Passing far *u8 to *u8 parameter is rejected."""
        source = '''
        fn takes_near(p: *u8) {}
        fn test() { takes_near(0x7E1000 as far *u8); }
        '''
        with pytest.raises(TypeCheckError, match="cannot pass far pointer"):
            compile_source(source)

    def test_far_var_to_near_param_rejected(self):
        """Passing a far *u8 variable to *u8 parameter is rejected."""
        source = '''
        fn takes_near(p: *u8) {}
        fn test() {
            let fp: far *u8 = 0x7E1000 as far *u8;
            takes_near(fp);
        }
        '''
        with pytest.raises(TypeCheckError, match="cannot pass far pointer"):
            compile_source(source)

    def test_near_to_far_param_allowed(self):
        """Passing *u8 to far *u8 parameter is allowed (widening)."""
        source = '''
        fn takes_far(p: far *u8) {}
        fn test() { takes_far(0x1000 as *u8); }
        '''
        compile_source(source)

    def test_explicit_cast_bypasses_error(self):
        """Explicit (far_ptr as *u8) cast to near is allowed."""
        source = '''
        fn takes_near(p: *u8) {}
        fn test() { takes_near((0x7E1000 as far *u8) as *u8); }
        '''
        compile_source(source)

    def test_ram_address_to_far_param_allowed(self):
        """&RAM_VAR passed to far *u8 param works (far address)."""
        source = '''
        #[ram] static mut BUF: u8;
        fn takes_far(p: far *u8) {}
        fn test() { takes_far(&BUF); }
        '''
        compile_source(source)
