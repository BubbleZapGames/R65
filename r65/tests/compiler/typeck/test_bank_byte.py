# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for bank_byte() built-in method on far pointers.

Validates:
- bank_byte() works on immutable far pointers
- bank_byte() rejects mutable far pointers
- bank_byte() rejects near pointers
- bank_byte() rejects non-pointer types
- Const evaluation for compile-time known values
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.hir.errors import HIRError


def compile_and_type_check(source: str):
    """Helper to compile and type check source code."""
    program = parse(source)
    program = expand_macros(program)
    builder = HIRBuilder()
    hir = builder.build_program(program)
    type_checker = TypeChecker(hir)
    type_checker.check()
    return hir


class TestBankByteTypeCheck:
    """Type checking tests for bank_byte() method."""

    def test_bank_byte_on_immutable_far_pointer(self):
        """bank_byte() should work on immutable far pointer variables."""
        source = """
        fn test(ptr @ A: u8) {
            let far_ptr: far *u8 = 0x7E2000 as far *u8;
            A = far_ptr.bank_byte();
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_bank_byte_on_mutable_far_pointer(self):
        """bank_byte() should work on mutable far pointer variables at runtime."""
        source = """
        fn test(ptr @ A: u8) {
            let mut far_ptr: far *u8 = 0x7E2000 as far *u8;
            A = far_ptr.bank_byte();
        }
        """
        compile_and_type_check(source)  # Should not raise

    def test_bank_byte_rejects_near_pointer(self):
        """bank_byte() should reject near pointers."""
        source = """
        #[zeropage]
        static mut BUF: u8;
        fn test(ptr @ A: u8) {
            let p: *u8 = &BUF;
            A = p.bank_byte();
        }
        """
        with pytest.raises(TypeCheckError, match="far pointers"):
            compile_and_type_check(source)

    def test_bank_byte_rejects_non_pointer(self):
        """bank_byte() should reject non-pointer types."""
        source = """
        fn test(val @ A: u8) {
            let x: u16 = 0x1234;
            A = x.bank_byte();
        }
        """
        with pytest.raises((TypeCheckError, HIRError)):
            compile_and_type_check(source)

    def test_bank_byte_rejects_arguments(self):
        """bank_byte() takes no arguments."""
        source = """
        fn test(ptr @ A: u8) {
            let far_ptr: far *u8 = 0x7E2000;
            A = far_ptr.bank_byte(1);
        }
        """
        with pytest.raises(HIRError, match="takes no arguments"):
            compile_and_type_check(source)


class TestBankByteConstEval:
    """Const evaluation tests for bank_byte()."""

    def test_const_eval_literal_cast(self):
        """bank_byte() on a cast literal should be const-evaluated."""
        source = """
        const BANK: u8 = (0x7E2000 as far *u8).bank_byte();
        fn test(val @ A: u8) {
            A = BANK;
        }
        """
        hir = compile_and_type_check(source)
        # Find the const and verify its value
        for decl in hir.declarations:
            from r65.compiler.hir.nodes import HIRConstDecl
            if isinstance(decl, HIRConstDecl) and decl.name == 'BANK':
                assert decl.value.value == 0x7E
                break
        else:
            pytest.fail("BANK const not found")

    def test_const_eval_bank_00(self):
        """bank_byte() of address 0x002000 should be 0x00."""
        source = """
        const BANK: u8 = (0x002000 as far *u8).bank_byte();
        fn test(val @ A: u8) {
            A = BANK;
        }
        """
        hir = compile_and_type_check(source)
        for decl in hir.declarations:
            from r65.compiler.hir.nodes import HIRConstDecl
            if isinstance(decl, HIRConstDecl) and decl.name == 'BANK':
                assert decl.value.value == 0x00
                break
        else:
            pytest.fail("BANK const not found")

    def test_const_eval_bank_7F(self):
        """bank_byte() of address 0x7F0000 should be 0x7F."""
        source = """
        const BANK: u8 = (0x7F0000 as far *u8).bank_byte();
        fn test(val @ A: u8) {
            A = BANK;
        }
        """
        hir = compile_and_type_check(source)
        for decl in hir.declarations:
            from r65.compiler.hir.nodes import HIRConstDecl
            if isinstance(decl, HIRConstDecl) and decl.name == 'BANK':
                assert decl.value.value == 0x7F
                break
        else:
            pytest.fail("BANK const not found")
