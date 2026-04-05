# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for auto-promotion of near pointers to far when target is WRAM.

Tests that the compiler automatically promotes *T parameters to far *T when
called with &RAM_BUFFER arguments, enabling correct D=S codegen for WRAM access.
"""

import pytest
from r65.tests.e2e import ExpectedState, CompilationError


class TestFarPointerPromotion:
    """Test automatic near-to-far pointer promotion."""

    def test_param_promotion_ram_buffer(self, e2e):
        """fn fill(ptr: *u8) called with &RAM_BUFFER promotes to far *u8, writes to WRAM."""
        result = e2e.run('''
            #[ram]
            static mut BUF: [u8; 4] = [0; 4];

            #[zeropage(0x40)]
            static mut RESULT: u8;

            fn write_first(ptr: *u8, val @ A: u8) {
                *ptr = val;
            }

            fn read_first(ptr: *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                write_first(&BUF, 0x42);
                RESULT = read_first(&BUF);
            }
        ''', ExpectedState(memory={
            0x7E0040: 0x42,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_let_binding_promotion(self, e2e):
        """let p: *u8 = &RAM_BUFFER promotes to far *u8, dereference works."""
        result = e2e.run('''
            #[ram]
            static mut BUF: [u8; 4] = [0; 4];

            #[zeropage(0x40)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                let mut p: *u8 = &BUF;
                *p = 0xAB;
                // Read back via another let binding to verify
                let q: *u8 = &BUF;
                RESULT = *q;
            }
        ''', ExpectedState(memory={
            0x7E0040: 0xAB,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_no_promotion_zeropage(self, e2e):
        """fn fill(ptr: *u8) called with &ZP_VAR stays near (no D=S overhead)."""
        result = e2e.run('''
            #[zeropage(0x40)]
            static mut ZP_VAL: u8 = 0;

            fn write_ptr(ptr: *u8, val @ A: u8) {
                *ptr = val;
            }

            #[zeropage(0x50)]
            static mut RESULT: u8;

            #[entry]
            fn main() {
                write_ptr(&ZP_VAL, 0x77);
                RESULT = ZP_VAL;
            }
        ''', ExpectedState(memory={
            0x7E0050: 0x77,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_mixed_callers_near_and_far(self, e2e):
        """Function called with both far (&RAM) and near (&ZP) args — param promoted,
        near caller zero-extends bank byte to 0."""
        result = e2e.run('''
            #[ram]
            static mut RAM_VAL: u8 = 0;

            #[zeropage(0x40)]
            static mut ZP_VAL: u8 = 0;

            #[zeropage(0x50)]
            static mut RESULT1: u8;
            #[zeropage(0x51)]
            static mut RESULT2: u8;

            fn write_val(ptr: *u8, val @ A: u8) {
                *ptr = val;
            }

            fn read_val(ptr: *u8) -> u8 {
                return *ptr;
            }

            #[entry]
            fn main() {
                write_val(&RAM_VAL, 0xAA);
                write_val(&ZP_VAL, 0xBB);
                RESULT1 = read_val(&RAM_VAL);
                RESULT2 = ZP_VAL;
            }
        ''', ExpectedState(memory={
            0x7E0050: 0xAA,
            0x7E0051: 0xBB,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_static_assignment_far_to_near_error(self, e2e):
        """static mut PTR: *u8; PTR = &RAM_BUF; should give a type error."""
        with pytest.raises(CompilationError, match="cannot assign far pointer to near pointer static"):
            e2e.compile('''
                #[ram]
                static mut BUF: [u8; 4] = [0; 4];

                #[zeropage(0x60)]
                static mut PTR: *u8;

                #[entry]
                fn main() {
                    PTR = &BUF;
                }
            ''')
