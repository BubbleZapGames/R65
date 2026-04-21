# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for E-XY16-REGION validator.

R65 assumes x16 (16-bit X/Y). Writing `STATUS.XY16 = false` drops into
x8 mode; the region validator enforces that the resulting region is
straight-line asm-only and is paired with a `STATUS.XY16 = true` restore
before any compiler-generated X/Y code executes.
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.typeck.errors import TypeCheckError


def compile_source(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    builder = HIRBuilder()
    hir = builder.build_program(program)
    TypeChecker(hir).check()
    return hir


class TestXy16RegionPositive:
    """Shapes that are safe and must compile clean."""

    def test_straight_line_asm_region(self):
        compile_source('''
            fn leaf() {
                STATUS.XY16 = false;
                asm!("LDX #$10");
                STATUS.XY16 = true;
            }
        ''')

    def test_no_xy16_writes_is_unaffected(self):
        compile_source('''
            fn normal(n: u8) -> u8 {
                let mut total: u8 = 0;
                for i in 0..10 {
                    total = total + n;
                }
                return total;
            }
        ''')

    def test_xy16_true_only_is_harmless(self):
        compile_source('''
            fn restore_only() {
                STATUS.XY16 = true;
                asm!("NOP");
            }
        ''')

    def test_two_separate_regions(self):
        compile_source('''
            fn two_regions() {
                STATUS.XY16 = false;
                asm!("LDX #$10");
                STATUS.XY16 = true;
                asm!("NOP");
                STATUS.XY16 = false;
                asm!("LDY #$20");
                STATUS.XY16 = true;
            }
        ''')


class TestXy16RegionViolations:
    """Shapes the validator must reject."""

    def test_unclosed_region_at_function_end(self):
        with pytest.raises(TypeCheckError, match="E-XY16-REGION"):
            compile_source('''
                fn forgot() {
                    STATUS.XY16 = false;
                    asm!("LDX #$10");
                }
            ''')

    def test_return_inside_region(self):
        with pytest.raises(TypeCheckError, match="return.*8-bit index region"):
            compile_source('''
                fn early_return() -> u8 {
                    STATUS.XY16 = false;
                    return 5;
                }
            ''')

    def test_if_inside_region(self):
        with pytest.raises(TypeCheckError, match="control flow.*8-bit index region"):
            compile_source('''
                fn branching(flag: u8) {
                    STATUS.XY16 = false;
                    if flag != 0 {
                        asm!("NOP");
                    }
                    STATUS.XY16 = true;
                }
            ''')

    def test_while_inside_region(self):
        with pytest.raises(TypeCheckError, match="control flow.*8-bit index region"):
            compile_source('''
                fn looping() {
                    STATUS.XY16 = false;
                    while false {
                        asm!("NOP");
                    }
                    STATUS.XY16 = true;
                }
            ''')

    def test_function_call_inside_region(self):
        with pytest.raises(TypeCheckError, match="function call.*8-bit index region"):
            compile_source('''
                fn helper() { asm!("NOP"); }
                fn caller() {
                    STATUS.XY16 = false;
                    helper();
                    STATUS.XY16 = true;
                }
            ''')

    def test_break_inside_region(self):
        with pytest.raises(TypeCheckError, match="break.*8-bit index region|continue.*8-bit index region"):
            compile_source('''
                fn loop_break() {
                    loop {
                        STATUS.XY16 = false;
                        break;
                    }
                }
            ''')
