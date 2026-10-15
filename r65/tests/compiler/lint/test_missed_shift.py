# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""L004 missed_shift: a signed `/` is not a shift.

Signed `/` rounds toward zero and `>>` rounds down (`-7 / 2` is -3,
`-7 >> 1` is -4), so suggesting the shift would change the result.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint


def _l004(body: str):
    program = expand_macros(parse('''
        #[zeropage(0x40)] static mut u: u8;
        #[zeropage(0x41)] static mut s: i8;
        #[entry] fn main() { ''' + body + ''' }
    ''', '<test>'))
    hir = HIRBuilder().build_program(program)
    TypeChecker(hir).check()
    diags = run_lint(hir, LintConfig(enabled_codes={"L004"}))
    return [d.message for d in diags.diagnostics if d.code == "L004"]


def test_signed_division_is_not_flagged():
    assert _l004("s = s / 2;") == []


def test_unsigned_division_and_signed_multiply_are_flagged():
    assert _l004("u = u / 2;") == ["`/ 2` could be written as `>> 1`"]
    assert _l004("s = s * 2;") == ["`* 2` could be written as `<< 1`"]
