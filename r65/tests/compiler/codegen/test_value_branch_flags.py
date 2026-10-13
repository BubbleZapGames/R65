# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A bool tested straight from A is compared unless N/Z still reflect A."""

import re

from r65.compiler.main import compile_string


def test_bool_returned_in_a_is_compared_before_branch():
    # After the JSR, N/Z are whatever the callee's epilogue left.
    asm = compile_string("""
#[zeropage] static mut V: u8;
#[zeropage] static mut R: u8;
#[inline(never)]
fn f() -> bool { return V == 3; }
#[entry]
fn main() -> ! {
    let b: bool = f();
    if b { R = 1; }
    loop {}
}
""")
    assert re.search(r"JSR f\n(\s*\.ACCU 8\n)?\s*CMP #\$00\n\s*BEQ", asm), asm
