# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A folded pointer compare (`CompareIndirect`) on the addressing paths the
string tests do not reach: a near pointer on the stack (`CMP (d,S),Y`) and
a far pointer under SET_DBR (`CMP (d,S),Y` with DBR set to its bank). The
far D=S path is strcmp/memcmp in test_string.py.
"""

import re

from r65.compiler.main import compile_string
from r65.tests.e2e import ExpectedState


def _fn_asm(source: str, name: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    return asm.split(f"\n{name}:", 1)[1].split("\nmain:", 1)[0]


NEAR = """
#[lowram] static mut A1: [u8; 8];
#[lowram] static mut B1: [u8; 8];
#[zeropage(0x10)] static mut R: [u8; 3];
fn cmpn(a: *u8, b: *u8) -> u8 {
    let mut i: u16 = 0;
    loop {
        let ca: u8 = a[i];
        let cb: u8 = b[i];
        if ca < cb { return 1; }
        if ca > cb { return 2; }
        if ca == 0 { return 0; }
        i++;
    }
}
#[entry] fn main() {
    A1[0] = 0x41; A1[1] = 0x42; A1[2] = 0;
    B1[0] = 0x41; B1[1] = 0x43; B1[2] = 0;
    R[0] = cmpn(&A1, &B1);
    R[1] = cmpn(&B1, &A1);
    R[2] = cmpn(&A1, &A1);
}
"""

# One far pointer parameter and enough zero-page traffic that the cost
# model picks SET_DBR over D=S (far_ptr_strategy._choose_strategy).
SET_DBR = """
#[ram] static mut A1: [u8; 8];
#[zeropage(0x10)] static mut R: [u8; 2];
#[zeropage(0x20)] static mut K: u8;
#[zeropage(0x21)] static mut Z: [u8; 8];
far fn find(a: far *u8) -> u8 {
    let mut i: u16 = 0;
    let k: u8 = K;
    Z[0] = k; Z[1] = k; Z[2] = k; Z[3] = k; Z[4] = k; Z[5] = k; Z[6] = k; Z[7] = k;
    loop {
        if k == a[i] { return i as u8; }
        if k < a[i] { return 0xFE; }
        i++;
    }
}
#[entry] fn main() {
    A1[0] = 1; A1[1] = 3; A1[2] = 5; A1[3] = 9;
    K = 5; R[0] = find(&A1);
    K = 4; R[1] = find(&A1);
}
"""


def test_near_pointer_on_the_stack(e2e):
    asm = _fn_asm(NEAR, "cmpn")
    assert re.search(r"CMP \(\$[0-9A-F]+,S\),Y", asm)
    result = e2e.run(NEAR, ExpectedState(memory={0x7E0010: [1, 2, 0]}))
    assert result.success, f"Failures: {result.failures} {result.error}"


def test_far_pointer_under_set_dbr(e2e):
    asm = _fn_asm(SET_DBR, "find")
    assert "PLB" in asm and "TCD" not in asm, "expected the SET_DBR strategy"
    assert re.search(r"CMP \(\$[0-9A-F]+,S\),Y", asm)
    result = e2e.run(SET_DBR, ExpectedState(memory={0x7E0010: [2, 0xFE]}))
    assert result.success, f"Failures: {result.failures} {result.error}"
