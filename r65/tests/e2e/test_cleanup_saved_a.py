# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""An epilogue that pushes the returned A keeps it on top past the SP adjust.

With X and Y restored by `#[preserves]`, the return value is parked with
PHA. The SP adjust that frees the frame (or the Pascal parameters) also
freed those bytes, so the PLA after it read the byte beyond and the return
went one byte off.
"""

from r65.tests.e2e import E2ETest, ExpectedState

LOOPY = '''
#[lowram(0x0300)] static mut OUT: [u16; 2];
{callee}
#[bank(0)]
#[inline(never)]
fn loopy(n: u16) -> u16 {{
    let mut s: u16 = 0;
    let mut i: u16 = 0;
    while i < n {{ s = s + (keep(i as u8, i) as u16); i = i + 1; }}
    return s;
}}
#[entry]
fn main() {{ OUT[0] = loopy(5); OUT[1] = loopy(3); }}
'''

KEEP = '''#[inline(never)]
#[preserves(X, Y)]
fn keep(v: u8, w: u16) -> u8 { return v + (w as u8); }'''


def run(source, expected, abi=None):
    result = E2ETest(abi=abi).run(source, ExpectedState(memory={0x7E0300: expected}))
    assert result.success, f"{result.error} {result.failures}"


def test_frame_freed_under_a_u16_return():
    run('''
        #[lowram(0x0300)] static mut OUT: [u16; 3];
        #[inline(never)] fn h(v: u16) -> u16 { return v + 1; }
        #[inline(never)]
        #[preserves(X, Y)]
        fn many(v @ A: u16) -> u16 {
            let a: u16 = h(v);
            let b: u16 = h(a);
            let c: u16 = h(b);
            let d: u16 = h(c);
            let e: u16 = h(d);
            return a + b + c + d + e + v;
        }

        #[entry]
        fn main() { OUT[0] = many(0x100); OUT[1] = many(0x10); OUT[2] = 0xBEEF; }
    ''', [0x0F, 0x06, 0x6F, 0x00, 0xEF, 0xBE])


def test_pascal_near_parameters_freed():
    run(LOOPY.format(callee=KEEP), [20, 0, 6, 0], abi="Pascal")


def test_pascal_far_parameters_freed():
    run(LOOPY.format(callee="#[bank(1)]\n" + KEEP), [20, 0, 6, 0], abi="Pascal")
