# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`ptr ± n` steps by n elements, whatever form n takes.

Codegen used to scale only an immediate step, so `w + n` with a run-time n
added n bytes, and `memory_incdec` folded a static `CUR += 1` into one
`INC`. MIR lowering now scales every step, so a pointer BinaryOp always
carries a byte offset.
"""

from r65.tests.e2e import ExpectedState


def test_runtime_and_static_pointer_steps_scale_by_element(e2e):
    result = e2e.run('''
        struct Rec { a: u8, b: u8, c: u16, d: [u8; 6] }

        #[zeropage(0x10)]
        static mut CUR: *Rec;
        #[zeropage(0x12)]
        static mut W: *u16;
        #[zeropage(0x14)]
        static mut N16: u16;
        #[zeropage(0x16)]
        static mut N8: u8;
        #[zeropage(0x20)]
        static mut OUT: [u16; 5];

        #[entry]
        fn main() {
            CUR = 0x0300 as *Rec;
            CUR += 1;                       // static step: +10, not INC
            W = 0x0400 as *u16;
            N16 = 3;
            N8 = 5;
            let w: *u16 = W;
            OUT[0] = (w + N16) as u16;      // run-time u16 step
            OUT[1] = (w + N8) as u16;       // run-time u8 step
            OUT[2] = (N16 + w) as u16;      // integer on the left
            OUT[3] = (w - N8) as u16;
            OUT[4] = ((0x2000 as *u16) + 1) as u16;  // constant fold
        }
    ''', ExpectedState(memory={
        0x10: [0x0A, 0x03],
        0x20: [0x06, 0x04, 0x0A, 0x04, 0x06, 0x04, 0xF6, 0x03, 0x02, 0x20],
    }))
    assert result.success, f"Failures: {result.failures}"


def test_far_word_pointer_runtime_step(e2e):
    result = e2e.run('''
        static TAB: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];

        #[zeropage(0x14)]
        static mut N: u16;
        #[zeropage(0x20)]
        static mut OUT: u16;

        #[entry]
        fn main() {
            N = 2;
            let p: far *u16 = &TAB as far *u16;
            let q: far *u16 = p + N;
            OUT = *q;
        }
    ''', ExpectedState(memory={0x20: [0x33, 0x33]}))
    assert result.success, f"Failures: {result.failures}"
