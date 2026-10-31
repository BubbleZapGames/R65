# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A pointer bound to X or Y, and a constant pointer, as the base of an access.

`let p @ X: *u8` and `let c @ Y: *Obj` reached codegen as a HardwareRegister
pointer, and a constant pointer as an Immediate, which _const_pointer read as
a vreg and crashed on. A pointer in X or Y is now addressed `abs,X` /
`abs,Y` at a constant offset, and a constant one absolutely.
"""

from r65.tests.e2e import ExpectedState


def test_y_bound_pointer(e2e):
    result = e2e.run('''
        struct Obj { kind: u8, x: u16 }

        #[lowram(0x0310)]
        static mut POOL: [Obj; 2];
        #[zeropage(0x20)]
        static mut OUT: u16;

        #[inline(never)]
        fn second_x() {
            let c @ Y: *Obj = &POOL[1] as *Obj;
            OUT = c.x;
        }

        #[entry]
        fn main() {
            POOL[1].x = 0x1234;
            second_x();
        }
    ''', ExpectedState(memory={0x20: [0x34, 0x12]}))
    assert result.success, f"Failures: {result.failures}"


def test_constant_pointers(e2e):
    result = e2e.run('''
        struct Obj { kind: u8, x: u16 }

        #[zeropage(0x20)]
        static mut OUT: u16;

        #[entry]
        fn main() {
            *(0x0300 as *u8) = 5;
            (0x0300 as *Obj).x = 0x0607;
            OUT = *(0x0301 as *u16);
        }
    ''', ExpectedState(memory={0x0300: [5, 0x07, 0x06], 0x20: [0x07, 0x06]}))
    assert result.success, f"Failures: {result.failures}"
