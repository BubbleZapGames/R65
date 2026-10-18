# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end values for calls through a function pointer with stack arguments.

The RTS/RTL trampoline read the pointer's slot one byte off per argument byte
(the pushes were counted twice), and its pushes - taken back by its own RTS
and the callee's - left the emitter's stack depth wrong, which failed a loop
at compile time. An untyped constant argument was also pushed at its own width
rather than the parameter's, and `FP = g` after a 16-bit store kept only g's
low byte.
"""

from r65.tests.e2e import E2ETest, ExpectedState


class TestFnPointerStackArgs:
    def test_values(self):
        # Default ABI whatever --abi says: FixedStack refuses an address-taken
        # function with stack parameters, and these are what is under test.
        result = E2ETest(abi="Default").run('''
            #[lowram(0x0300)] static mut OUT: [u16; 8];
            #[lowram(0x0320)] static mut N: u16;
            #[inline(never)] fn g(v: u16) { OUT[N] = v; N = N + 1; }
            #[inline(never)] far fn h(v: u16) { OUT[N] = v | 0x8000; N = N + 1; }
            #[lowram(0x0330)] static mut FP: fn(u16);
            #[lowram(0x0333)] static mut FFP: far fn(u16);

            #[inline(never)] fn go() {
                let mut i: u16 = 0;
                while i < 2 { FP(i); FFP(i); i = i + 1; }
                FP(1);
                FP(0x1234);
            }

            #[entry]
            fn main() { N = 0; FP = g; FFP = h; go(); OUT[7] = 0xBEEF; }
        ''', ExpectedState(memory={
            0x0300: [0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x01, 0x80,
                     0x01, 0x00, 0x34, 0x12, 0x00, 0x00, 0xEF, 0xBE],
        }))
        assert result.success, f"Failures: {result.failures}"
