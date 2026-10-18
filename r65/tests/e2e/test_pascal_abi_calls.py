# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Programs pinned to the Pascal ABI, whatever --abi says.

A call in a loop failed to compile: the callee removes its parameters, which
the caller's stack depth never saw. The result space was pushed a PHA per
byte, two bytes each in m16; and a value in A for a later parameter was
overwritten by the pushes before it, left to right.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def run(source: str, expected: ExpectedState):
    result = E2ETest(abi="Pascal").run(source, expected)
    assert result.success, f"Failures: {result.failures} {result.error or ''}"


class TestPascalCalls:
    def test_values(self):
        run('''
            #[lowram(0x0300)] static mut LOG: [u16; 16];
            #[lowram(0x0340)] static mut N: u16;
            #[inline(never)] fn note(v: u16) { LOG[N] = v; N = N + 1; X = 0x7777; Y = 0x7777; }
            #[inline(never)] fn pair(a: u16, b: u16) { note(a); note(b); }
            #[inline(never)] fn twice(v: u16) -> u16 { return v + v; }

            #[inline(never)] fn go() {
                let mut i: u16 = 0;
                while i < 2 { pair(i, i + 0x10); i = i + 1; }
                let mut j: u16 = 0;
                while j < 2 { pair(j + 0x20, j); j = j + 1; }
                note(twice(0x0123));
            }

            #[entry]
            fn main() { N = 0; go(); }
        ''', ExpectedState(memory={
            0x0300: [0x00, 0x00, 0x10, 0x00, 0x01, 0x00, 0x11, 0x00,
                     0x20, 0x00, 0x00, 0x00, 0x21, 0x00, 0x01, 0x00,
                     0x46, 0x02],
        }))

    def test_result_stored_to_a_local(self):
        """The result is pulled off the stack and stored to a frame slot:
        the callee has already removed its parameters, so the slot is where
        it was before the call."""
        run('''
            #[lowram(0x0300)] static mut LOG: [u16; 4];
            #[lowram(0x0340)] static mut N: u16;
            #[inline(never)] fn note(v: u16) { LOG[N] = v; N = N + 1; }
            #[inline(never)] fn twice(v: u16) -> u16 { return v + v; }
            #[inline(never)] fn go() {
                let a: u16 = twice(3);
                let b: u16 = twice(5);
                note(a); note(b);
            }
            #[entry]
            fn main() { N = 0; go(); }
        ''', ExpectedState(memory={0x0300: [6, 0, 10, 0]}))
