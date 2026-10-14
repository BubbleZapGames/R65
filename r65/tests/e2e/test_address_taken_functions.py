# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A function whose address is taken keeps the calling convention a pointer
call uses.

Two passes each had their own idea of "address taken", and each missed one:

- far_to_near looked for an `Immediate` holding a name, but MIR spells a
  function address `FunctionPointer`, so it never saw one. `bump`, called
  directly from its own bank, became near; the call through `FP` pushed a
  return bank that RTS left on the stack, so `call_fp` returned to the wrong
  address.
- Scratch parameter promotion did not count a `fn` static in ROM. `put`
  read its arguments from scratch while the call through `HP` pushed them.
"""

import pytest

from r65.tests.e2e import ExpectedState


def test_far_fn_called_directly_and_through_a_pointer(e2e):
    result = e2e.run('''
        #[zeropage(0x10)] static mut O: [u8; 2];
        #[zeropage(0x20)] static mut FP: far fn();

        far fn bump() { O[0] = O[0] + 1; }

        #[inline(never)]
        fn call_fp() { FP(); }

        #[entry]
        fn main() {
            O[0] = 0;
            O[1] = 0;
            bump();
            FP = bump;
            call_fp();
            O[1] = O[1] + 1;
            call_fp();
            O[1] = O[1] + 1;
        }
    ''', ExpectedState(memory={0x7E0010: [3, 2]}))
    assert result.success, f"Failures: {result.failures}"


@pytest.mark.requires_stack_params
def test_fn_in_a_rom_static_keeps_stack_params(e2e):
    result = e2e.run('''
        #[zeropage(0x20, register)] static mut S0: u8;
        #[zeropage(0x21, register)] static mut S1: u8;
        #[zeropage(0x10)] static mut O: [u8; 2];

        fn put(v: u8, w: u8) { O[0] = v; O[1] = w; }

        static HP: fn(u8, u8) = put;

        #[inline(never)]
        fn call_hp() { HP(7, 8); }

        #[entry]
        fn main() { put(3, 4); call_hp(); }
    ''', ExpectedState(memory={0x7E0010: [7, 8]}))
    assert result.success, f"Failures: {result.failures}"
