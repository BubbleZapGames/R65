# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Inlining under Pascal binds a bound parameter's argument where the callee
reads it: on the stack.

Pascal passes every parameter on the stack whatever it binds, and builds the
callee to read it there. The inliner went by the binding instead - loading X
or nothing at all - so the inlined body read a frame slot nothing wrote: a
`#[soa]` method's index `self`, or any `@ A`, `@ X` or `@ VAR` parameter.
"""

from r65.tests.e2e import E2ETest, ExpectedState


def run(source, expected):
    result = E2ETest(abi="Pascal").run(source, ExpectedState(memory=expected),
                                       extra_args=["-O1"])
    assert result.success, f"{result.error} {result.failures}"


def test_soa_index_self():
    run('''
        #[soa]
        struct Enemy { kind: u8, hp: u16 }
        impl Enemy { fn hurt(self, d: u16) { self.hp = self.hp - d; } }
        #[lowram(0x0301)] static mut POOL: [Enemy; 4];

        #[entry]
        fn main() {
            for i in 0..4 { POOL[i].hp = 100; }
            for i in 0..4 { POOL[i].hurt(i as u16 + 1); }
        }
    ''', {0x7E0305: [99, 0, 98, 0, 97, 0, 96, 0]})


def test_register_and_variable_bound_params():
    run('''
        #[lowram(0x0300)] static mut OUT: [u16; 3];
        #[lowram(0x0310)] static mut T: u8;
        #[inline] fn add(a @ A: u8, x @ X: u16) -> u16 { return (a as u16) + x; }
        #[inline] fn viat(v @ T: u8) -> u8 { return v + 1; }

        #[entry]
        fn main() {
            for i in 0..2 { OUT[i] = add((i as u8) + 3, 0x100); }
            OUT[2] = viat(9) as u16;
        }
    ''', {0x7E0300: [0x03, 0x01, 0x04, 0x01, 10, 0]})
