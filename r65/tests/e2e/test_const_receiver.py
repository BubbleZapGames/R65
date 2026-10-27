# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A method only ever called on one instance takes its address as a constant.

`optimize/const_args.py` folds a stack argument every call passes the same
constant into the callee, and an access through the now-constant `self`
becomes a load or store of the static itself: no push at the call, no
`(sr,S),Y` in the body.
"""

import re

from r65.compiler.main import compile_string
from r65.tests.e2e import ExpectedState


def _fn_asm(source: str, name: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    body = asm.split(f"\n{name}:", 1)[1]
    return re.split(r"\n(?!\s)(?!" + re.escape(name) + r"__)\w+:", body, maxsplit=1)[0]


ONE_INSTANCE = '''
    struct Player { x: u8, hp: u16 }
    #[lowram(0x0300)] static mut P: Player;

    impl Player {
        #[inline(never)]
        fn heal(*self, n: u8) { self.x = self.x + n; self.hp = self.hp + 1; }
        #[inline(never)]
        fn tick(*self) { self.heal(1); }
    }

    #[entry]
    fn main() {
        P.x = 0; P.hp = 0x00FF;
        P.heal(3);
        P.heal(4);
        P.tick();
    }
'''


def test_single_instance_method_addresses_the_static(e2e):
    heal = _fn_asm(ONE_INSTANCE, "Player__heal")
    assert ",S),Y" not in heal and "$0300" in heal, heal
    # tick hands its constant self on, so heal's call from it folds too.
    tick = _fn_asm(ONE_INSTANCE, "Player__tick")
    assert "#$0300" not in tick, tick
    main = _fn_asm(ONE_INSTANCE, "main")
    assert "#$0300" not in main, main
    result = e2e.run(ONE_INSTANCE, ExpectedState(memory={0x7E0300: [8, 0x02, 0x01]}))
    assert result.success, f"Failures: {result.failures}"


FAR_SELF = '''
    struct Player { x: u8, hp: u16 }
    static mut R: Player;
    #[zeropage(0x10)] static mut OUT: [u8; 3];

    impl Player {
        #[inline(never)]
        fn hurt(far *self) { self.x = self.x - 1; self.hp = self.hp - 1; }
    }

    #[entry]
    fn main() {
        R.x = 9; R.hp = 0x0100;
        R.hurt();
        R.hurt();
        OUT[0] = R.x;
        OUT[1] = R.hp as u8;
        OUT[2] = (R.hp >> 8) as u8;
    }
'''


def test_far_self_on_a_ram_static_addresses_it_long(e2e):
    hurt = _fn_asm(FAR_SELF, "Player__hurt")
    assert "[" not in hurt and "PLB" not in hurt and "$7E" in hurt, hurt
    result = e2e.run(FAR_SELF, ExpectedState(memory={0x7E0010: [7, 0xFE, 0x00]}))
    assert result.success, f"Failures: {result.failures}"


TWO_INSTANCES = '''
    struct Player { x: u8, hp: u16 }
    #[lowram(0x0300)] static mut P1: Player;
    #[lowram(0x0303)] static mut P2: Player;

    impl Player {
        #[inline(never)]
        fn heal(*self, n: u8) { self.x = self.x + n; self.hp = self.hp + 1; }
    }

    #[entry]
    fn main() {
        P1.x = 0; P1.hp = 0; P2.x = 10; P2.hp = 10;
        P1.heal(3);
        P2.heal(3);
    }
'''


def test_two_instances_keep_the_pointer(e2e):
    heal = _fn_asm(TWO_INSTANCES, "Player__heal")
    assert "),Y" in heal, heal
    result = e2e.run(TWO_INSTANCES, ExpectedState(memory={0x7E0300: [3, 1, 0, 13, 11, 0]}))
    assert result.success, f"Failures: {result.failures}"


ADDRESS_TAKEN = '''
    #[lowram(0x0300)] static mut V: u8;

    #[inline(never)]
    fn bump(p: *u8) { *p = *p + 1; }

    #[entry]
    fn main() {
        V = 5;
        bump(&V as *u8);
        let f: fn(*u8) = bump;
        f(&V as *u8);
    }
'''


def test_a_function_reached_by_pointer_keeps_its_parameter(e2e):
    bump = _fn_asm(ADDRESS_TAKEN, "bump")
    assert "$0300" not in bump, bump
    result = e2e.run(ADDRESS_TAKEN, ExpectedState(memory={0x7E0300: [7]}))
    assert result.success, f"Failures: {result.failures}"
