# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
A far call through a function pointer held in a low-RAM or direct-page static
reads the pointer in place: `optimize/memory_operand.py` folds the load into
the call, and codegen emits PHK / PEA / JML [abs] (`JML [abs]` reads bank 0)
instead of copying the pointer out and running the RTL trampoline.

The callee is entered in m8, as through the trampoline: a call made while A
was 16-bit from pushing a u16 argument once ran the callee in m16.
"""

import re

from r65.compiler.main import compile_string
from r65.tests.e2e import E2ETest, ExpectedState

TRAMPOLINE = "Indirect far call via trampoline"


def _fn_asm(source: str, name: str) -> str:
    asm = compile_string(source, cfg_options=["snes"])
    body = asm.split(f"\n{name}:", 1)[1]
    return re.split(r"\n(?!\s)(?!" + re.escape(name) + r"__)\w+:", body, maxsplit=1)[0]


def _run(source: str, memory: dict):
    # Default ABI whatever --abi says: FixedStack refuses an address-taken
    # function with stack parameters.
    result = E2ETest(abi="Default").run(source, ExpectedState(memory=memory))
    assert result.success, f"Failures: {result.failures}"


GLOBAL = '''
    #[lowram(0x0300)] static mut OUT: [u8; 4];
    #[lowram(0x0310)] static mut FP: fn(u16) -> bool;
    #[inline(never)] fn h(t: u16) -> bool { OUT[0] = 0x5A; OUT[1] = t as u8; return true; }
    #[inline(never)] fn go() { if FP(0x1234) { OUT[2] = 0xC3; } }
    #[entry] fn main() { FP = h; go(); OUT[3] = 0xEE; }
'''


def test_static_fn_pointer_is_called_in_place():
    go = _fn_asm(GLOBAL, "go")
    assert "JML [$0310]" in go and TRAMPOLINE not in go, go
    _run(GLOBAL, {0x0300: [0x5A, 0x34, 0xC3, 0xEE]})


# nscene's shape: the pointer is a field of a single instance, so the
# constant-receiver fold has already made `self.f` a load of the static.
FIELD = '''
    type Tr = fn(u16) -> bool;
    struct C { a: u8, f: Tr }
    #[lowram(0x0320)] static mut CC: C;
    #[lowram(0x0300)] static mut OUT: [u8; 3];
    #[inline(never)] fn h(t: u16) -> bool { OUT[0] = 0x5A; OUT[1] = t as u8; return false; }
    impl C {
        #[inline(never)]
        fn run(*self, t: u16) { let f: Tr = self.f; if !f(t) { OUT[2] = 0xC3; } }
    }
    #[entry] fn main() { CC.f = h; CC.run(0x0077); }
'''


def test_field_of_a_single_instance_is_called_in_place():
    run = _fn_asm(FIELD, "C__run")
    assert "JML [$0321]" in run and TRAMPOLINE not in run, run
    _run(FIELD, {0x0300: [0x5A, 0x77, 0xC3]})


def test_ram_static_keeps_the_trampoline():
    # A #[ram] static sits in bank $7E, out of JML [abs]'s reach.
    source = GLOBAL.replace("#[lowram(0x0310)] static mut FP", "#[ram] static mut FP")
    go = _fn_asm(source, "go")
    assert "JML [" not in go and TRAMPOLINE in go, go
    _run(source, {0x0300: [0x5A, 0x34, 0xC3, 0xEE]})


def test_store_between_load_and_call_keeps_the_copy():
    # The call must use the pointer as loaded, not as rewritten after.
    source = '''
        #[lowram(0x0300)] static mut OUT: [u8; 2];
        #[lowram(0x0310)] static mut FP: fn(u16) -> bool;
        #[inline(never)] fn h(t: u16) -> bool { OUT[0] = 0x11; return true; }
        #[inline(never)] fn k(t: u16) -> bool { OUT[1] = 0x22; return true; }
        #[inline(never)] fn go() { let f: fn(u16) -> bool = FP; FP = k; f(0); }
        #[entry] fn main() { OUT[0] = 0; OUT[1] = 0; FP = h; go(); }
    '''
    go = _fn_asm(source, "go")
    assert "JML [$0310]" not in go, go
    _run(source, {0x0300: [0x11, 0x00]})
