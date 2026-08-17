# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Q10's `to_string`, and its use from `format!("{s}", ...)`.

Q10 is a newtype, so it cannot `impl ToString` — trait dispatch stores a TypeId
byte at offset 0, which would overlap the value. `format!` resolves `to_string`
by name rather than through a vtable, so an inherent method serves the same role
and is dispatched statically.

Two decimal places, so output is rounded: the scale is 1/64 = 0.015625.
"""

from pathlib import Path as _Path

import pytest
from r65.tests.e2e import ExpectedState

_STDLIB = _Path(__file__).parent.parent.parent.parent / "stdlib"
# string.r65 provides format!; it needs mod8 from math.r65, which needs sneslib.
PRELUDE = (f'include!("{_STDLIB / "sneslib.r65"}")\n'
           f'include!("{_STDLIB / "math.r65"}")\n'
           f'include!("{_STDLIB / "string.r65"}")\n'
           f'include!("{_STDLIB / "Q10.r65"}")\n')

BUF = 0x7E2000


def read_n(cpu, n: int, addr=BUF) -> str:
    """Read exactly `n` bytes. `to_string` writes no terminator, so the returned
    count is the only thing that says where the string ends."""
    return ''.join(chr(cpu.memory.read(addr + i)) for i in range(n))


def read_str(cpu, addr=BUF, limit=16) -> str:
    """Read up to a NUL — valid for `format!` output, which does terminate."""
    out = []
    for i in range(limit):
        b = cpu.memory.read(addr + i)
        if b == 0:
            break
        out.append(chr(b))
    return ''.join(out)


def program(body: str) -> str:
    return (PRELUDE
            + "#[ram]\nstatic mut BUF: [u8; 16];\n"
            + "#[zeropage(0x30)]\nstatic mut LEN: u8;\n"
            + "#[entry]\nfn main() { " + body + " }")


class TestToString:
    """Direct calls, covering sign, both digit counts, and rounding."""

    @pytest.mark.parametrize("expr,expected", [
        ("Q10::from_int(0)",     "0.00"),
        ("Q10::from_int(1)",     "1.00"),
        ("Q10::from_int(100)",   "100.00"),
        ("Q10::from_int(511)",   "511.00"),
        ("Q10::from(12, 48)",    "12.75"),
        ("Q10::from(0, 32)",     "0.50"),
        ("Q10::from(0, 1)",      "0.02"),      # 1/64 = 0.015625, rounded
        ("Q10::from(0, 63)",     "0.98"),      # 63/64 = 0.984375, rounded
        ("Q10::from_int(0 - 1)", "-1.00"),
        ("Q10::from_int(0 - 100)", "-100.00"),
        ("Q10::from(0 - 12, 0)", "-12.00"),
    ])
    def test_formats(self, e2e, expr, expected):
        src = program(f"let q: Q10 = {expr};"
                      " LEN = q.to_string(&BUF as far *u8) as u8;")
        r = e2e.run(src, ExpectedState(memory={0x7E0030: len(expected)}))
        assert r.success, f"Failures: {r.failures}"
        assert read_n(r.cpu, len(expected)) == expected

    def test_writes_nothing_past_the_returned_count(self, e2e):
        """No terminator: `format!` adds one for the whole string, so a
        per-fragment NUL would be a store one byte beyond what was asked for.
        The buffer is pre-filled so an extra write would show."""
        src = program("let mut i: u16 = 0;"
                      " while i < 16 { BUF[i] = 0xFF; i = i + 1; }"
                      " let q: Q10 = Q10::from(12, 48);"
                      " LEN = q.to_string(&BUF as far *u8) as u8;")
        r = e2e.run(src, ExpectedState(memory={0x7E0030: 5}))
        assert r.success, f"Failures: {r.failures}"
        assert read_n(r.cpu, 5) == "12.75"
        assert r.cpu.memory.read(BUF + 5) == 0xFF, (
            "to_string must not write past the count it returns")


class TestFormatMacro:
    """`format!`'s {s} resolves `to_string` by name, so a newtype works even
    though it cannot implement the trait."""

    def test_format_s(self, e2e):
        src = program('let q: Q10 = Q10::from(3, 16); format!(BUF, "{s}", q);')
        r = e2e.run(src, ExpectedState(memory={}))
        assert r.success, f"Failures: {r.failures}"
        assert read_str(r.cpu) == "3.25"

    def test_format_with_surrounding_text(self, e2e):
        """`Q10::from(-2, 32)` is -1.5, not -2.5: `from` ORs the fraction into
        `n << 6`, and on a negative that moves the value toward zero
        (-128 | 32 == -96). Asserting the arithmetic value keeps this test
        honest about what `from` builds."""
        src = program('let q: Q10 = Q10::from(0 - 2, 32);'
                      ' format!(BUF, "v={s}!", q);')
        r = e2e.run(src, ExpectedState(memory={}))
        assert r.success, f"Failures: {r.failures}"
        assert read_str(r.cpu) == "v=-1.50!"
