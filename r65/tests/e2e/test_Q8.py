# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for the Q8.8 fixed-point library.

`Q8` is a newtype over `u16`, so these check two things at once: that the
fixed-point arithmetic is right, and that wrapping it in a distinct type costs
nothing — values still live in two bytes and ride in registers.

Unsigned, unlike Q10. That removes the sign handling from `mul` but adds three
hazards these tests target: `b - a` wraps rather than going negative (see
`lerp`), a 1/256 fraction rounds up to a whole unit that the integer part has no
room for (see `to_string` and `round`), and a full-u16 divisor needs a 17-bit
remainder (see `div`).
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
Q8_PATH = STDLIB_DIR / "Q8.r65"

PRELUDE = f'''
            include!("{SNESLIB_PATH}")
            include!("{Q8_PATH}")
'''


def program(body: str, statics: str = "#[zeropage(0x10)]\nstatic mut OUT: Q8;") -> str:
    return f"{PRELUDE}\n{statics}\n\n#[entry]\nfn main() {{\n{body}\n}}"


def raw(v: int):
    """Little-endian bytes of a Q8's raw u16 payload."""
    v &= 0xFFFF
    return [v & 0xFF, v >> 8]


class TestQ8Storage:
    """A Q8 is two bytes, like the u16 it wraps."""

    def test_static_variable(self, e2e):
        r = e2e.run(program("OUT = Q8(0x1234);"),
                    ExpectedState(memory={0x7E0010: raw(0x1234)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_top_of_range(self, e2e):
        """255.996 — the largest Q8. A signed payload could not hold it."""
        r = e2e.run(program("OUT = Q8(65535);"),
                    ExpectedState(memory={0x7E0010: raw(65535)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_constant(self, e2e):
        r = e2e.run(program("OUT = Q8_ONE;"),
                    ExpectedState(memory={0x7E0010: raw(256)}))
        assert r.success, f"{r.error} {r.failures}"


class TestQ8Conversions:
    """from_int / from / to_int / to_frac."""

    def test_from_int(self, e2e):
        r = e2e.run(program("OUT = Q8::from_int(100);"),
                    ExpectedState(memory={0x7E0010: raw(25600)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_from_int_at_the_top(self, e2e):
        """255 is the largest integer part; 255 << 8 is 65280."""
        r = e2e.run(program("OUT = Q8::from_int(255);"),
                    ExpectedState(memory={0x7E0010: raw(65280)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_from_whole_and_fraction(self, e2e):
        """12 + 192/256 = 12.75 -> 3264."""
        r = e2e.run(program("OUT = Q8::from(12, 192);"),
                    ExpectedState(memory={0x7E0010: raw(3264)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_fraction_at_the_top_of_its_range(self, e2e):
        r = e2e.run(program("OUT = Q8::from(0, 255);"),
                    ExpectedState(memory={0x7E0010: raw(255)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_both_fields_are_u8_so_neither_can_overflow(self, e2e):
        """Unlike Q10's `from`, both parameters are exactly their field width, so
        there is nothing to mask off."""
        r = e2e.run(program("OUT = Q8::from(255, 255);"),
                    ExpectedState(memory={0x7E0010: raw(65535)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_to_int(self, e2e):
        r = e2e.run(program("let q: Q8 = Q8::from(100, 128);\n"
                            "OUT2 = q.to_int();",
                            statics=("#[zeropage(0x10)]\nstatic mut OUT: Q8;\n"
                                     "#[zeropage(0x14)]\nstatic mut OUT2: u8;")),
                    ExpectedState(memory={0x7E0014: 100}))
        assert r.success, f"{r.error} {r.failures}"

    def test_to_frac(self, e2e):
        r = e2e.run(program("let q: Q8 = Q8::from(100, 128);\n"
                            "OUT2 = q.to_frac();",
                            statics=("#[zeropage(0x10)]\nstatic mut OUT: Q8;\n"
                                     "#[zeropage(0x14)]\nstatic mut OUT2: u8;")),
                    ExpectedState(memory={0x7E0014: 128}))
        assert r.success, f"{r.error} {r.failures}"


class TestQ8Round:
    """Ties round up, and the top of the range saturates rather than wrapping.

    No sign correction is needed — the payload is unsigned.
    """

    def _round(self, e2e, rawv, expected):
        r = e2e.run(program(f"let q: Q8 = Q8({rawv});\nOUT2 = q.round();",
                            statics=("#[zeropage(0x10)]\nstatic mut OUT: Q8;\n"
                                     "#[zeropage(0x14)]\nstatic mut OUT2: u8;")),
                    ExpectedState(memory={0x7E0014: expected}))
        assert r.success, f"round({rawv}): {r.error} {r.failures}"

    def test_below_half_rounds_down(self, e2e):
        self._round(e2e, 256 + 127, 1)

    def test_exactly_half_rounds_up(self, e2e):
        self._round(e2e, 256 + 128, 2)

    def test_whole_is_unchanged(self, e2e):
        self._round(e2e, 256, 1)

    def test_differs_from_to_int(self, e2e):
        """1.75 rounds to 2 but truncates to 1."""
        self._round(e2e, 256 + 192, 2)

    def test_last_value_that_rounds_normally(self, e2e):
        """255.496 — the largest raw whose `+ 128` bias does not overflow."""
        self._round(e2e, 65407, 255)

    def test_saturates_rather_than_wrapping(self, e2e):
        """255.5 would round to 256. Unguarded, `65408 + 128` wraps to 127 and
        the shift yields 0 — the nastiest possible answer for the largest input.
        """
        self._round(e2e, 65408, 255)

    def test_top_of_range_saturates(self, e2e):
        self._round(e2e, 65535, 255)


class TestQ8InheritedOperators:
    """Add, subtract and compare come from u16 and stay Q8-typed."""

    def test_addition(self, e2e):
        r = e2e.run(program("OUT = Q8(256) + Q8(128);"),
                    ExpectedState(memory={0x7E0010: raw(384)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_subtraction(self, e2e):
        r = e2e.run(program("OUT = Q8(384) - Q8(128);"),
                    ExpectedState(memory={0x7E0010: raw(256)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_comparison_is_unsigned(self, e2e):
        """32768 is negative as an i16 but must compare above 256 here."""
        r = e2e.run(program("if Q8(32768) > Q8(256) { OUT = Q8(1); }\n"
                            "else { OUT = Q8(0); }"),
                    ExpectedState(memory={0x7E0010: raw(1)}))
        assert r.success, f"{r.error} {r.failures}"


class TestQ8Mul:
    """Scaling multiply via the SNES hardware 8x8 unit: (self * other) >> 8.

    `a.mul(b)`, not `a * b` — Q8 inherits `*` from u16, which multiplies the raw
    values and lands 256x too high.
    """

    def _mul(self, e2e, a, b, expected):
        r = e2e.run(program(f"let x: Q8 = Q8({a});\nlet y: Q8 = Q8({b});\n"
                            "OUT = x.mul(y);"),
                    ExpectedState(memory={0x7E0010: raw(expected)}))
        assert r.success, f"Q8({a}).mul(Q8({b})): {r.error} {r.failures}"

    def test_one_times_one(self, e2e):
        self._mul(e2e, 256, 256, 256)            # 1.0 * 1.0 = 1.0

    def test_two_times_a_half(self, e2e):
        self._mul(e2e, 512, 128, 256)            # 2.0 * 0.5 = 1.0

    def test_half_times_half(self, e2e):
        self._mul(e2e, 128, 128, 64)             # 0.5 * 0.5 = 0.25

    def test_by_zero(self, e2e):
        self._mul(e2e, 2560, 0, 0)

    def test_crosses_the_byte_boundary(self, e2e):
        """10.0 * 3.0 = 30.0; 2560 * 768 needs the high partial products."""
        self._mul(e2e, 2560, 768, 7680)

    def test_large_operands(self, e2e):
        """200.0 * 1.25 = 250.0 — near the top of the range."""
        self._mul(e2e, 51200, 320, 64000)

    def test_underflows_toward_zero(self, e2e):
        """0.5 * 1/256 is below the scale, so it truncates to zero."""
        self._mul(e2e, 128, 1, 0)


class TestQ8Div:
    """Scaling divide: (self << 8) / other, by software long division.

    Needs no `snes` cfg. The remainder needs 17 bits here — a Q8 divisor may be
    the full 65535 — so the 17th bit is carried separately; Q10 escapes that,
    its divisor being an i16 magnitude.
    """

    def _div(self, e2e, a, b, expected):
        r = e2e.run(program(f"let x: Q8 = Q8({a});\nlet y: Q8 = Q8({b});\n"
                            "OUT = x.div(y);"),
                    ExpectedState(memory={0x7E0010: raw(expected)}))
        assert r.success, f"Q8({a}).div(Q8({b})): {r.error} {r.failures}"

    def test_identity(self, e2e):
        self._div(e2e, 768, 768, 256)             # 3.0 / 3.0 = 1.0

    def test_by_one(self, e2e):
        self._div(e2e, 768, 256, 768)             # 3.0 / 1.0 = 3.0

    def test_by_a_fraction_grows(self, e2e):
        self._div(e2e, 512, 128, 1024)            # 2.0 / 0.5 = 4.0

    def test_yields_a_fraction(self, e2e):
        self._div(e2e, 256, 512, 128)             # 1.0 / 2.0 = 0.5

    def test_three_quarters(self, e2e):
        self._div(e2e, 768, 1024, 192)            # 3.0 / 4.0 = 0.75

    def test_larger_operands(self, e2e):
        self._div(e2e, 3072, 768, 1024)           # 12.0 / 3.0 = 4.0

    def test_zero_numerator(self, e2e):
        self._div(e2e, 0, 768, 0)

    def test_truncates(self, e2e):
        """1.0 / 3.0 is 85.33/256, truncated to 85."""
        self._div(e2e, 256, 768, 85)

    def test_huge_divisor_exercises_the_carry(self, e2e):
        """A divisor above 32767 is what a 16-bit remainder cannot hold. 1.0 /
        255.996 is 1.0039/256, truncated to 1."""
        self._div(e2e, 256, 65535, 1)

    def test_max_over_max_is_one(self, e2e):
        self._div(e2e, 65535, 65535, 256)

    def test_by_zero_saturates(self, e2e):
        self._div(e2e, 768, 0, 65535)

    def test_runtime_values_not_just_literals(self, e2e):
        """Through statics, so nothing const-folds."""
        r = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q8;
            #[zeropage(0x20)] static mut AV: Q8;
            #[zeropage(0x22)] static mut BV: Q8;
            #[entry]
            fn main() {{
                AV = Q8(7680);
                BV = Q8(384);
                OUT = AV.div(BV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(5120)}))   # 30.0 / 1.5 = 20.0
        assert r.success, f"{r.error} {r.failures}"


class TestQ8DivU8:
    """Divide by an integer count, via the SNES hardware divider.

    Dividing a fixed-point value by an integer preserves the scale, so this is a
    straight 16/8 with no shifting — one hardware operation against the 24
    software steps `div` needs.
    """

    def _div(self, e2e, a, d, expected):
        r = e2e.run(program(f"let x: Q8 = Q8({a});\nOUT = x.div_u8({d});"),
                    ExpectedState(memory={0x7E0010: raw(expected)}))
        assert r.success, f"Q8({a}).div_u8({d}): {r.error} {r.failures}"

    def test_by_one_is_identity(self, e2e):
        self._div(e2e, 3072, 1, 3072)

    def test_exact(self, e2e):
        self._div(e2e, 3072, 3, 1024)             # 12.0 / 3 = 4.0

    def test_halves_a_fraction(self, e2e):
        self._div(e2e, 256, 2, 128)               # 1.0 / 2 = 0.5

    def test_truncates(self, e2e):
        """1.0 / 3 is 85.33/256, truncated to 85."""
        self._div(e2e, 256, 3, 85)

    def test_zero_numerator(self, e2e):
        self._div(e2e, 0, 7, 0)

    def test_largest_dividend(self, e2e):
        """65535 / 255 = 257 — the dividend is 16-bit even though d is not."""
        self._div(e2e, 65535, 255, 257)

    def test_by_zero_saturates(self, e2e):
        self._div(e2e, 3072, 0, 65535)

    def test_agrees_with_div(self, e2e):
        """The two divides must agree for a whole-number divisor — one through
        hardware, the other through 24 software steps."""
        r = e2e.run(program("let a: Q8 = Q8(3072);\n"
                            "OUT = a.div_u8(3);\n"
                            "OUT2 = a.div(Q8::from_int(3));",
                            statics=("#[zeropage(0x10)]\nstatic mut OUT: Q8;\n"
                                     "#[zeropage(0x12)]\nstatic mut OUT2: Q8;")),
                    ExpectedState(memory={0x7E0010: raw(1024),
                                          0x7E0012: raw(1024)}))
        assert r.success, f"{r.error} {r.failures}"


class TestQ8Lerp:
    """Interpolation. Both directions are written out because the payload is
    unsigned: `b - a` would wrap when b < a."""

    def _lerp(self, e2e, a, b, t, expected):
        r = e2e.run(program(f"OUT = Q8::lerp(Q8({a}), Q8({b}), Q8({t}));"),
                    ExpectedState(memory={0x7E0010: raw(expected)}))
        assert r.success, f"lerp({a}, {b}, {t}): {r.error} {r.failures}"

    def test_t_zero_gives_a(self, e2e):
        self._lerp(e2e, 256, 768, 0, 256)

    def test_t_one_gives_b(self, e2e):
        self._lerp(e2e, 256, 768, 256, 768)

    def test_midpoint(self, e2e):
        self._lerp(e2e, 256, 768, 128, 512)       # 1.0 -> 3.0, half = 2.0

    def test_descending(self, e2e):
        """b < a, the case that would wrap if the subtraction were unguarded."""
        self._lerp(e2e, 768, 256, 128, 512)

    def test_descending_to_zero(self, e2e):
        self._lerp(e2e, 768, 0, 256, 0)

    def test_quarter_step(self, e2e):
        self._lerp(e2e, 0, 1024, 64, 256)         # 0.0 -> 4.0 at 0.25 = 1.0

    def test_runtime_values_not_just_literals(self, e2e):
        r = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q8;
            #[zeropage(0x20)] static mut AV: Q8;
            #[zeropage(0x22)] static mut BV: Q8;
            #[zeropage(0x24)] static mut TV: Q8;
            #[entry]
            fn main() {{
                AV = Q8(5120);
                BV = Q8(1024);
                TV = Q8(128);
                OUT = Q8::lerp(AV, BV, TV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(3072)}))   # 20.0 -> 4.0, half = 12.0
        assert r.success, f"{r.error} {r.failures}"


class TestQ8Clamp:
    """`clamp!` is a method macro yielding a value, not mutating its receiver."""

    LIMITS = "Q8(256), Q8(1024)"

    def _clamp(self, e2e, start, expected):
        r = e2e.run(program(f"OUT = Q8({start}).clamp!({self.LIMITS});"),
                    ExpectedState(memory={0x7E0010: raw(expected)}))
        assert r.success, f"clamp({start}): {r.error} {r.failures}"

    def test_below_the_floor(self, e2e):
        self._clamp(e2e, 0, 256)

    def test_above_the_ceiling(self, e2e):
        self._clamp(e2e, 60000, 1024)

    def test_inside_is_unchanged(self, e2e):
        self._clamp(e2e, 512, 512)

    def test_on_the_bounds(self, e2e):
        self._clamp(e2e, 256, 256)
        self._clamp(e2e, 1024, 1024)


class TestQ8ToString:
    """Two decimal places, truncated. Reachable from `format!("{s}", q)`."""

    BUF = 0x7E2000

    def read_n(self, cpu, n):
        return ''.join(chr(cpu.memory.read(self.BUF + i)) for i in range(n))

    def _str(self, e2e, expr, expected):
        r = e2e.run(program(f"let q: Q8 = {expr};\n"
                            "LEN = q.to_string(&BUF as far *u8) as u8;",
                            statics=("#[ram]\nstatic mut BUF: [u8; 16];\n"
                                     "#[zeropage(0x30)]\nstatic mut LEN: u8;")),
                    ExpectedState(memory={0x7E0030: len(expected)}))
        assert r.success, f"{expr}: {r.error} {r.failures}"
        assert self.read_n(r.cpu, len(expected)) == expected

    def test_zero(self, e2e):
        self._str(e2e, "Q8(0)", "0.00")

    def test_one(self, e2e):
        self._str(e2e, "Q8::from_int(1)", "1.00")

    def test_three_digits(self, e2e):
        self._str(e2e, "Q8::from_int(100)", "100.00")

    def test_top_of_the_integer_range(self, e2e):
        self._str(e2e, "Q8::from_int(255)", "255.00")

    def test_a_half(self, e2e):
        self._str(e2e, "Q8::from(0, 128)", "0.50")

    def test_three_quarters(self, e2e):
        self._str(e2e, "Q8::from(12, 192)", "12.75")

    def test_smallest_fraction_truncates_to_zero(self, e2e):
        """1/256 is 0.0039, which truncates to "0.00"."""
        self._str(e2e, "Q8::from(0, 1)", "0.00")

    def test_largest_fraction_cannot_carry(self, e2e):
        """255/256 is 0.996. Rounding would give "1.00" and have to carry into
        the integer part; truncation caps the hundredths at 99 instead."""
        self._str(e2e, "Q8::from(0, 255)", "0.99")

    def test_carry_would_have_hit_the_top_of_the_range(self, e2e):
        """255 + 255/256 is the largest Q8. Rounding the fraction would need to
        increment an integer part that has no room left."""
        self._str(e2e, "Q8(65535)", "255.99")

    def test_writes_nothing_past_the_returned_count(self, e2e):
        r = e2e.run(program("let mut i: u16 = 0;\n"
                            "while i < 16 { BUF[i] = 0xFF; i = i + 1; }\n"
                            "let q: Q8 = Q8::from(12, 192);\n"
                            "LEN = q.to_string(&BUF as far *u8) as u8;",
                            statics=("#[ram]\nstatic mut BUF: [u8; 16];\n"
                                     "#[zeropage(0x30)]\nstatic mut LEN: u8;")),
                    ExpectedState(memory={0x7E0030: 5}))
        assert r.success, f"{r.error} {r.failures}"
        assert self.read_n(r.cpu, 5) == "12.75"
        assert r.cpu.memory.read(self.BUF + 5) == 0xFF, (
            "to_string must not write past the count it returns")


class TestQ8IsNominal:
    """What the newtype does and does not guarantee."""

    def test_a_u16_flows_in(self, e2e):
        """Transparent-in: any value assignable to the payload is a valid Q8."""
        r = e2e.run(program("let n: u16 = 3264;\nOUT = n;"),
                    ExpectedState(memory={0x7E0010: raw(3264)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_q8_does_not_pass_as_a_u16(self, e2e):
        r = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: u16;
            #[entry]
            fn main() {{ OUT = Q8::from_int(3); }}
        ''', ExpectedState(memory={}))
        assert not r.success, "expected a compile error, got none"

    def test_q8_does_not_mix_with_q10(self, e2e):
        """Two fixed-point newtypes on different scales must not interchange."""
        r = e2e.run(f'''{PRELUDE}
            include!("{STDLIB_DIR / "Q10.r65"}")
            #[zeropage(0x10)] static mut OUT: Q8;
            #[entry]
            fn main() {{ OUT = Q10::from_int(3); }}
        ''', ExpectedState(memory={}))
        assert not r.success, "expected a compile error, got none"
