# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for the Q10.6 fixed-point library.

`Q10` is a newtype over `i16`, so these check two things at once: that the
fixed-point arithmetic is right, and that wrapping it in a distinct type costs
nothing — values still live in two bytes and ride in registers.
"""

from pathlib import Path
from r65.tests.e2e import ExpectedState

STDLIB_DIR = Path(__file__).parent.parent.parent.parent / "stdlib"
SNESLIB_PATH = STDLIB_DIR / "sneslib.r65"
Q10_PATH = STDLIB_DIR / "q10_type.r65"

PRELUDE = f'''
            include!("{SNESLIB_PATH}")
            include!("{Q10_PATH}")
'''


def program(body: str, statics: str = "#[zeropage(0x10)]\nstatic mut OUT: Q10;") -> str:
    return f"{PRELUDE}\n{statics}\n\n#[entry]\nfn main() {{\n{body}\n}}"


def raw(v: int):
    """Little-endian bytes of a Q10's raw i16 payload."""
    v &= 0xFFFF
    return [v & 0xFF, v >> 8]


class TestQ10Storage:
    """A Q10 is two bytes, like the i16 it wraps."""

    def test_static_variable(self, e2e):
        result = e2e.run(program("OUT = Q10(64);"),
                         ExpectedState(memory={0x7E0010: [64, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_let_binding(self, e2e):
        result = e2e.run(program("let v: Q10 = Q10(128);\nOUT = v;"),
                         ExpectedState(memory={0x7E0010: [128, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_constant(self, e2e):
        """`const Q10_ONE: Q10 = 64;` — a payload literal flows into the newtype."""
        result = e2e.run(program("OUT = Q10_ONE;"),
                         ExpectedState(memory={0x7E0010: [64, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_negative_value(self, e2e):
        result = e2e.run(program("OUT = Q10(0 - 192);"),
                         ExpectedState(memory={0x7E0010: [0x40, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Conversions:
    """Q10::from_int / Q10::from and the accessor methods."""

    def test_from_int(self, e2e):
        """3.0 -> 3 << 6 = 192."""
        result = e2e.run(program("OUT = Q10::from_int(3);"),
                         ExpectedState(memory={0x7E0010: [192, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_whole_and_fraction(self, e2e):
        """1.5 -> (1 << 6) | 32 = 96."""
        result = e2e.run(program("OUT = Q10::from(1, 32);"),
                         ExpectedState(memory={0x7E0010: [96, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_with_a_negative_whole_part(self, e2e):
        """The sign lives in the whole part; `f` is unsigned. -1 + 32/64."""
        result = e2e.run(program("OUT = Q10::from(0 - 1, 32);"),
                         ExpectedState(memory={0x7E0010: raw(-32)}))
        assert result.success, f"Failures: {result.failures}"

    def test_fraction_at_the_top_of_its_range(self, e2e):
        result = e2e.run(program("OUT = Q10::from(0, 63);"),
                         ExpectedState(memory={0x7E0010: [63, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_from_with_runtime_values(self, e2e):
        """`f` is a 1-byte stack slot read in m16, so the high byte comes from
        whatever sits next to it — GUARD proves the mask handles that."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q10;
            #[zeropage(0x20)] static mut NV: i16;
            #[zeropage(0x22)] static mut FV: u8;
            #[zeropage(0x23)] static mut GUARD: u8;
            #[entry]
            fn main() {{
                GUARD = 0xEE;
                NV = 3;
                FV = 48;
                OUT = Q10::from(NV, FV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(240), 0x7E0023: 0xEE}))
        assert result.success, f"Failures: {result.failures}"

    def test_negative_fraction_is_rejected(self, e2e):
        """`f` is u8, so a negative numerator is a compile error rather than
        being masked into 59."""
        result = e2e.run(program("OUT = Q10::from(1, -5);"), ExpectedState(memory={}))
        assert not result.success, "expected a compile error for a negative fraction"
        assert "does not fit in type u8" in (result.error or ""), result.error

    def test_oversized_fraction_is_rejected(self, e2e):
        result = e2e.run(program("OUT = Q10::from(1, 300);"), ExpectedState(memory={}))
        assert not result.success, "expected a compile error for f > 255"
        assert "does not fit in type u8" in (result.error or ""), result.error

    def test_to_int(self, e2e):
        result = e2e.run(
            program("OUT = Q10::from_int(3).to_int();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: [3, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_to_frac(self, e2e):
        """The fractional part of 1.5 is 32/64."""
        result = e2e.run(
            program("OUT = Q10::from(1, 32).to_frac();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: [32, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_round_trip_through_a_function(self, e2e):
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)]
            static mut OUT: Q10;

            fn add_one(val: Q10) -> Q10 {{ return val + 1; }}

            #[entry]
            fn main() {{ OUT = add_one(Q10(63)); }}
        ''', ExpectedState(memory={0x7E0010: [64, 0]}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10InheritedOperators:
    """Addition, subtraction and comparison come from i16 and stay Q10."""

    def test_addition(self, e2e):
        result = e2e.run(program("OUT = Q10::from_int(3) + Q10::from(0, 32);"),
                         ExpectedState(memory={0x7E0010: [224, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_subtraction(self, e2e):
        result = e2e.run(program("OUT = Q10::from_int(3) - Q10::from_int(1);"),
                         ExpectedState(memory={0x7E0010: [128, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_negation(self, e2e):
        """Unary minus is inherited: -3.0 is 0xFF40."""
        result = e2e.run(program("OUT = -Q10::from_int(3);"),
                         ExpectedState(memory={0x7E0010: [0x40, 0xFF]}))
        assert result.success, f"Failures: {result.failures}"

    def test_signed_comparison(self, e2e):
        """A negative Q10 must compare as signed, not as a large unsigned."""
        result = e2e.run(
            program("if Q10::from_int(0 - 2) < Q10(0) { OUT = 1; } else { OUT = 2; }",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: u8;"),
            ExpectedState(memory={0x7E0010: 1}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Abs:
    def test_abs_positive(self, e2e):
        result = e2e.run(program("OUT = Q10::from_int(5).abs();"),
                         ExpectedState(memory={0x7E0010: [0x40, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_negative(self, e2e):
        result = e2e.run(program("OUT = Q10::from_int(0 - 5).abs();"),
                         ExpectedState(memory={0x7E0010: [0x40, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_abs_zero(self, e2e):
        result = e2e.run(program("OUT = Q10(0).abs();"),
                         ExpectedState(memory={0x7E0010: [0, 0]}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10Mul:
    """Scaling multiply via the SNES hardware 8x8 unit."""

    def test_mul_integers(self, e2e):
        """3.0 * 4.0 = 12.0 (192 * 256 -> 768)."""
        result = e2e.run(program("OUT = q10_mul(Q10(192), Q10(256));"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x03]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_fractional(self, e2e):
        """1.5 * 2.0 = 3.0 (96 * 128 -> 192)."""
        result = e2e.run(program("OUT = q10_mul(Q10(96), Q10(128));"),
                         ExpectedState(memory={0x7E0010: [192, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_larger_values(self, e2e):
        """100.0 * 5.0 = 500.0 (6400 * 320 -> 32000)."""
        result = e2e.run(program("OUT = q10_mul(Q10(6400), Q10(320));"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x7D]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_negative(self, e2e):
        """-2.0 * 3.5 = -7.0 (-128 * 224 -> -448 = 0xFE40)."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)]
            static mut OUT: Q10;

            #[entry]
            fn main() {{
                let a: Q10 = Q10(0 - 128);
                OUT = q10_mul(a, Q10(224));
            }}
        ''', ExpectedState(memory={0x7E0010: [0x40, 0xFE]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_both_negative(self, e2e):
        """-2.0 * -3.5 = 7.0 (448 = 0x01C0)."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)]
            static mut OUT: Q10;

            #[entry]
            fn main() {{
                let a: Q10 = Q10(0 - 128);
                let b: Q10 = Q10(0 - 224);
                OUT = q10_mul(a, b);
            }}
        ''', ExpectedState(memory={0x7E0010: [0xC0, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_zero(self, e2e):
        result = e2e.run(program("OUT = q10_mul(Q10(192), Q10(0));"),
                         ExpectedState(memory={0x7E0010: [0, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_carry_path(self, e2e):
        """Large operands exercise the 32-bit accumulate carry."""
        result = e2e.run(program("OUT = q10_mul(Q10(16384), Q10(128));"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x80]}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10IsNominal:
    """What the newtype does and does not guarantee.

    `Q10` is transparent in and opaque out, so the protection is one-directional:
    a Q10 result cannot silently be consumed as a raw i16, and cannot be confused
    with a different newtype. An unscaled i16 flowing *in* is still accepted by
    design — see `test_an_unscaled_i16_still_flows_in`, which pins that.

    Under the old `type q10 = i16;` alias neither direction was checked.
    """

    def _rejects(self, e2e, body, expect):
        result = e2e.run(program(body), ExpectedState(memory={}))
        assert not result.success, f"expected a compile error, got none for: {body}"
        assert expect in (result.error or ""), \
            f"expected {expect!r} in error, got:\n{result.error}"

    def test_q10_does_not_pass_as_an_i16(self, e2e):
        self._rejects(e2e, "let n: i16 = Q10::from_int(3);", "found Q10")

    def test_q10_does_not_pass_into_an_i16_parameter(self, e2e):
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)]
            static mut OUT: i16;

            fn takes_raw(n: i16) -> i16 {{ return n; }}

            #[entry]
            fn main() {{ OUT = takes_raw(Q10::from_int(3)); }}
        ''', ExpectedState(memory={}))
        assert not result.success, "expected a compile error, got none"
        assert "found Q10" in (result.error or ""), result.error

    def test_an_unscaled_i16_still_flows_in(self, e2e):
        """Transparent-in is deliberate and not width-limited: any value
        assignable to the payload — a literal or a typed i16 — is a valid Q10.
        The newtype tracks the type of results; it does not force every call
        site to spell out the constructor.
        """
        result = e2e.run(program("let raw: i16 = 192;\nOUT = q10_mul(raw, Q10(256));"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x03]}))
        assert result.success, f"Failures: {result.failures}"

    def test_q10_does_not_mix_with_another_newtype(self, e2e):
        result = e2e.run(f'''{PRELUDE}
            struct Ticks(i16);
            #[zeropage(0x10)]
            static mut OUT: Q10;

            #[entry]
            fn main() {{ let t: Ticks = 4; OUT = Q10::from_int(1) + t; }}
        ''', ExpectedState(memory={}))
        assert not result.success, "expected a compile error, got none"
        assert "mismatched types" in (result.error or ""), result.error

    def test_unscaled_multiply_is_still_rejected(self, e2e):
        """`*` is restricted to power-of-2 constants, so the wrong spelling of a
        Q10 multiply does not quietly compile either."""
        self._rejects(e2e, "OUT = Q10::from_int(3) * Q10::from_int(4);", "power-of-2")


class TestQ10Lerp:
    """Linear interpolation. `t` of Q10_ONE yields `b`; `t` is not clamped."""

    def test_t_zero_gives_a(self, e2e):
        r = e2e.run(program("OUT = Q10::lerp(Q10(64), Q10(192), Q10(0));"),
                    ExpectedState(memory={0x7E0010: raw(64)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_t_one_gives_b(self, e2e):
        r = e2e.run(program("OUT = Q10::lerp(Q10(64), Q10(192), Q10_ONE);"),
                    ExpectedState(memory={0x7E0010: raw(192)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_midpoint(self, e2e):
        """1.0 -> 3.0 at t=0.5 is 2.0 (128)."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(64), Q10(192), Q10_HALF);"),
                    ExpectedState(memory={0x7E0010: raw(128)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_across_zero(self, e2e):
        """-1.0 -> 1.0 at t=0.5 is 0.0."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(0 - 64), Q10(64), Q10_HALF);"),
                    ExpectedState(memory={0x7E0010: raw(0)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_descending(self, e2e):
        """b < a: 3.0 -> 1.0 at t=0.5 is 2.0."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(192), Q10(64), Q10_HALF);"),
                    ExpectedState(memory={0x7E0010: raw(128)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_negative_result(self, e2e):
        """1.0 -> -3.0 at t=0.5 is -1.0 (-64)."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(64), Q10(0 - 192), Q10_HALF);"),
                    ExpectedState(memory={0x7E0010: raw(-64)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_extrapolates_past_b(self, e2e):
        """t = 2.0 is not clamped: 0.0 -> 1.0 at t=2 is 2.0."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(0), Q10(64), Q10(128));"),
                    ExpectedState(memory={0x7E0010: raw(128)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_quarter_step(self, e2e):
        """0.0 -> 4.0 at t=0.25 is 1.0 (64)."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(0), Q10(256), Q10(16));"),
                    ExpectedState(memory={0x7E0010: raw(64)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_truncates_rather_than_rounds(self, e2e):
        """0.0 -> 1.5 at t=21/64 is 31.5 raw, truncated to 31."""
        r = e2e.run(program("OUT = Q10::lerp(Q10(0), Q10(96), Q10(21));"),
                    ExpectedState(memory={0x7E0010: raw(31)}))
        assert r.success, f"{r.error} {r.failures}"

    def test_runtime_values_not_just_literals(self, e2e):
        """Through statics, so nothing const-folds."""
        r = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q10;
            #[zeropage(0x20)] static mut AV: Q10;
            #[zeropage(0x22)] static mut BV: Q10;
            #[zeropage(0x24)] static mut TV: Q10;
            #[entry]
            fn main() {{
                AV = Q10(0 - 128);
                BV = Q10(384);
                TV = Q10(48);
                OUT = Q10::lerp(AV, BV, TV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(256)}))
        assert r.success, f"{r.error} {r.failures}"
