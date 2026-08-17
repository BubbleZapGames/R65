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
Q10_PATH = STDLIB_DIR / "Q10.r65"

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

    def test_to_int_floors_negatives(self, e2e):
        """`>>` is a logical shift for signed types too, so a bare `self.0 >> 6`
        shifted the sign bit down and turned -2.0 into 1022."""
        result = e2e.run(
            program("OUT = Q10(0 - 128).to_int();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: raw(-2)}))
        assert result.success, f"Failures: {result.failures}"

    def test_to_int_floors_rather_than_truncating(self, e2e):
        """-1.5 floors to -2; truncation would give -1."""
        result = e2e.run(
            program("OUT = Q10(0 - 96).to_int();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: raw(-2)}))
        assert result.success, f"Failures: {result.failures}"

    def test_to_frac_of_a_negative(self, e2e):
        """Fraction is x - floor(x), so -1.5 has fraction 0.5 (raw 32)."""
        result = e2e.run(
            program("OUT = Q10(0 - 96).to_frac();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: raw(32)}))
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


class TestQ10Round:
    """Nearest whole number, ties up — contrast with to_int, which floors."""

    def _round(self, e2e, raw_value, expected):
        result = e2e.run(
            program(f"OUT = Q10({raw_value}).round();",
                    statics="#[zeropage(0x10)]\nstatic mut OUT: i16;"),
            ExpectedState(memory={0x7E0010: raw(expected)}))
        assert result.success, f"Failures: {result.failures}"

    def test_below_half_rounds_down(self, e2e):
        self._round(e2e, 31, 0)          # 0.484

    def test_exactly_half_rounds_up(self, e2e):
        self._round(e2e, 32, 1)          # 0.5

    def test_one_and_a_half(self, e2e):
        self._round(e2e, 96, 2)          # 1.5

    def test_just_below_one_and_a_half(self, e2e):
        self._round(e2e, 95, 1)          # 1.484

    def test_negative_half_rounds_toward_positive(self, e2e):
        self._round(e2e, "0 - 32", 0)    # -0.5 -> 0, ties go up

    def test_negative_one_and_a_half(self, e2e):
        self._round(e2e, "0 - 96", -1)   # -1.5 -> -1

    def test_just_past_negative_one_and_a_half(self, e2e):
        self._round(e2e, "0 - 97", -2)   # -1.516

    def test_whole_values_are_unchanged(self, e2e):
        self._round(e2e, 192, 3)

    def test_differs_from_to_int(self, e2e):
        """The pair that motivates having both: 1.5 floors to 1, rounds to 2."""
        result = e2e.run(
            program("FLOORED = Q10(96).to_int();\nROUNDED = Q10(96).round();",
                    statics=("#[zeropage(0x10)]\nstatic mut FLOORED: i16;\n"
                             "#[zeropage(0x12)]\nstatic mut ROUNDED: i16;")),
            ExpectedState(memory={0x7E0010: raw(1), 0x7E0012: raw(2)}))
        assert result.success, f"Failures: {result.failures}"


LIMITS_LITERAL = "Q10(0), Q10(64)"


class TestQ10Clamp:
    """`clamp!` is a method macro yielding a value, not mutating its receiver."""

    LIMITS = "Q10(0 - 64), Q10(64)"

    def _clamp(self, e2e, start, expected):
        result = e2e.run(
            program(f"OUT = Q10({start}).clamp!({self.LIMITS});"),
            ExpectedState(memory={0x7E0010: raw(expected)}))
        assert result.success, f"Failures: {result.failures}"

    def test_below_range_is_raised(self, e2e):
        self._clamp(e2e, "0 - 500", -64)

    def test_above_range_is_lowered(self, e2e):
        self._clamp(e2e, 500, 64)

    def test_inside_range_is_untouched(self, e2e):
        self._clamp(e2e, 32, 32)

    def test_exactly_on_the_bounds(self, e2e):
        self._clamp(e2e, "0 - 64", -64)
        self._clamp(e2e, 64, 64)

    def test_clamps_a_local(self, e2e):
        result = e2e.run(program(
            "let v: Q10 = Q10(900);\nOUT = v.clamp!(Q10(0), Q10(128));"),
            ExpectedState(memory={0x7E0010: raw(128)}))
        assert result.success, f"Failures: {result.failures}"

    def test_reassigns_its_own_receiver(self, e2e):
        """The idiomatic in-place use, now written explicitly."""
        result = e2e.run(program(
            f"OUT = Q10(500);\nOUT = OUT.clamp!({self.LIMITS});"),
            ExpectedState(memory={0x7E0010: raw(64)}))
        assert result.success, f"Failures: {result.failures}"

    def test_composes_in_a_larger_expression(self, e2e):
        """A value, so it can feed arithmetic directly."""
        result = e2e.run(program(
            f"OUT = Q10(500).clamp!({self.LIMITS}) + Q10(1);"),
            ExpectedState(memory={0x7E0010: raw(65)}))
        assert result.success, f"Failures: {result.failures}"

    def test_receiver_is_evaluated_once(self, e2e):
        """`self` is substituted textually, so naming it three times would call
        a function receiver three times. The block binds it once."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q10;
            #[zeropage(0x20)] static mut CALLS: u8;
            fn source() -> Q10 {{ CALLS = CALLS + 1; return Q10(500); }}
            #[entry]
            fn main() {{
                CALLS = 0;
                OUT = source().clamp!(Q10(0 - 64), Q10(64));
            }}
        ''', ExpectedState(memory={0x7E0010: raw(64), 0x7E0020: 1}))
        assert result.success, f"Failures: {result.failures}"

    def test_inside_a_loop(self, e2e):
        result = e2e.run(program(
            "OUT = Q10(0);\nN = 0;\n"
            f"while N < 10 {{ OUT = (OUT + 16).clamp!({LIMITS_LITERAL}); N = N + 1; }}",
            statics=("#[zeropage(0x10)]\nstatic mut OUT: Q10;\n"
                     "#[zeropage(0x12)]\nstatic mut N: u8;")),
            ExpectedState(memory={0x7E0010: raw(64), 0x7E0012: 10}))
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
    """Scaling multiply via the SNES hardware 8x8 unit.

    `a.mul(b)`, not `a * b` — Q10 inherits `*` from i16, which multiplies the raw
    values and lands 64x too high. Was the free function
    `q10_mul(a @ A: Q10, b: Q10)`; as a method `self` arrives in A, the same ABI.
    """

    def test_mul_integers(self, e2e):
        """3.0 * 4.0 = 12.0 (192 * 256 -> 768)."""
        result = e2e.run(program("OUT = Q10(192).mul(Q10(256));"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x03]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_fractional(self, e2e):
        """1.5 * 2.0 = 3.0 (96 * 128 -> 192)."""
        result = e2e.run(program("OUT = Q10(96).mul(Q10(128));"),
                         ExpectedState(memory={0x7E0010: [192, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_larger_values(self, e2e):
        """100.0 * 5.0 = 500.0 (6400 * 320 -> 32000)."""
        result = e2e.run(program("OUT = Q10(6400).mul(Q10(320));"),
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
                OUT = a.mul(Q10(224));
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
                OUT = a.mul(b);
            }}
        ''', ExpectedState(memory={0x7E0010: [0xC0, 0x01]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_by_zero(self, e2e):
        result = e2e.run(program("OUT = Q10(192).mul(Q10(0));"),
                         ExpectedState(memory={0x7E0010: [0, 0]}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_two_fractions(self, e2e):
        """0.5 * 0.5 = 0.25 — both operands below one."""
        result = e2e.run(program("OUT = Q10(32).mul(Q10(32));"),
                         ExpectedState(memory={0x7E0010: raw(16)}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_positive_by_negative(self, e2e):
        """The mirror of test_mul_negative — the sign flip on the second operand."""
        result = e2e.run(program("OUT = Q10(128).mul(Q10(0 - 32));"),
                         ExpectedState(memory={0x7E0010: raw(-64)}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_underflows_toward_zero(self, e2e):
        """0.5 * 1/64 falls below the 1/64 scale and truncates to zero."""
        result = e2e.run(program("OUT = Q10(32).mul(Q10(1));"),
                         ExpectedState(memory={0x7E0010: raw(0)}))
        assert result.success, f"Failures: {result.failures}"

    def test_mul_carry_path(self, e2e):
        """Large operands exercise the 32-bit accumulate carry."""
        result = e2e.run(program("OUT = Q10(16384).mul(Q10(128));"),
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
        result = e2e.run(program("let raw: i16 = 256;\nOUT = Q10(192).mul(raw);"),
                         ExpectedState(memory={0x7E0010: [0x00, 0x03]}))
        assert result.success, f"Failures: {result.failures}"

    def test_a_raw_i16_cannot_be_the_receiver(self, e2e):
        """Transparent-in applies to parameters, not to the receiver: method
        lookup goes by the receiver's own type. Tighter than the free function
        `q10_mul` was, and in the direction this library already argues for --
        wrap deliberately where a raw number becomes a fixed-point one."""
        result = e2e.run(program("let raw: i16 = 192;\nOUT = raw.mul(Q10(256));"),
                         ExpectedState(memory={}))
        assert not result.success, "expected a compile error, got none"

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


class TestQ10Div:
    """Scaling divide: (self << 6) / other, by software long division.

    `a.div(b)`, not `a / b` — Q10 inherits `/` from i16, which cancels the scale
    and yields a ratio 64x too small (and only accepts power-of-two constants
    anyway). Needs no `snes` cfg: the hardware divider takes an 8-bit divisor,
    while this needs 22 dividend bits over a 16-bit one.
    """

    def _div(self, e2e, a: int, b: int, expected: int):
        result = e2e.run(
            program(f"let x: Q10 = Q10({a});\n"
                    f"let y: Q10 = Q10({b});\n"
                    f"OUT = x.div(y);"),
            ExpectedState(memory={0x7E0010: raw(expected)}))
        assert result.success, f"Q10({a}).div(Q10({b})): {result.failures}"

    def test_by_one(self, e2e):
        self._div(e2e, 192, 64, 192)               # 3.0 / 1.0 = 3.0

    def test_identity(self, e2e):
        self._div(e2e, 192, 192, 64)              # 3.0 / 3.0 = 1.0

    def test_by_a_fraction_grows(self, e2e):
        self._div(e2e, 128, 32, 256)              # 2.0 / 0.5 = 4.0

    def test_yields_a_fraction(self, e2e):
        self._div(e2e, 64, 128, 32)               # 1.0 / 2.0 = 0.5

    def test_three_quarters(self, e2e):
        self._div(e2e, 192, 256, 48)              # 3.0 / 4.0 = 0.75

    def test_larger_operands(self, e2e):
        """768 << 6 is 49152 — past 16 bits, so the wide dividend path matters."""
        self._div(e2e, 768, 192, 256)             # 12.0 / 3.0 = 4.0

    def test_zero_numerator(self, e2e):
        self._div(e2e, 0, 192, 0)

    def test_truncates_toward_zero(self, e2e):
        """1.0 / 3.0 is 21.33/64, truncated to 21 — not rounded to 21.33."""
        self._div(e2e, 64, 192, 21)

    def test_negative_numerator(self, e2e):
        self._div(e2e, -64, 128, -32)             # -1.0 / 2.0 = -0.5

    def test_negative_divisor(self, e2e):
        self._div(e2e, 64, -128, -32)             # 1.0 / -2.0 = -0.5

    def test_both_negative(self, e2e):
        self._div(e2e, -64, -128, 32)             # -1.0 / -2.0 = 0.5

    def test_smallest_divisor(self, e2e):
        """Dividing by 1/64 multiplies by 64: 4.0 / (1/64) = 256.0."""
        self._div(e2e, 256, 1, 16384)

    def test_divide_by_zero_saturates(self, e2e):
        self._div(e2e, 192, 0, 32767)

    def test_divide_by_zero_saturates_negative(self, e2e):
        self._div(e2e, -192, 0, -32767)

    def test_runtime_values_not_just_literals(self, e2e):
        """Through statics, so nothing const-folds."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q10;
            #[zeropage(0x20)] static mut AV: Q10;
            #[zeropage(0x22)] static mut BV: Q10;
            #[entry]
            fn main() {{
                AV = Q10(0 - 768);
                BV = Q10(96);
                OUT = AV.div(BV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(-512)}))   # -12.0 / 1.5 = -8.0
        assert result.success, f"Failures: {result.failures}"

    def test_round_trips_through_mul(self, e2e):
        """`div` and `mul` check each other: 12.0 / 3.0 * 3.0 is exactly 12.0 for
        operands that divide evenly, so a scale error in either would show."""
        result = e2e.run(
            program("let a: Q10 = Q10(768);\n"
                    "let b: Q10 = Q10(192);\n"
                    "OUT = a.div(b).mul(b);"),
            ExpectedState(memory={0x7E0010: raw(768)}))
        assert result.success, f"Failures: {result.failures}"


class TestQ10DivU8:
    """Divide by an integer count, via the SNES hardware divider.

    Dividing a fixed-point value by an integer preserves the scale, so this is a
    straight 16/8 divide with no shifting — one hardware operation against the 22
    software steps `div` needs for a Q10 divisor.
    """

    def _div(self, e2e, a: int, d: int, expected: int):
        result = e2e.run(
            program(f"let x: Q10 = Q10({a});\nOUT = x.div_u8({d});"),
            ExpectedState(memory={0x7E0010: raw(expected)}))
        assert result.success, f"Q10({a}).div_u8({d}): {result.failures}"

    def test_by_one_is_identity(self, e2e):
        self._div(e2e, 768, 1, 768)

    def test_exact(self, e2e):
        self._div(e2e, 768, 3, 256)                # 12.0 / 3 = 4.0

    def test_halves_a_fraction(self, e2e):
        self._div(e2e, 64, 2, 32)                  # 1.0 / 2 = 0.5

    def test_truncates_toward_zero(self, e2e):
        """1.0 / 3 is 21.33/64, truncated to 21."""
        self._div(e2e, 64, 3, 21)

    def test_zero_numerator(self, e2e):
        self._div(e2e, 0, 7, 0)

    def test_largest_divisor(self, e2e):
        """255 is the widest the 8-bit divisor takes; 32767/255 = 128.5 -> 128."""
        self._div(e2e, 32767, 255, 128)

    def test_dividend_above_the_signed_byte_range(self, e2e):
        """The dividend is 16-bit even though the divisor is not."""
        self._div(e2e, 16384, 4, 4096)             # 256.0 / 4 = 64.0

    def test_negative(self, e2e):
        self._div(e2e, -768, 3, -256)

    def test_negative_truncates_toward_zero(self, e2e):
        """-1.0 / 3 is -21.33/64; toward zero is -21, not -22."""
        self._div(e2e, -64, 3, -21)

    def test_by_zero_saturates(self, e2e):
        self._div(e2e, 768, 0, 32767)

    def test_by_zero_saturates_negative(self, e2e):
        self._div(e2e, -768, 0, -32767)

    def test_runtime_divisor(self, e2e):
        """Through statics, so the divisor is not a folded constant."""
        result = e2e.run(f'''{PRELUDE}
            #[zeropage(0x10)] static mut OUT: Q10;
            #[zeropage(0x20)] static mut AV: Q10;
            #[zeropage(0x22)] static mut DV: u8;
            #[entry]
            fn main() {{
                AV = Q10(0 - 1920);
                DV = 6;
                OUT = AV.div_u8(DV);
            }}
        ''', ExpectedState(memory={0x7E0010: raw(-320)}))   # -30.0 / 6 = -5.0
        assert result.success, f"Failures: {result.failures}"

    def test_agrees_with_div(self, e2e):
        """The two divides must give the same answer when the divisor is a whole
        number — one goes through hardware, the other through 22 software steps."""
        result = e2e.run(
            program("let a: Q10 = Q10(768);\n"
                    "OUT = a.div_u8(3);\n"
                    "OUT2 = a.div(Q10::from_int(3));",
                    statics=("#[zeropage(0x10)]\nstatic mut OUT: Q10;\n"
                             "#[zeropage(0x12)]\nstatic mut OUT2: Q10;")),
            ExpectedState(memory={0x7E0010: raw(256), 0x7E0012: raw(256)}))
        assert result.success, f"Failures: {result.failures}"
