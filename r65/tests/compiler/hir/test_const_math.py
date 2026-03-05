"""
Tests for const math builtins (fixed_sin, fixed_cos, etc.).

Verifies compile-time evaluation, error handling, and const fn integration.
"""
import pytest
import math

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError, HIRProgram,
    HIRStaticDecl, HIRIntegerLiteral, HIRArrayLiteralExpr,
)
from r65.compiler.typeck import TypeChecker


def build_hir(source: str) -> HIRProgram:
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    return builder.build_program(ast_prog)


def build_and_typecheck(source: str) -> HIRProgram:
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    hir = builder.build_program(ast_prog)
    tc = TypeChecker(hir)
    tc.check()
    return hir


def eval_const(source: str) -> int:
    """Build HIR and return the evaluated_value of the first const declaration."""
    hir = build_hir(source)
    return hir.declarations[0].evaluated_value


class TestFixedSin:
    def test_sin_zero(self):
        assert eval_const("const VAL: i16 = fixed_sin(0, 256, 127);") == 0

    def test_sin_quarter(self):
        """sin(pi/2) = 1.0 -> 127"""
        assert eval_const("const VAL: i16 = fixed_sin(64, 256, 127);") == 127

    def test_sin_half(self):
        """sin(pi) ~ 0"""
        assert eval_const("const VAL: i16 = fixed_sin(128, 256, 127);") == 0

    def test_sin_three_quarter(self):
        """sin(3pi/2) = -1.0 -> -127"""
        assert eval_const("const VAL: i16 = fixed_sin(192, 256, 127);") == -127

    def test_sin_zero_table_size_error(self):
        with pytest.raises(HIRError, match="table_size must not be zero"):
            build_hir("const VAL: i16 = fixed_sin(0, 0, 127);")


class TestFixedCos:
    def test_cos_zero(self):
        """cos(0) = 1.0 -> 127"""
        assert eval_const("const VAL: i16 = fixed_cos(0, 256, 127);") == 127

    def test_cos_quarter(self):
        """cos(pi/2) ~ 0"""
        assert eval_const("const VAL: i16 = fixed_cos(64, 256, 127);") == 0

    def test_cos_half(self):
        """cos(pi) = -1.0 -> -127"""
        assert eval_const("const VAL: i16 = fixed_cos(128, 256, 127);") == -127


class TestFixedAtan2:
    def test_atan2_positive_x(self):
        assert eval_const("const VAL: u16 = fixed_atan2(0, 1, 256);") == 0

    def test_atan2_positive_y(self):
        """atan2(1, 0) = 90 degrees = 64/256"""
        assert eval_const("const VAL: u16 = fixed_atan2(1, 0, 256);") == 64

    def test_atan2_negative_x(self):
        """atan2(0, -1) = 180 degrees = 128/256"""
        assert eval_const("const VAL: u16 = fixed_atan2(0, -1, 256);") == 128

    def test_atan2_origin(self):
        assert eval_const("const VAL: u16 = fixed_atan2(0, 0, 256);") == 0


class TestFixedSqrt:
    def test_sqrt_perfect(self):
        assert eval_const("const VAL: u16 = fixed_sqrt(100, 1);") == 10

    def test_sqrt_scaled(self):
        expected = round(math.sqrt(2) * 256)
        assert eval_const("const VAL: u16 = fixed_sqrt(2, 256);") == expected

    def test_sqrt_zero(self):
        assert eval_const("const VAL: u16 = fixed_sqrt(0, 256);") == 0

    def test_sqrt_negative_error(self):
        with pytest.raises(HIRError, match="must not be negative"):
            build_hir("const VAL: u16 = fixed_sqrt(-1, 256);")


class TestFixedLog2:
    def test_log2_powers(self):
        """log2(8) * 256 = 3 * 256 = 768"""
        assert eval_const("const VAL: i16 = fixed_log2(8, 256);") == 768

    def test_log2_one(self):
        assert eval_const("const VAL: i16 = fixed_log2(1, 256);") == 0

    def test_log2_zero_error(self):
        with pytest.raises(HIRError, match="must be positive"):
            build_hir("const VAL: i16 = fixed_log2(0, 256);")

    def test_log2_negative_error(self):
        with pytest.raises(HIRError, match="must be positive"):
            build_hir("const VAL: i16 = fixed_log2(-5, 256);")


class TestFixedExp2:
    def test_exp2_zero(self):
        """2^(0/256) * 256 = 256"""
        assert eval_const("const VAL: u16 = fixed_exp2(0, 256, 256);") == 256

    def test_exp2_one(self):
        """2^(256/256) * 256 = 512"""
        assert eval_const("const VAL: u16 = fixed_exp2(256, 256, 256);") == 512

    def test_exp2_zero_scale_error(self):
        with pytest.raises(HIRError, match="in_scale must not be zero"):
            build_hir("const VAL: u16 = fixed_exp2(1, 0, 256);")


class TestFixedLerp:
    def test_lerp_start(self):
        assert eval_const("const VAL: i16 = fixed_lerp(0, 100, 0, 10);") == 0

    def test_lerp_end(self):
        assert eval_const("const VAL: i16 = fixed_lerp(0, 100, 10, 10);") == 100

    def test_lerp_mid(self):
        assert eval_const("const VAL: i16 = fixed_lerp(0, 100, 5, 10);") == 50

    def test_lerp_negative(self):
        assert eval_const("const VAL: i16 = fixed_lerp(-100, 100, 5, 10);") == 0

    def test_lerp_zero_tmax_error(self):
        with pytest.raises(HIRError, match="t_max must not be zero"):
            build_hir("const VAL: i16 = fixed_lerp(0, 100, 5, 0);")


class TestConstFnIntegration:
    """Test const math builtins inside const fn (transpiled path)."""

    def test_sin_table_via_const_fn(self):
        """Generate sin table inside const fn using for loop."""
        hir = build_hir('''
        const fn build_sin() -> [i16; 4] {
            let mut t: [i16; 4] = [0; 4];
            for i in 0..4 {
                t[i] = fixed_sin(i as u16, 4, 127);
            }
            return t;
        }
        static TABLE: [i16; 4] = build_sin();
        ''')
        static_decl = None
        for d in hir.declarations:
            if isinstance(d, HIRStaticDecl) and d.name == 'TABLE':
                static_decl = d
                break
        assert static_decl is not None
        assert isinstance(static_decl.initializer, HIRArrayLiteralExpr)
        values = [e.value for e in static_decl.initializer.elements]
        # sin(0)=0, sin(pi/2)=127, sin(pi)=0, sin(3pi/2)=-127
        assert values == [0, 127, 0, -127]

    def test_lerp_table_via_const_fn(self):
        """Generate lerp ramp inside const fn."""
        hir = build_hir('''
        const fn build_ramp() -> [i16; 5] {
            let mut t: [i16; 5] = [0; 5];
            for i in 0..5 {
                t[i] = fixed_lerp(0, 100, i as u16, 4);
            }
            return t;
        }
        static RAMP: [i16; 5] = build_ramp();
        ''')
        static_decl = None
        for d in hir.declarations:
            if isinstance(d, HIRStaticDecl) and d.name == 'RAMP':
                static_decl = d
                break
        assert static_decl is not None
        values = [e.value for e in static_decl.initializer.elements]
        assert values == [0, 25, 50, 75, 100]

    def test_type_checking(self):
        """Const math builtins pass type checking."""
        build_and_typecheck('''
        const SIN_VAL: i16 = fixed_sin(0, 256, 127);
        const COS_VAL: i16 = fixed_cos(0, 256, 127);
        const ATAN_VAL: u16 = fixed_atan2(1, 0, 256);
        const SQRT_VAL: u16 = fixed_sqrt(100, 1);
        const LOG_VAL: i16 = fixed_log2(8, 256);
        const EXP_VAL: u16 = fixed_exp2(256, 256, 256);
        const LERP_VAL: i16 = fixed_lerp(0, 100, 5, 10);
        ''')


class TestClamping:
    def test_sin_large_amplitude_clamped(self):
        assert eval_const("const VAL: i16 = fixed_sin(64, 256, 50000);") == 32767

    def test_sqrt_large_clamped(self):
        assert eval_const("const VAL: u16 = fixed_sqrt(65535, 1000);") == 65535
