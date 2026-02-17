"""Tests for I32 (32-bit signed integer) stdlib module.

Compiles the actual stdlib/I32.r65 through the full pipeline:
parse -> preprocess -> expand macros -> HIR build -> type check.
"""

import tempfile
import pytest
from pathlib import Path
from r65.compiler.frontend import parse, preprocess, expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.hir.cfg import CfgEvaluator
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError


STDLIB_DIR = str(Path(__file__).resolve().parents[3] / "stdlib")


def build_and_check(source: str, cfg_features=None):
    """Compile R65 source with stdlib includes through the full pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "test.r65"
        src_path.write_text(source)
        src_str = str(src_path)

        program = parse(source, src_str)
        program = preprocess(program, src_str, include_paths=[STDLIB_DIR])
        program = expand_macros(program)

        cfg = CfgEvaluator(set(cfg_features or []), {})
        builder = HIRBuilder(source_file=src_str, cfg_evaluator=cfg,
                             include_paths=[STDLIB_DIR])
        hir_prog = builder.build_program(program)

        checker = TypeChecker(hir_prog)
        checker.check()
        return hir_prog


HEADER = 'include!("sneslib.r65")\ninclude!("I32.r65")\n'


# =============================================================================
# Compilation
# =============================================================================

class TestI32Compilation:
    """I32 stdlib compiles through the full pipeline."""

    def test_compiles_with_snes(self):
        """Compiles with SNES hardware cfg."""
        build_and_check(HEADER, cfg_features=["snes"])

    def test_compiles_software(self):
        """Compiles without SNES cfg (software fallback)."""
        build_and_check(HEADER)


# =============================================================================
# Struct and Impl
# =============================================================================

class TestI32Struct:
    """I32 struct definition."""

    def test_struct_fields(self):
        """I32 has lo and hi u16 fields."""
        from r65.compiler.hir import HIRStructDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        structs = [d for d in hir.declarations if isinstance(d, HIRStructDecl)]
        i32 = next(s for s in structs if s.name == "I32")
        assert len(i32.fields) == 2
        assert i32.fields[0].name == "lo"
        assert i32.fields[1].name == "hi"

    def test_static_declaration(self):
        """I32 can be declared as a static variable."""
        build_and_check(HEADER + """
#[zeropage]
static mut VALUE: I32;
""", cfg_features=["snes"])


class TestI32ImplBlock:
    """I32 impl block structure."""

    def test_is_far_impl(self):
        """impl far I32 is parsed as far."""
        from r65.compiler.hir import HIRImplDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        impls = [d for d in hir.declarations if isinstance(d, HIRImplDecl)]
        i32_impl = next(i for i in impls if i.struct_name == "I32")
        assert i32_impl.is_far is True

    def test_method_mangling(self):
        """Methods are mangled to I32__name."""
        from r65.compiler.hir import HIRImplDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        impls = [d for d in hir.declarations if isinstance(d, HIRImplDecl)]
        i32_impl = next(i for i in impls if i.struct_name == "I32")
        names = [m.name for m in i32_impl.methods]
        assert "I32__add" in names
        assert "I32__sub" in names
        assert "I32__neg" in names
        assert "I32__abs" in names
        assert "I32__cmp" in names
        assert "I32__copy" in names
        assert "I32__shl" in names
        assert "I32__sar" in names


# =============================================================================
# Conversion Methods
# =============================================================================

class TestI32ConversionMethods:
    """I32 conversion and copy methods."""

    def test_from_i16(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.from_i16(1000 as i16); }
""", cfg_features=["snes"])

    def test_from_i16_negative(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.from_i16(-100 as i16); }
""", cfg_features=["snes"])

    def test_from_u16(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.from_u16(1000); }
""", cfg_features=["snes"])

    def test_to_i16(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
#[zeropage]
static mut R: u16;
fn test() { A = V.to_i16(); R = A; }
""", cfg_features=["snes"])

    def test_copy(self):
        build_and_check(HEADER + """
#[zeropage]
static mut DST: I32;
#[zeropage]
static mut SRC: far *I32;
fn test() { DST.copy(SRC); }
""", cfg_features=["snes"])


# =============================================================================
# Sign Methods
# =============================================================================

class TestI32SignMethods:
    """I32 sign-related methods."""

    def test_is_negative(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
#[zeropage]
static mut R: bool;
fn test() { R = V.is_negative(); }
""", cfg_features=["snes"])

    def test_neg(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.neg(); }
""", cfg_features=["snes"])

    def test_abs(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.abs(); }
""", cfg_features=["snes"])


# =============================================================================
# Arithmetic Methods
# =============================================================================

class TestI32ArithmeticMethods:
    """I32 arithmetic methods."""

    def test_add(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
fn test() { A_VAL.add(B_PTR); }
""", cfg_features=["snes"])

    def test_sub(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
fn test() { A_VAL.sub(B_PTR); }
""", cfg_features=["snes"])

    def test_mul(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
fn test() { A_VAL.mul(B_PTR); }
""", cfg_features=["snes"])

    def test_div(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
fn test() { A_VAL.div(B_PTR); }
""", cfg_features=["snes"])

    def test_mod(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
fn test() { A_VAL.mod(B_PTR); }
""", cfg_features=["snes"])


# =============================================================================
# Comparison and Shifts
# =============================================================================

class TestI32ComparisonMethod:
    """I32 comparison method."""

    def test_cmp_returns_u8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: I32;
#[zeropage]
static mut B_PTR: far *I32;
#[zeropage]
static mut FLAGS: u8;
fn test() { A = A_VAL.cmp(B_PTR); FLAGS = A; }
""", cfg_features=["snes"])


class TestI32ShiftMethods:
    """I32 shift methods."""

    def test_shl(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.shl(4); }
""", cfg_features=["snes"])

    def test_sar(self):
        """Arithmetic shift right preserves sign."""
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.sar(4); }
""", cfg_features=["snes"])


# =============================================================================
# Hardware Methods
# =============================================================================

class TestI32HardwareMethods:
    """I32 SNES hardware-accelerated methods."""

    def test_div_i8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
#[zeropage]
static mut REM: u8;
fn test() { A = V.div_i8(10); REM = A; }
""", cfg_features=["snes"])

    def test_mod_i8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() { V.mod_i8(10); }
""", cfg_features=["snes"])


# =============================================================================
# Macros
# =============================================================================

class TestI32Macros:
    """I32 convenience macros."""

    def test_i32_add(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *I32;
#[zeropage]
static mut B_PTR: far *I32;
#[zeropage]
static mut RESULT: I32;
fn test() { i32_add!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_i32_sub(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *I32;
#[zeropage]
static mut B_PTR: far *I32;
#[zeropage]
static mut RESULT: I32;
fn test() { i32_sub!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_i32_neg(self):
        build_and_check(HEADER + """
#[zeropage]
static mut SRC: far *I32;
#[zeropage]
static mut RESULT: I32;
fn test() { i32_neg!(RESULT, SRC); }
""", cfg_features=["snes"])

    def test_i32_abs(self):
        build_and_check(HEADER + """
#[zeropage]
static mut SRC: far *I32;
#[zeropage]
static mut RESULT: I32;
fn test() { i32_abs!(RESULT, SRC); }
""", cfg_features=["snes"])


# =============================================================================
# Usage Patterns
# =============================================================================

class TestI32UsagePatterns:
    """Common I32 usage patterns."""

    def test_init_and_accumulate(self):
        build_and_check(HEADER + """
#[zeropage]
static mut COUNTER: I32;
#[zeropage]
static mut INCREMENT: I32;
#[zeropage]
static mut INC_PTR: far *I32;
fn init() {
    COUNTER.from_i16(0 as i16);
    INCREMENT.from_i16(1 as i16);
}
fn tick() { COUNTER.add(INC_PTR); }
""", cfg_features=["snes"])

    def test_method_via_pointer(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
#[zeropage]
static mut PTR: far *I32;
#[zeropage]
static mut R: u16;
fn test() { A = PTR.to_i16(); R = A; }
""", cfg_features=["snes"])

    def test_negative_literals(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
fn test() {
    V.from_i16(-1 as i16);
    V.from_i16(-32768 as i16);
}
""", cfg_features=["snes"])


# =============================================================================
# Error Cases
# =============================================================================

class TestI32Errors:
    """I32 type errors."""

    def test_wrong_argument_type(self):
        """Passing far *u16 where far *I32 expected fails."""
        with pytest.raises(TypeCheckError):
            build_and_check(HEADER + """
#[zeropage]
static mut V: I32;
#[zeropage]
static mut BAD: far *u16;
fn test() { V.add(BAD); }
""", cfg_features=["snes"])

    def test_near_pointer_to_far_impl(self):
        """Using near pointer with far impl method fails."""
        with pytest.raises(TypeCheckError):
            build_and_check(HEADER + """
#[zeropage]
static mut PTR: *I32;
fn test() -> i16 { return PTR.to_i16(); }
""", cfg_features=["snes"])

    def test_i32_vs_u32_type_mismatch(self):
        """I32 and U32 are different types - can't mix them."""
        with pytest.raises(TypeCheckError):
            build_and_check(
                'include!("sneslib.r65")\n'
                'include!("U32.r65")\n'
                'include!("I32.r65")\n'
                """
#[zeropage]
static mut SIGNED: I32;
#[zeropage]
static mut UNSIGNED_PTR: far *U32;
fn test() { SIGNED.add(UNSIGNED_PTR); }
""", cfg_features=["snes"])
