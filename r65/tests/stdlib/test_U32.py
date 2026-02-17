"""Tests for U32 (32-bit unsigned integer) stdlib module.

Compiles the actual stdlib/U32.r65 through the full pipeline:
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


HEADER = 'include!("sneslib.r65")\ninclude!("U32.r65")\n'


# =============================================================================
# Compilation
# =============================================================================

class TestU32Compilation:
    """U32 stdlib compiles through the full pipeline."""

    def test_compiles_with_snes(self):
        """Compiles with SNES hardware cfg (hardware multiply/divide)."""
        build_and_check(HEADER, cfg_features=["snes"])

    def test_compiles_software(self):
        """Compiles without SNES cfg (software fallback)."""
        build_and_check(HEADER)


# =============================================================================
# Struct and Impl
# =============================================================================

class TestU32Struct:
    """U32 struct definition."""

    def test_struct_fields(self):
        """U32 has lo and hi u16 fields."""
        from r65.compiler.hir import HIRStructDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        structs = [d for d in hir.declarations if isinstance(d, HIRStructDecl)]
        u32 = next(s for s in structs if s.name == "U32")
        assert len(u32.fields) == 2
        assert u32.fields[0].name == "lo"
        assert u32.fields[1].name == "hi"

    def test_static_declaration(self):
        """U32 can be declared as a static variable."""
        build_and_check(HEADER + """
#[zeropage]
static mut VALUE: U32;
""", cfg_features=["snes"])


class TestU32ImplBlock:
    """U32 impl block structure."""

    def test_is_far_impl(self):
        """impl far U32 is parsed as far."""
        from r65.compiler.hir import HIRImplDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        impls = [d for d in hir.declarations if isinstance(d, HIRImplDecl)]
        u32_impl = next(i for i in impls if i.struct_name == "U32")
        assert u32_impl.is_far is True

    def test_method_mangling(self):
        """Methods are mangled to U32__name."""
        from r65.compiler.hir import HIRImplDecl
        hir = build_and_check(HEADER, cfg_features=["snes"])
        impls = [d for d in hir.declarations if isinstance(d, HIRImplDecl)]
        u32_impl = next(i for i in impls if i.struct_name == "U32")
        names = [m.name for m in u32_impl.methods]
        assert "U32__add" in names
        assert "U32__sub" in names
        assert "U32__mul" in names
        assert "U32__div" in names
        assert "U32__cmp" in names
        assert "U32__copy" in names
        assert "U32__shl" in names
        assert "U32__shr" in names


# =============================================================================
# Method Calls
# =============================================================================

class TestU32ConversionMethods:
    """U32 conversion and copy methods."""

    def test_from_u16(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
fn test() { V.from_u16(1000); }
""", cfg_features=["snes"])

    def test_to_u16(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
#[zeropage]
static mut R: u16;
fn test() { A = V.to_u16(); R = A; }
""", cfg_features=["snes"])

    def test_copy(self):
        build_and_check(HEADER + """
#[zeropage]
static mut DST: U32;
#[zeropage]
static mut SRC: far *U32;
fn test() { DST.copy(SRC); }
""", cfg_features=["snes"])


class TestU32ArithmeticMethods:
    """U32 arithmetic methods."""

    def test_add(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
fn test() { A_VAL.add(B_PTR); }
""", cfg_features=["snes"])

    def test_sub(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
fn test() { A_VAL.sub(B_PTR); }
""", cfg_features=["snes"])

    def test_mul(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
fn test() { A_VAL.mul(B_PTR); }
""", cfg_features=["snes"])

    def test_div(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
fn test() { A_VAL.div(B_PTR); }
""", cfg_features=["snes"])

    def test_mod(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
fn test() { A_VAL.mod(B_PTR); }
""", cfg_features=["snes"])


class TestU32ComparisonMethod:
    """U32 comparison method."""

    def test_cmp_returns_u8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_VAL: U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut FLAGS: u8;
fn test() { A = A_VAL.cmp(B_PTR); FLAGS = A; }
""", cfg_features=["snes"])


class TestU32ShiftMethods:
    """U32 shift methods."""

    def test_shl(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
fn test() { V.shl(4); }
""", cfg_features=["snes"])

    def test_shr(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
fn test() { V.shr(4); }
""", cfg_features=["snes"])


class TestU32HardwareMethods:
    """U32 SNES hardware-accelerated methods."""

    def test_div_u8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
#[zeropage]
static mut REM: u8;
fn test() { A = V.div_u8(10); REM = A; }
""", cfg_features=["snes"])

    def test_mod_u8(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
fn test() { V.mod_u8(10); }
""", cfg_features=["snes"])


# =============================================================================
# Macros
# =============================================================================

class TestU32Macros:
    """U32 convenience macros."""

    def test_u32_add(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut RESULT: U32;
fn test() { u32_add!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_u32_sub(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut RESULT: U32;
fn test() { u32_sub!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_u32_mul(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut RESULT: U32;
fn test() { u32_mul!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_u32_div(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut RESULT: U32;
fn test() { u32_div!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])

    def test_u32_mod(self):
        build_and_check(HEADER + """
#[zeropage]
static mut A_PTR: far *U32;
#[zeropage]
static mut B_PTR: far *U32;
#[zeropage]
static mut RESULT: U32;
fn test() { u32_mod!(RESULT, A_PTR, B_PTR); }
""", cfg_features=["snes"])


# =============================================================================
# Usage Patterns
# =============================================================================

class TestU32UsagePatterns:
    """Common U32 usage patterns."""

    def test_init_and_accumulate(self):
        build_and_check(HEADER + """
#[zeropage]
static mut COUNTER: U32;
#[zeropage]
static mut INCREMENT: U32;
#[zeropage]
static mut INC_PTR: far *U32;
fn init() {
    COUNTER.from_u16(0 as u16);
    INCREMENT.from_u16(1 as u16);
}
fn tick() { COUNTER.add(INC_PTR); }
""", cfg_features=["snes"])

    def test_method_via_pointer(self):
        build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
#[zeropage]
static mut PTR: far *U32;
#[zeropage]
static mut R: u16;
fn test() { A = PTR.to_u16(); R = A; }
""", cfg_features=["snes"])


# =============================================================================
# Error Cases
# =============================================================================

class TestU32Errors:
    """U32 type errors."""

    def test_wrong_argument_type(self):
        """Passing far *u16 where far *U32 expected fails."""
        with pytest.raises(TypeCheckError):
            build_and_check(HEADER + """
#[zeropage]
static mut V: U32;
#[zeropage]
static mut BAD: far *u16;
fn test() { V.add(BAD); }
""", cfg_features=["snes"])

    def test_near_pointer_to_far_impl(self):
        """Using near pointer with far impl method fails."""
        with pytest.raises(TypeCheckError):
            build_and_check(HEADER + """
#[zeropage]
static mut PTR: *U32;
fn test() -> u16 { return PTR.to_u16(); }
""", cfg_features=["snes"])
