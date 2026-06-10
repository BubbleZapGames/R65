# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for operator overloading desugaring (Tier A: compound assignment)."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.frontend.preprocessor import preprocess
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError


def build_and_check(source: str):
    program = parse(source, "test.r65")
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


def compile_to_asm(source: str) -> str:
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.codegen.codegen import ProgramCodeGenerator
    program = parse(source, "test.r65")
    program = preprocess(program, "test.r65")
    program = expand_macros(program)
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    mir_prog = MIRBuilder().build_program(hir_prog)
    return ProgramCodeGenerator().generate(mir_prog)


def _prog(trait, method, op):
    return f'''
        struct V {{ a: u8 }}
        impl {trait} for V {{
            fn {method}(*self, other: *V) {{ self.a = self.a; }}
        }}
        #[lowram] static mut P: V;
        #[lowram] static mut Q: V;
        fn main() {{ P {op}= Q; }}
    '''


class TestCompoundAssignOverload:
    @pytest.mark.parametrize("trait,method,op", [
        ("AddAssign", "add_assign", "+"),
        ("SubAssign", "sub_assign", "-"),
        ("MulAssign", "mul_assign", "*"),
        ("DivAssign", "div_assign", "/"),
        ("BitAndAssign", "bitand_assign", "&"),
        ("BitOrAssign", "bitor_assign", "|"),
        ("BitXorAssign", "bitxor_assign", "^"),
        ("ShlAssign", "shl_assign", "<<"),
        ("ShrAssign", "shr_assign", ">>"),
    ])
    def test_operator_resolves_to_method(self, trait, method, op):
        asm = compile_to_asm(_prog(trait, method, op))
        assert f"V__{method}" in asm

    def test_mul_assign_skips_power_of_2_restriction(self):
        # `*=` on an aggregate must not require a power-of-2 operand.
        build_and_check(_prog("MulAssign", "mul_assign", "*"))

    def test_missing_impl_rejected(self):
        with pytest.raises(TypeCheckError, match="does not implement"):
            build_and_check('''
                struct V { a: u8 }
                #[lowram] static mut P: V;
                #[lowram] static mut Q: V;
                fn main() { P += Q; }
            ''')


_CMP_IMPLS = '''
    struct C { v: u8 }
    impl PartialEq for C { fn eq(*self, other: *C) -> bool { return self.v == other.v; } }
    impl PartialOrd for C {
        fn cmp(*self, other: *C) -> i8 {
            if self.v < other.v { return -1; }
            if self.v > other.v { return 1; }
            return 0;
        }
    }
    #[lowram] static mut P: C;
    #[lowram] static mut Q: C;
    #[zeropage(0x10)] static mut R: u8;
'''


class TestComparisonOverload:
    @pytest.mark.parametrize("op,method", [
        ("==", "C__eq"), ("!=", "C__eq"),
        ("<", "C__cmp"), ("<=", "C__cmp"), (">", "C__cmp"), (">=", "C__cmp"),
    ])
    def test_comparison_dispatches(self, op, method):
        asm = compile_to_asm(_CMP_IMPLS + f'''
            fn main() {{ if P {op} Q {{ R = 1; }} else {{ R = 0; }} }}
        ''')
        assert method in asm

    def test_missing_partial_ord_rejected(self):
        with pytest.raises(TypeCheckError, match="does not implement '<'"):
            build_and_check('''
                struct V { x: u8 }
                impl PartialEq for V { fn eq(*self, o: *V) -> bool { return true; } }
                #[lowram] static mut P: V;
                #[lowram] static mut Q: V;
                fn main() { if P < Q { P.x = 1; } }
            ''')

    def test_missing_partial_eq_rejected(self):
        with pytest.raises(TypeCheckError, match="does not implement '=='"):
            build_and_check('''
                struct V { x: u8 }
                #[lowram] static mut P: V;
                #[lowram] static mut Q: V;
                fn main() { if P == Q { P.x = 1; } }
            ''')


class TestPrimitiveCompoundAssignUnchanged:
    def test_primitive_plus_equals_still_works(self):
        asm = compile_to_asm('''
            #[zeropage(0x10)] static mut C: u8;
            fn main() { C = 5; C += 3; }
        ''')
        assert ("INC" in asm) or ("ADC" in asm)

    def test_primitive_star_equals_nonpow2_still_rejected(self):
        # Regression: the power-of-2 rule still applies to primitive `*=`.
        with pytest.raises(TypeCheckError):
            build_and_check('''
                #[zeropage(0x10)] static mut C: u8;
                fn main() { C = 5; C *= 3; }
            ''')


_ADD_IMPL = '''
    struct V { a: u8 }
    impl AddAssign for V { fn add_assign(*self, other: *V) { self.a = self.a; } }
    #[lowram] static mut P: V;
    #[lowram] static mut Q: V;
    #[zeropage(0x10)] static mut N: u16;
'''


class TestCompoundAssignWrongRhs:
    """E-OVL-002: `a OP= b` where b is not the same struct as a."""

    def test_scalar_literal_rhs_rejected(self):
        # `P += 5` — the leaky "cannot take address of non-lvalue" must NOT surface.
        with pytest.raises(TypeCheckError, match="expects a 'V' operand"):
            build_and_check(_ADD_IMPL + 'fn main() { P += 5; }')

    def test_scalar_variable_rhs_rejected(self):
        # `P += N` where N is a u16 — the exact "add a scalar to a wide int" mistake.
        with pytest.raises(TypeCheckError, match="expects a 'V' operand"):
            build_and_check(_ADD_IMPL + 'fn main() { P += N; }')

    def test_error_names_both_types(self):
        try:
            build_and_check(_ADD_IMPL + 'fn main() { P += N; }')
            assert False, "expected TypeCheckError"
        except TypeCheckError as e:
            msg = str(e)
            assert "'V'" in msg and "u16" in msg          # expected vs found
            assert "add_assign" not in msg                # no desugar-internal leak

    def test_matching_struct_rhs_accepted(self):
        build_and_check(_ADD_IMPL + 'fn main() { P += Q; }')


class TestValueProducingOperatorsRejected:
    """E-OVL-003: value-producing arithmetic on aggregates is not overloadable."""

    @pytest.mark.parametrize("op", ['+', '-', '*', '/', '&', '|', '^', '<<', '>>'])
    def test_value_binary_op_rejected(self, op):
        # `R = P OP Q` must report the operator, not "cannot assign by value".
        # `*`,`/`,`<<`,`>>` also prove the power-of-2 / constant-shift validator
        # does not mask the aggregate diagnostic.
        with pytest.raises(TypeCheckError, match="not overloadable for aggregate"):
            build_and_check(
                _ADD_IMPL + f'#[lowram] static mut R: V;\nfn main() {{ R = P {op} Q; }}')
