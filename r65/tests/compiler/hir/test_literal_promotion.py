# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for overflow-aware literal promotion.

When unsuffixed integer literals produce a compile-time result that overflows
the inferred type, the type checker should promote operand and result types
to u16/i16 automatically.
"""
from r65.compiler.frontend import Parser
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker


def build_and_typecheck(source: str, expand: bool = False):
    parser = Parser()
    ast = parser.parse(source)
    if expand:
        ast = expand_macros(ast)
    hir = HIRBuilder().build_program(ast)
    TypeChecker(hir).check()
    return hir


def main_rhs(hir):
    """Return the RHS expression of `<lhs> = <rhs>;` in main()."""
    for decl in hir.declarations:
        if getattr(decl, 'name', None) == 'main':
            return decl.body.statements[0].expr.value
    raise AssertionError("main not found")


class TestLiteralPromotion:
    """Unsuffixed-literal promotion driven by compile-time overflow."""

    def test_shift_overflow_promotes(self):
        """8 << 5 = 256 → overflows u8, both operand and result promote to u16."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;
            fn main() { VAL = 8 << 5; }
        ''')
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u16'
        assert rhs.left.expr_type.name == 'u16'

    def test_shift_no_overflow_stays_u8(self):
        """1 << 0 = 1 fits u8 → stays u8."""
        hir = build_and_typecheck("fn main() { A = 1 << 0; }")
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u8'
        assert rhs.left.expr_type.name == 'u8'

    def test_shift_max_u8_stays(self):
        """1 << 7 = 128 fits u8 → stays u8."""
        hir = build_and_typecheck("fn main() { A = 1 << 7; }")
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u8'
        assert rhs.left.expr_type.name == 'u8'

    def test_multiply_overflow_promotes(self):
        """32 * 32 = 1024 overflows u8 → promote to u16."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;
            fn main() { VAL = 32 * 32; }
        ''')
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u16'
        assert rhs.left.expr_type.name == 'u16'
        assert rhs.right.expr_type.name == 'u16'

    def test_add_overflow_promotes(self):
        """200 + 100 = 300 overflows u8 → promote to u16."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;
            fn main() { VAL = 200 + 100; }
        ''')
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u16'
        assert rhs.left.expr_type.name == 'u16'
        assert rhs.right.expr_type.name == 'u16'

    def test_nested_no_promotion(self):
        """(8 << 4) + 1 = 129 fits u8 at each step → stays u8."""
        hir = build_and_typecheck("fn main() { A = (8 << 4) + 1; }")
        rhs = main_rhs(hir)
        assert rhs.op == '+'
        assert rhs.expr_type.name == 'u8'
        assert rhs.left.op == '<<'
        assert rhs.left.expr_type.name == 'u8'
        assert rhs.right.expr_type.name == 'u8'

    def test_macro_shift_promotes(self):
        """Promotion still applies after macro expansion."""
        hir = build_and_typecheck('''
            #[zeropage]
            static mut VAL: u16;

            macro_rules! make_val($v:expr, $s:expr) {
                VAL = $v << $s;
            }

            fn main() { make_val!(8, 5); }
        ''', expand=True)
        rhs = main_rhs(hir)
        assert rhs.expr_type.name == 'u16'
        assert rhs.left.expr_type.name == 'u16'
