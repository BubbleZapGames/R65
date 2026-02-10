"""
Tests for const fn support.

Tests parsing, const evaluation, compile-time folding, operator restrictions,
match expressions, iteration limits, and error reporting for const fn.
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError, HIRProgram,
    HIRFunctionDecl, HIRConstDecl, HIRIntegerLiteral, HIRBooleanLiteral,
    HIRFunctionCall,
)
from r65.compiler.typeck import TypeChecker


def parse_program(source: str):
    """Helper to parse source to AST."""
    parser = Parser()
    return parser.parse(source)


def build_hir(source: str) -> HIRProgram:
    """Helper to parse and build HIR from source."""
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    return builder.build_program(ast_prog)


def build_and_typecheck(source: str) -> HIRProgram:
    """Helper to parse, build HIR, and type check."""
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    hir = builder.build_program(ast_prog)
    tc = TypeChecker(hir)
    tc.check()
    return hir


class TestConstFnParsing:
    """Test parsing and HIR propagation of const fn declarations."""

    def test_parse_const_fn_flags(self):
        """const/far flags parse correctly on function declarations."""
        # Regular fn
        ast_prog = parse_program("fn foo(x: u8) -> u8 { return x; }")
        assert ast_prog.items[0].is_const is False
        assert ast_prog.items[0].is_far is False

        # const fn
        ast_prog = parse_program("const fn foo(x: u8) -> u8 { return x; }")
        assert ast_prog.items[0].is_const is True
        assert ast_prog.items[0].is_far is False

        # const far fn
        ast_prog = parse_program("const far fn foo() -> u8 { return 0; }")
        assert ast_prog.items[0].is_const is True
        assert ast_prog.items[0].is_far is True

    def test_parse_const_impl_method(self):
        """const fn in impl block should parse correctly."""
        ast_prog = parse_program("""
        struct Foo { x: u8 }
        impl Foo {
            const fn bar(*self) -> u8 { return 0; }
        }
        """)
        method = ast_prog.items[1].methods[0]
        assert method.is_const is True
        assert method.name == "bar"

    def test_hir_propagates_is_const(self):
        """HIR function declarations preserve is_const flag."""
        hir = build_hir("const fn double(x: u8) -> u8 { return x * 2; }")
        func = hir.declarations[0]
        assert isinstance(func, HIRFunctionDecl)
        assert func.is_const is True

        hir = build_hir("fn foo() { }")
        assert hir.declarations[0].is_const is False

    def test_array_size(self):
        """Const fn result used as array size."""
        build_hir("""
        const fn buf_size() -> u16 { return 256; }
        static BUF: [u8; buf_size()] = [0; buf_size()];
        """)


class TestConstFnEvaluation:
    """Test const fn evaluation in const contexts."""

    def test_parameter_arithmetic(self):
        """Const fn with parameters and arithmetic."""
        source = """
        const fn double(x: u8) -> u8 { return x * 2; }
        const fn add(a: u8, b: u8) -> u8 { return a + b; }
        const DOUBLED: u8 = double(5);
        const SUM: u8 = add(10, 20);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['DOUBLED'].evaluated_value == 10
        assert decls['SUM'].evaluated_value == 30

    def test_let_bindings_and_casts(self):
        """Const fn with let bindings and type casts."""
        source = """
        const fn calc(x: u8) -> u8 {
            let y: u8 = x + 1;
            let z: u8 = y * 2;
            return z;
        }
        const fn tile_offset(x: u8, y: u8) -> u16 {
            return (y as u16) * 32 + (x as u16);
        }
        const CALC: u8 = calc(5);
        const TILE: u16 = tile_offset(5, 3);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['CALC'].evaluated_value == 12    # (5+1)*2
        assert decls['TILE'].evaluated_value == 101   # 3*32 + 5

    def test_control_flow(self):
        """Const fn with if/else, while, and for loops."""
        source = """
        const fn max_val(a: u8, b: u8) -> u8 {
            if a > b { return a; } else { return b; }
        }
        const fn popcount(n: u8) -> u8 {
            let mut count: u8 = 0;
            let mut val: u8 = n;
            while val != 0 {
                count = count + 1;
                val = val & (val - 1);
            }
            return count;
        }
        const fn sum_to(n: u8) -> u16 {
            let mut total: u16 = 0;
            for i in 0..n {
                total = total + (i as u16);
            }
            return total;
        }
        const MAX1: u8 = max_val(10, 20);
        const MAX2: u8 = max_val(30, 20);
        const PCNT: u8 = popcount(0b10110100);
        const SUM: u16 = sum_to(10);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['MAX1'].evaluated_value == 20  # else branch
        assert decls['MAX2'].evaluated_value == 30  # then branch
        assert decls['PCNT'].evaluated_value == 4   # 4 bits set
        assert decls['SUM'].evaluated_value == 45   # 0+1+...+9

    def test_const_referencing_other_const(self):
        """Const fn body referencing a const declaration."""
        source = """
        const MULTIPLIER: u8 = 3;
        const fn scale(x: u8) -> u8 { return x * MULTIPLIER; }
        const SCALED: u8 = scale(10);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'SCALED'][0]
        assert const_decl.evaluated_value == 30

    def test_const_fn_calling_const_fn(self):
        """Const fn cross-calling with forward references and multi-level nesting."""
        source = """
        const fn quadruple(x: u8) -> u8 { return double(double(x)); }
        const fn double(x: u8) -> u8 { return x * 2; }
        const fn add_one(x: u8) -> u8 { return x + 1; }
        const fn double_plus_one(x: u8) -> u8 { return add_one(x * 2); }
        const fn transform(x: u8) -> u8 { return double_plus_one(add_one(x)); }
        const QUAD: u8 = quadruple(3);
        const XFORM: u8 = transform(5);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['QUAD'].evaluated_value == 12    # 3*2*2
        assert decls['XFORM'].evaluated_value == 13   # add_one(5)=6, 6*2=12, add_one(12)=13

    def test_const_fn_called_in_multiple_consts(self):
        """Same const fn called from multiple const declarations."""
        source = """
        const fn make_mask(bit: u8) -> u8 { return 1 << bit; }
        const MASK0: u8 = make_mask(0);
        const MASK3: u8 = make_mask(3);
        const MASK7: u8 = make_mask(7);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['MASK0'].evaluated_value == 1
        assert decls['MASK3'].evaluated_value == 8
        assert decls['MASK7'].evaluated_value == 128

    def test_match_patterns(self):
        """Const fn with match: literal, or-pattern, identifier, and wildcard patterns."""
        source = """
        const fn describe(n: u8) -> u8 {
            return match n {
                0 => 10,
                1 => 20,
                2 | 3 | 4 => 30,
                x => x * 2,
            };
        }
        const D0: u8 = describe(0);
        const D1: u8 = describe(1);
        const D3: u8 = describe(3);
        const D10: u8 = describe(10);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['D0'].evaluated_value == 10   # literal 0
        assert decls['D1'].evaluated_value == 20   # literal 1
        assert decls['D3'].evaluated_value == 30   # or-pattern 2|3|4
        assert decls['D10'].evaluated_value == 20  # identifier x => x*2

    def test_match_enum_patterns(self):
        """Const fn with match on enum variants."""
        source = """
        enum Dir { Up = 0, Down = 1, Left = 2, Right = 3 }
        const fn opposite(d: u8) -> u8 {
            return match d {
                Dir::Up => Dir::Down as u8,
                Dir::Down => Dir::Up as u8,
                Dir::Left => Dir::Right as u8,
                Dir::Right => Dir::Left as u8,
                _ => 255,
            };
        }
        const OPP: u8 = opposite(Dir::Up as u8);
        const OPP2: u8 = opposite(Dir::Left as u8);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['OPP'].evaluated_value == 1   # Down
        assert decls['OPP2'].evaluated_value == 3   # Right

    def test_match_in_const_declaration(self):
        """Match expression used directly in const declaration (not inside const fn)."""
        source = """
        const MODE: u8 = 2;
        const RESULT: u8 = match MODE {
            0 => 10,
            1 => 20,
            2 => 30,
            _ => 0,
        };
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['RESULT'].evaluated_value == 30

    def test_unrestricted_operators(self):
        """Const fn bypasses runtime operator restrictions (*, /, %, <<, >> with any operands)."""
        source = """
        const fn clamp_byte(x: u16) -> u8 {
            if x > 255 { return 255; }
            return x as u8;
        }
        const fn brightness(r: u8, g: u8, b: u8) -> u8 {
            return clamp_byte((r as u16 + g as u16 + b as u16) / 3);
        }
        const fn tile_addr(row: u8, col: u8, stride: u8) -> u16 {
            return (row as u16) * (stride as u16) + (col as u16);
        }
        const fn shift(val: u16, amt: u16) -> u16 {
            return val << amt;
        }
        const BRIGHT: u8 = brightness(100, 200, 150);
        const ADDR: u16 = tile_addr(5, 3, 64);
        const SHIFTED: u16 = shift(1, 10);
        """
        hir = build_and_typecheck(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['BRIGHT'].evaluated_value == 150   # (100+200+150)/3
        assert decls['ADDR'].evaluated_value == 323     # 5*64 + 3
        assert decls['SHIFTED'].evaluated_value == 1024  # 1<<10


class TestConstFnFolding:
    """Test const fn calls being folded to literals in expression context."""

    def test_const_fold_in_function_body(self):
        """Const fn call with literal args should be folded to integer literal."""
        source = """
        const fn offset(x: u8) -> u8 { return x * 4; }
        fn foo() {
            A = offset(3);
        }
        """
        hir = build_hir(source)
        func = [d for d in hir.declarations if isinstance(d, HIRFunctionDecl) and d.name == 'foo'][0]
        assign = func.body.statements[0].expr  # HIRAssignment
        assert isinstance(assign.value, HIRIntegerLiteral)
        assert assign.value.value == 12

    def test_runtime_fallback_with_non_const_args(self):
        """Const fn called with non-const args should emit a runtime call."""
        source = """
        const fn double(x: u8) -> u8 { return x * 2; }
        fn foo(val @ A: u8) -> u8 {
            return double(val);
        }
        """
        hir = build_hir(source)
        func = [d for d in hir.declarations if isinstance(d, HIRFunctionDecl) and d.name == 'foo'][0]
        ret_stmt = func.body.statements[0]
        assert isinstance(ret_stmt.values[0], HIRFunctionCall)


class TestConstFnErrors:
    """Test error cases for const fn."""

    def test_call_errors(self):
        """Errors for invalid const fn calls: non-const fn, wrong arg count, division by zero."""
        with pytest.raises(HIRError, match="not a const fn"):
            build_hir("""
            fn bar(x: u8) -> u8 { return x; }
            const VAL: u8 = bar(5);
            """)
        with pytest.raises(HIRError, match="expects 2"):
            build_hir("""
            const fn add(a: u8, b: u8) -> u8 { return a + b; }
            const VAL: u8 = add(5);
            """)
        with pytest.raises(HIRError):
            build_hir("""
            const fn bad() -> u8 { return 10 / 0; }
            const VAL: u8 = bad();
            """)

    def test_invalid_body_caught_at_definition(self):
        """Invalid const fn bodies error at definition time: non-const call, registers, static mut, asm."""
        with pytest.raises(HIRError, match="'helper' is not a const fn"):
            build_hir("""
            fn helper(x: u8) -> u8 { return x + 1; }
            const fn bad(x: u8) -> u8 { return helper(x); }
            fn main() { A = bad(5); }
            """)
        with pytest.raises(HIRError, match="Cannot access hardware register"):
            build_hir("""
            const fn bad() -> u8 { return X; }
            fn main() { A = bad(); }
            """)
        with pytest.raises(HIRError, match="Cannot access runtime variable 'COUNTER'"):
            build_hir("""
            #[ram]
            static mut COUNTER: u8 = 0;
            const fn bad() -> u8 { return COUNTER; }
            """)
        with pytest.raises(HIRError, match="Inline assembly.*cannot be used in const fn"):
            build_hir("""
            const fn bad() -> u8 { asm!("NOP"); return 0; }
            """)

    def test_unsupported_expressions(self):
        """Unsupported expression types give meaningful error messages."""
        with pytest.raises(HIRError, match="Array indexing is not supported in const fn"):
            build_hir("""
            #[ram]
            static mut BUF: [u8; 4] = [0; 4];
            const fn bad() -> u8 { return BUF[0]; }
            """)
        with pytest.raises(HIRError, match="Pointer dereference is not supported in const fn"):
            build_hir("""
            const fn bad(p: *u8) -> u8 { return *p; }
            """)
        with pytest.raises(HIRError, match="Address-of operator is not supported in const fn"):
            build_hir("""
            #[ram]
            static mut VAL: u8 = 0;
            const fn bad() -> u16 { return &VAL as u16; }
            """)

    def test_infinite_loop_caught(self):
        """Infinite loops (while, loop, nested) are caught by iteration limit."""
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir("""
            const fn hang() -> u8 { while true { } return 0; }
            const VAL: u8 = hang();
            """)
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir("""
            const fn hang() -> u8 { loop { } return 0; }
            const VAL: u8 = hang();
            """)
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir("""
            const fn hang() -> u8 {
                let mut x: u8 = 0;
                while true { while true { x = x + 1; } }
                return x;
            }
            const VAL: u8 = hang();
            """)
