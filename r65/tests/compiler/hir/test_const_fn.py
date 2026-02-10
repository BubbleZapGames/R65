"""
Tests for const fn support.

Tests parsing, const evaluation, compile-time folding, operator restrictions,
iteration limits, and error reporting for const fn.
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

    def test_parse_const_fn(self):
        """const fn should parse with is_const=True."""
        ast_prog = parse_program("const fn foo(x: u8) -> u8 { return x; }")
        func = ast_prog.items[0]
        assert func.is_const is True
        assert func.name == "foo"
        assert func.is_far is False

    def test_parse_regular_fn_not_const(self):
        """Regular fn should have is_const=False."""
        ast_prog = parse_program("fn foo(x: u8) -> u8 { return x; }")
        func = ast_prog.items[0]
        assert func.is_const is False

    def test_parse_const_far_fn(self):
        """const far fn should parse with both flags."""
        ast_prog = parse_program("const far fn foo() -> u8 { return 0; }")
        func = ast_prog.items[0]
        assert func.is_const is True
        assert func.is_far is True

    def test_parse_const_impl_method(self):
        """const fn in impl block should parse correctly."""
        ast_prog = parse_program("""
        struct Foo { x: u8 }
        impl Foo {
            const fn bar(*self) -> u8 { return 0; }
        }
        """)
        impl_decl = ast_prog.items[1]
        method = impl_decl.methods[0]
        assert method.is_const is True
        assert method.name == "bar"

    def test_hir_propagates_is_const(self):
        """HIR function declaration should have is_const from AST."""
        hir = build_hir("const fn double(x: u8) -> u8 { return x * 2; }")
        func = hir.declarations[0]
        assert isinstance(func, HIRFunctionDecl)
        assert func.is_const is True

    def test_hir_regular_fn_not_const(self):
        """HIR regular function should have is_const=False."""
        hir = build_hir("fn foo() { }")
        func = hir.declarations[0]
        assert func.is_const is False

    def test_array_size(self):
        """Const fn result used as array size."""
        source = """
        const fn buf_size() -> u16 { return 256; }
        static BUF: [u8; buf_size()] = [0; buf_size()];
        """
        build_hir(source)  # Should parse and build without error


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

    def test_let_bindings(self):
        """Const fn with let bindings."""
        source = """
        const fn calc(x: u8) -> u8 {
            let y: u8 = x + 1;
            let z: u8 = y * 2;
            return z;
        }
        const CALC: u8 = calc(5);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 12  # (5+1)*2

    def test_type_casts(self):
        """Const fn with type casts."""
        source = """
        const fn tile_offset(x: u8, y: u8) -> u16 {
            return (y as u16) * 32 + (x as u16);
        }
        const TILE: u16 = tile_offset(5, 3);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 3 * 32 + 5  # 101

    def test_if_else_both_branches(self):
        """Const fn with if/else evaluates correct branch."""
        source = """
        const fn max_val(a: u8, b: u8) -> u8 {
            if a > b { return a; } else { return b; }
        }
        const MAX1: u8 = max_val(10, 20);
        const MAX2: u8 = max_val(30, 20);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['MAX1'].evaluated_value == 20  # else branch
        assert decls['MAX2'].evaluated_value == 30  # then branch

    def test_while_loop(self):
        """Const fn with while loop (popcount)."""
        source = """
        const fn popcount(n: u8) -> u8 {
            let mut count: u8 = 0;
            let mut val: u8 = n;
            while val != 0 {
                count = count + 1;
                val = val & (val - 1);
            }
            return count;
        }
        const PCNT: u8 = popcount(0b10110100);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 4  # 0b10110100 has 4 bits set

    def test_for_loop(self):
        """Const fn with for loop."""
        source = """
        const fn sum_to(n: u8) -> u16 {
            let mut total: u16 = 0;
            for i in 0..n {
                total = total + (i as u16);
            }
            return total;
        }
        const SUM: u16 = sum_to(10);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 45  # 0+1+2+...+9

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

    def test_enum_variant_in_const_fn(self):
        """Const fn using enum variant value as argument."""
        source = """
        enum Dir { Up = 0, Down = 1, Left = 2, Right = 3 }
        const fn opposite(d: u8) -> u8 {
            if d == 0 { return 1; }
            if d == 1 { return 0; }
            if d == 2 { return 3; }
            return 2;
        }
        const OPP: u8 = opposite(Dir::Up as u8);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 1

    def test_const_fn_calling_const_fn(self):
        """Const fn calling another const fn, including forward references."""
        source = """
        const fn quadruple(x: u8) -> u8 { return double(double(x)); }
        const fn double(x: u8) -> u8 { return x * 2; }
        const QUAD: u8 = quadruple(3);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 12  # 3*2*2

    def test_three_level_nesting(self):
        """Three levels of const fn nesting with shared helpers."""
        source = """
        const fn add_one(x: u8) -> u8 { return x + 1; }
        const fn double_plus_one(x: u8) -> u8 { return add_one(x * 2); }
        const fn transform(x: u8) -> u8 { return double_plus_one(add_one(x)); }
        const RESULT: u8 = transform(5);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 13  # add_one(5)=6, 6*2=12, add_one(12)=13

    def test_const_fn_called_in_multiple_consts(self):
        """Same const fn called from multiple const declarations (also tests bitwise ops)."""
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

    def test_match_literal_patterns(self):
        """Const fn with match expression using literal patterns."""
        source = """
        const fn describe(n: u8) -> u8 {
            return match n {
                0 => 10,
                1 => 20,
                2 => 30,
                _ => 255,
            };
        }
        const A0: u8 = describe(0);
        const A1: u8 = describe(1);
        const A2: u8 = describe(2);
        const A3: u8 = describe(99);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['A0'].evaluated_value == 10
        assert decls['A1'].evaluated_value == 20
        assert decls['A2'].evaluated_value == 30
        assert decls['A3'].evaluated_value == 255

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

    def test_match_or_patterns(self):
        """Const fn with or-patterns in match arms."""
        source = """
        const fn classify(n: u8) -> u8 {
            return match n {
                0 | 1 => 0,
                2 | 3 | 4 => 1,
                _ => 2,
            };
        }
        const C0: u8 = classify(1);
        const C1: u8 = classify(3);
        const C2: u8 = classify(10);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['C0'].evaluated_value == 0
        assert decls['C1'].evaluated_value == 1
        assert decls['C2'].evaluated_value == 2

    def test_match_identifier_pattern(self):
        """Const fn with identifier pattern binding in match."""
        source = """
        const fn transform(n: u8) -> u8 {
            return match n {
                0 => 100,
                x => x * 2,
            };
        }
        const T0: u8 = transform(0);
        const T5: u8 = transform(5);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['T0'].evaluated_value == 100
        assert decls['T5'].evaluated_value == 10

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
        const fn compute(a: u8, b: u8) -> u16 {
            return (a as u16) * (b as u16);
        }
        const fn divide_and_mod(a: u16, b: u16) -> u16 {
            return (a / b) + (a % b);
        }
        const fn shift(val: u16, amt: u16) -> u16 {
            return val << amt;
        }
        const MUL: u16 = compute(7, 9);
        const DIVMOD: u16 = divide_and_mod(100, 7);
        const SHIFTED: u16 = shift(1, 10);
        """
        hir = build_and_typecheck(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['MUL'].evaluated_value == 63       # 7*9
        assert decls['DIVMOD'].evaluated_value == 16     # 100/7=14, 100%7=2, 14+2=16
        assert decls['SHIFTED'].evaluated_value == 1024  # 1<<10

    def test_complex_math(self):
        """Const fn with complex arithmetic combining multiple unrestricted operators."""
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
        const BRIGHT: u8 = brightness(100, 200, 150);
        const ADDR: u16 = tile_addr(5, 3, 64);
        """
        hir = build_and_typecheck(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['BRIGHT'].evaluated_value == 150  # (100+200+150)/3
        assert decls['ADDR'].evaluated_value == 323    # 5*64 + 3


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

    def test_non_const_fn_in_const_context(self):
        """Regular fn called in const context should error."""
        source = """
        fn bar(x: u8) -> u8 { return x; }
        const VAL: u8 = bar(5);
        """
        with pytest.raises(HIRError, match="not a const fn"):
            build_hir(source)

    def test_wrong_arg_count(self):
        """Wrong argument count should error."""
        source = """
        const fn add(a: u8, b: u8) -> u8 { return a + b; }
        const VAL: u8 = add(5);
        """
        with pytest.raises(HIRError, match="expects 2"):
            build_hir(source)

    def test_division_by_zero(self):
        """Division by zero in const fn should give clear error."""
        source = """
        const fn bad() -> u8 { return 10 / 0; }
        const VAL: u8 = bad();
        """
        with pytest.raises(HIRError):
            build_hir(source)

    def test_invalid_body_caught_at_definition(self):
        """Const fn with invalid body errors at definition time, even if only called at runtime."""
        # Non-const fn call
        with pytest.raises(HIRError, match="'helper' is not a const fn"):
            build_hir("""
            fn helper(x: u8) -> u8 { return x + 1; }
            const fn bad(x: u8) -> u8 { return helper(x); }
            fn main() { A = bad(5); }
            """)
        # Hardware register access
        with pytest.raises(HIRError, match="Cannot access hardware register"):
            build_hir("""
            const fn bad() -> u8 { return X; }
            fn main() { A = bad(); }
            """)

    def test_const_fn_accesses_static_mut(self):
        """Const fn accessing a static mut variable should error."""
        source = """
        #[ram]
        static mut COUNTER: u8 = 0;
        const fn bad() -> u8 { return COUNTER; }
        const VAL: u8 = bad();
        """
        with pytest.raises(HIRError, match="Cannot access runtime variable 'COUNTER'"):
            build_hir(source)

    def test_const_fn_accesses_hardware_register(self):
        """Const fn accessing a hardware register should error."""
        source = """
        const fn bad() -> u8 { return A; }
        const VAL: u8 = bad();
        """
        with pytest.raises(HIRError, match="Cannot access hardware register 'A'"):
            build_hir(source)

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
