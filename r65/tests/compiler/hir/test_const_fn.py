"""
Tests for const fn support.

Tests parsing, const evaluation, and compile-time folding of const fn.
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError, HIRProgram,
    HIRFunctionDecl, HIRConstDecl, HIRIntegerLiteral, HIRBooleanLiteral,
    HIRFunctionCall,
)
from r65.compiler.typeck import TypeChecker


def build_hir(source: str) -> HIRProgram:
    """Helper to parse and build HIR from source."""
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    return builder.build_program(ast_prog)


def parse_program(source: str):
    """Helper to parse source to AST."""
    parser = Parser()
    return parser.parse(source)


class TestConstFnParsing:
    """Test parsing of const fn declarations."""

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


class TestConstFnEvaluation:
    """Test const fn evaluation in const contexts."""

    def test_simple_return(self):
        """Const fn returning a literal."""
        source = """
        const fn five() -> u8 { return 5; }
        const VAL: u8 = five();
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 5

    def test_parameter_arithmetic(self):
        """Const fn with parameter arithmetic."""
        source = """
        const fn double(x: u8) -> u8 { return x * 2; }
        const DOUBLED: u8 = double(5);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 10

    def test_two_parameters(self):
        """Const fn with two parameters."""
        source = """
        const fn add(a: u8, b: u8) -> u8 { return a + b; }
        const SUM: u8 = add(10, 20);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 30

    def test_if_else(self):
        """Const fn with if/else."""
        source = """
        const fn max_val(a: u8, b: u8) -> u8 {
            if a > b {
                return a;
            } else {
                return b;
            }
        }
        const MAX: u8 = max_val(10, 20);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 20

    def test_if_else_other_branch(self):
        """Const fn with if/else - other branch taken."""
        source = """
        const fn max_val(a: u8, b: u8) -> u8 {
            if a > b {
                return a;
            } else {
                return b;
            }
        }
        const MAX: u8 = max_val(30, 20);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 30

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

    def test_const_fn_calling_const_fn(self):
        """Const fn calling another const fn."""
        source = """
        const fn double(x: u8) -> u8 { return x * 2; }
        const fn quadruple(x: u8) -> u8 { return double(double(x)); }
        const QUAD: u8 = quadruple(3);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 12  # 3*2*2

    def test_const_fn_forward_reference(self):
        """Const fn calling another const fn defined after it (forward reference)."""
        source = """
        const fn quadruple(x: u8) -> u8 { return double(double(x)); }
        const fn double(x: u8) -> u8 { return x * 2; }
        const QUAD: u8 = quadruple(3);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 12  # 3*2*2

    def test_three_level_nesting(self):
        """Three levels of const fn nesting."""
        source = """
        const fn add_one(x: u8) -> u8 { return x + 1; }
        const fn double_plus_one(x: u8) -> u8 { return add_one(x * 2); }
        const fn transform(x: u8) -> u8 { return double_plus_one(add_one(x)); }
        const RESULT: u8 = transform(5);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 13  # add_one(5)=6, 6*2=12, add_one(12)=13

    def test_multiple_const_fns_shared_helper(self):
        """Multiple const fns sharing a common helper."""
        source = """
        const fn clamp_byte(x: u16) -> u8 {
            if x > 255 { return 255; }
            return x as u8;
        }
        const fn brightness(r: u8, g: u8, b: u8) -> u8 {
            return clamp_byte((r as u16 + g as u16 + b as u16) / 3);
        }
        const fn saturate(x: u8, boost: u8) -> u8 {
            return clamp_byte((x as u16) + (boost as u16));
        }
        const BRIGHT: u8 = brightness(100, 200, 150);
        const SAT: u8 = saturate(200, 100);
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['BRIGHT'].evaluated_value == 150  # (100+200+150)/3 = 150
        assert decls['SAT'].evaluated_value == 255  # clamped to 255

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

    def test_bitwise_operations(self):
        """Const fn with bitwise operations."""
        source = """
        const fn make_mask(bit: u8) -> u8 {
            return 1 << bit;
        }
        const MASK: u8 = make_mask(5);
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 32  # 1 << 5

    def test_no_params(self):
        """Const fn with no parameters."""
        source = """
        const fn magic() -> u8 { return 42; }
        const MAGIC: u8 = magic();
        """
        hir = build_hir(source)
        const_decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert const_decl.evaluated_value == 42


class TestConstFnArraySize:
    """Test const fn used in array size context."""

    def test_array_size(self):
        """Const fn result used as array size."""
        source = """
        const fn buf_size() -> u16 { return 256; }
        static BUF: [u8; buf_size()] = [0; buf_size()];
        """
        hir = build_hir(source)
        # Should parse and build without error


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
        # The assignment value should be folded to a literal
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
        # The return value should be a function call (not folded)
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

    def test_division_by_zero(self):
        """Division by zero in const fn should give clear error."""
        source = """
        const fn bad() -> u8 {
            return 10 / 0;
        }
        const VAL: u8 = bad();
        """
        with pytest.raises(HIRError):
            build_hir(source)

    def test_wrong_arg_count(self):
        """Wrong argument count should error."""
        source = """
        const fn add(a: u8, b: u8) -> u8 { return a + b; }
        const VAL: u8 = add(5);
        """
        with pytest.raises(HIRError, match="expects 2"):
            build_hir(source)

    def test_const_fn_calls_non_const_fn(self):
        """Const fn calling a non-const fn should give clear error."""
        source = """
        fn helper(x: u8) -> u8 { return x + 1; }
        const fn bad(x: u8) -> u8 { return helper(x); }
        const VAL: u8 = bad(5);
        """
        with pytest.raises(HIRError, match="'helper' is not a const fn"):
            build_hir(source)

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

    def test_invalid_const_fn_body_caught_at_definition(self):
        """Const fn with invalid body should error even if only called at runtime."""
        source = """
        fn helper(x: u8) -> u8 { return x + 1; }
        const fn bad(x: u8) -> u8 { return helper(x); }
        fn main() {
            A = bad(5);
        }
        """
        with pytest.raises(HIRError, match="'helper' is not a const fn"):
            build_hir(source)

    def test_const_fn_register_in_body_caught_at_definition(self):
        """Const fn using register in body should error at definition time."""
        source = """
        const fn bad() -> u8 { return X; }
        fn main() {
            A = bad();
        }
        """
        with pytest.raises(HIRError, match="Cannot access hardware register 'X'"):
            build_hir(source)

    def test_infinite_while_loop(self):
        """Const fn with infinite while loop should error."""
        source = """
        const fn hang() -> u8 {
            while true { }
            return 0;
        }
        const VAL: u8 = hang();
        """
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir(source)

    def test_infinite_loop(self):
        """Const fn with infinite loop should error."""
        source = """
        const fn hang() -> u8 {
            loop { }
            return 0;
        }
        const VAL: u8 = hang();
        """
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir(source)

    def test_nested_infinite_loops(self):
        """Nested infinite loops share a single counter and get caught."""
        source = """
        const fn hang() -> u8 {
            let mut x: u8 = 0;
            while true {
                while true { x = x + 1; }
            }
            return x;
        }
        const VAL: u8 = hang();
        """
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir(source)


class TestConstFnImplMethod:
    """Test const fn on impl methods."""

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


def build_and_typecheck(source: str) -> HIRProgram:
    """Helper to parse, build HIR, and type check."""
    parser = Parser()
    ast_prog = parser.parse(source)
    builder = HIRBuilder()
    hir = builder.build_program(ast_prog)
    tc = TypeChecker(hir)
    tc.check()
    return hir


class TestConstFnUnrestrictedOperators:
    """Test that const fn bodies allow full multiply, divide, modulo, and shift."""

    def test_arbitrary_multiply(self):
        """Const fn can multiply by any value, not just 1/2/4/8."""
        source = """
        const fn multiply(a: u8, b: u8) -> u16 {
            return (a as u16) * (b as u16);
        }
        const RESULT: u16 = multiply(7, 9);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 63

    def test_multiply_by_non_power_of_two(self):
        """Const fn can multiply by 3, 5, 7, etc."""
        source = """
        const fn triple(x: u8) -> u16 {
            return (x as u16) * 3;
        }
        const RESULT: u16 = triple(10);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 30

    def test_arbitrary_divide(self):
        """Const fn can divide by any value, not just 1/2/4/8."""
        source = """
        const fn divide(a: u16, b: u16) -> u16 {
            return a / b;
        }
        const RESULT: u16 = divide(100, 7);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 14

    def test_modulo(self):
        """Const fn can use modulo with any operands."""
        source = """
        const fn modulo(a: u16, b: u16) -> u16 {
            return a % b;
        }
        const RESULT: u16 = modulo(100, 7);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 2

    def test_variable_shift(self):
        """Const fn can shift by variable amounts."""
        source = """
        const fn shift_left(val: u16, amt: u16) -> u16 {
            return val << amt;
        }
        const RESULT: u16 = shift_left(1, 10);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 1024

    def test_complex_math(self):
        """Const fn with complex arithmetic using unrestricted operators."""
        source = """
        const fn tile_addr(row: u8, col: u8, stride: u8) -> u16 {
            return (row as u16) * (stride as u16) + (col as u16);
        }
        const ADDR: u16 = tile_addr(5, 3, 64);
        """
        hir = build_and_typecheck(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl)][0]
        assert decl.evaluated_value == 323  # 5*64 + 3
