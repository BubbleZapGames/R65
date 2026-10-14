# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for const fn support.

Tests parsing, const evaluation, compile-time folding, operator restrictions,
match expressions, iteration limits, and error reporting for const fn.
"""
import pytest

from r65.compiler.frontend import Parser
from r65.compiler.hir import (
    HIRBuilder, HIRError, HIRProgram,
    HIRFunctionDecl, HIRConstDecl, HIRStaticDecl,
    HIRIntegerLiteral, HIRBooleanLiteral, HIRArrayLiteralExpr,
    HIRFunctionCall,
)
from r65.compiler.main import compile_string
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

    def test_zero_arg_const_fn_called_from_const_fn(self):
        """A callee with no parameters transpiles without a dangling separator."""
        source = """
        const fn base() -> u8 { return 5; }
        const fn table() -> [u8; 3] { return [5, 6, 7]; }
        const fn shifted() -> u8 { return base() + 1; }
        const fn second() -> u8 { let t: [u8; 3] = table(); return t[1]; }
        const SHIFTED: u8 = shifted();
        const SECOND: u8 = second();
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['SHIFTED'].evaluated_value == 6
        assert decls['SECOND'].evaluated_value == 6

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
        with pytest.raises(HIRError, match="Cannot access runtime variable 'BUF'"):
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


class TestConstFnArrays:
    """Test const fn with array support."""

    def test_array_fill_and_return(self):
        """Const fn can create, fill, and return an array."""
        source = """
        const fn make_table() -> [u8; 4] {
            let mut t: [u8; 4] = [0; 4];
            t[0] = 10;
            t[1] = 20;
            t[2] = 30;
            t[3] = 40;
            return t;
        }
        const TABLE: [u8; 4] = make_table();
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'TABLE'][0]
        assert decl.evaluated_value == [10, 20, 30, 40]

    def test_array_literal_return(self):
        """Const fn can return an array literal directly."""
        source = """
        const fn palette() -> [u8; 3] {
            return [0xFF, 0x80, 0x00];
        }
        const PAL: [u8; 3] = palette();
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'PAL'][0]
        assert decl.evaluated_value == [0xFF, 0x80, 0x00]

    def test_array_index_read(self):
        """Const fn can read array elements by index."""
        source = """
        const fn first_element() -> u8 {
            let arr: [u8; 3] = [10, 20, 30];
            return arr[1];
        }
        const VAL: u8 = first_element();
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'VAL'][0]
        assert decl.evaluated_value == 20

    def test_array_loop_mutation(self):
        """Const fn can mutate array in a loop."""
        source = """
        const fn squares() -> [u8; 5] {
            let mut t: [u8; 5] = [0; 5];
            for i in 0..5 {
                t[i] = (i as u8) * (i as u8);
            }
            return t;
        }
        const SQ: [u8; 5] = squares();
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'SQ'][0]
        assert decl.evaluated_value == [0, 1, 4, 9, 16]

    def test_recursive_fibonacci_table(self):
        """Const fn fibonacci with recursive calls to generate table."""
        source = """
        const fn fibonacci(n: u8) -> u8 {
            if n <= 1 { return n; }
            let mut a: u8 = 0;
            let mut b: u8 = 1;
            for i in 2..n+1 {
                let tmp: u8 = b;
                b = a + b;
                a = tmp;
            }
            return b;
        }
        const fn generate_fib_table() -> [u8; 12] {
            let mut table: [u8; 12] = [0; 12];
            let mut i: u8 = 0;
            while i < 12 {
                table[i] = fibonacci(i);
                i = i + 1;
            }
            return table;
        }
        const FIB_TABLE: [u8; 12] = generate_fib_table();
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'FIB_TABLE'][0]
        assert decl.evaluated_value == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    def test_array_const_fn_as_static_initializer(self):
        """Const fn returning array used as static initializer folds to HIRArrayLiteralExpr."""
        source = """
        const fn make_data() -> [u8; 3] {
            return [1, 2, 3];
        }
        static DATA: [u8; 3] = make_data();
        """
        hir = build_hir(source)
        static_decl = [d for d in hir.declarations if isinstance(d, HIRStaticDecl) and d.name == 'DATA'][0]
        assert isinstance(static_decl.initializer, HIRArrayLiteralExpr)
        assert len(static_decl.initializer.elements) == 3
        assert [e.value for e in static_decl.initializer.elements] == [1, 2, 3]

    def test_array_return_type_checked(self):
        """Const fn with array return type passes type checking."""
        source = """
        const fn make_table() -> [u8; 4] {
            return [1, 2, 3, 4];
        }
        const TABLE: [u8; 4] = make_table();
        """
        build_and_typecheck(source)


class TestConstFnNewtypes:
    """Test const fn interaction with newtypes."""

    def test_construction_in_body(self):
        """`Newtype(x)` inside a const fn body is a retype, not a call."""
        source = """
        struct TileId(u8);
        const fn make(n: u8) -> TileId { return TileId(n + 1); }
        const T: TileId = make(5);
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'T'][0]
        assert decl.evaluated_value == 6

    def test_payload_access_in_body(self):
        """`t.0` inside a const fn body yields the payload."""
        source = """
        struct TileId(u8);
        const fn raw(t: TileId) -> u8 { return t.0; }
        const N: u8 = raw(TileId(7));
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'N'][0]
        assert decl.evaluated_value == 7

    def test_payload_access_in_const_expr(self):
        """`T.0` in a const declaration yields the payload."""
        source = """
        struct TileId(u8);
        const T: TileId = TileId(7);
        const N: u8 = T.0;
        """
        hir = build_hir(source)
        decl = [d for d in hir.declarations if isinstance(d, HIRConstDecl) and d.name == 'N'][0]
        assert decl.evaluated_value == 7

    def test_newtype_array_return(self):
        """Const fn returning an array of newtypes folds elementwise."""
        source = """
        struct TileId(u8);
        const fn listed() -> [TileId; 3] { return [TileId(1), TileId(2), TileId(3)]; }
        const fn filled() -> [TileId; 3] { return [TileId(7); 3]; }
        const LISTED: [TileId; 3] = listed();
        const FILLED: [TileId; 3] = filled();
        """
        hir = build_hir(source)
        decls = {d.name: d for d in hir.declarations if isinstance(d, HIRConstDecl)}
        assert decls['LISTED'].evaluated_value == [1, 2, 3]
        assert decls['FILLED'].evaluated_value == [7, 7, 7]

    def test_sixteen_bit_payload_element_width(self):
        """A 16-bit payload emits two bytes per element, sign included."""
        source = """
        struct Q10(i16);
        const fn table() -> [Q10; 3] { return [Q10(100), Q10(-2), Q10(400)]; }
        static TABLE: [Q10; 3] = table();
        """
        asm = compile_string(source, "test.r65")
        assert '.db $64, $00, $FE, $FF, $90, $01' in asm

    def test_bad_payload_index_rejected(self):
        """`.1` is rejected in both const positions — a newtype has only '.0'."""
        with pytest.raises(HIRError, match=r"only field '\.0'"):
            build_hir("""
            struct TileId(u8);
            const fn raw(t: TileId) -> u8 { return t.1; }
            const N: u8 = raw(TileId(3));
            """)
        with pytest.raises(HIRError, match=r"only field '\.0'"):
            build_hir("""
            struct TileId(u8);
            const T: TileId = TileId(7);
            const N: u8 = T.1;
            """)

    def test_construction_arity_checked_in_body(self):
        """A newtype takes exactly one value, inside a const fn body too."""
        with pytest.raises(HIRError, match="takes exactly 1 value"):
            build_hir("""
            struct TileId(u8);
            const fn make(n: u8) -> TileId { return TileId(n, 2); }
            const T: TileId = make(1);
            """)


def const_values(source: str) -> dict:
    """Every const's evaluated value, by name."""
    hir = build_hir(source)
    return {d.name: d.evaluated_value for d in hir.declarations if isinstance(d, HIRConstDecl)}


class TestConstFnOperators:
    """`*`, `/` and `%` are unrestricted in const fns and between constants."""

    def test_full_operators_in_const_fn(self):
        values = const_values("""
        const fn f(a: u8, b: u8) -> u8 { let mut x: u8 = a % b; x %= 4; return x * b / 3; }
        const R: u8 = f(23, 7);
        """)
        assert values['R'] == 4

    def test_between_constants(self):
        values = const_values("""
        const W: u16 = 7;
        const H: u16 = W * 13 / 3 % 10;
        """)
        assert values['H'] == 0
        build_and_typecheck("""
        const W: u8 = 7;
        const H: u8 = 5;
        fn main() { A = W * H / 3; }
        """)

    def test_division_truncates_toward_zero(self):
        values = const_values("""
        const fn d(a: i8, b: i8) -> i8 { return a / b; }
        const fn m(a: i8, b: i8) -> i8 { return a % b; }
        const DV: i8 = d(-7, 2);
        const MV: i8 = m(-7, 2);
        const D2: i16 = (0 - 7) / 2;
        """)
        assert (values['DV'], values['MV'], values['D2']) == (-3, -1, -3)

    def test_modulo_rejected_at_runtime(self):
        with pytest.raises(Exception, match="only allowed in const fn bodies"):
            build_and_typecheck("""
            #[zeropage]
            static mut Z: u8;
            fn main() { Z = Z % 7; }
            """)

    def test_runtime_call_of_const_only_body(self):
        """A const fn that divides by a variable cannot run: say so at the call."""
        with pytest.raises(Exception, match="const fn 'outer' is called with a non-constant argument"):
            build_and_typecheck("""
            const fn inner(a: u8, b: u8) -> u8 { return a / b; }
            const fn outer(a: u8) -> u8 { return inner(a, 3); }
            #[zeropage]
            static mut Z: u8;
            fn main() { Z = outer(Z); }
            """)


class TestConstFnFailuresSurface:
    """A const fn that fails with constant arguments reports why."""

    def test_iteration_limit_reported(self):
        with pytest.raises(HIRError, match="exceeded maximum iteration limit"):
            build_hir("""
            const fn f() -> [u8; 2] { loop { } return [1, 2]; }
            static VAL: [u8; 2] = f();
            """)

    def test_non_constant_static_initializer_rejected(self):
        """A ROM aggregate whose initializer is not data must not compile to nothing."""
        with pytest.raises(Exception, match="must be a compile-time constant"):
            compile_string("""
            const fn f(x: u8) -> [u8; 2] { return [x, x]; }
            #[zeropage]
            static mut Z: u8;
            static VAL: [u8; 2] = f(Z);
            #[entry] fn main() -> ! { A = VAL[1]; loop {} }
            """, "test.r65")

    def test_recursion_limit(self):
        with pytest.raises(HIRError, match="recursion deeper than 256"):
            build_hir("""
            const fn s(n: u16) -> u16 { if n == 0 { return 0; } return 1 + s(n - 1); }
            const R: u16 = s(2000);
            """)
        assert const_values("""
            const fn s(n: u16) -> u16 { if n == 0 { return 0; } return 1 + s(n - 1); }
            const R: u16 = s(200);
            """)['R'] == 200

    def test_index_out_of_bounds(self):
        with pytest.raises(HIRError, match="index 255 out of bounds"):
            build_hir("""
            const fn f(i: u8) -> u8 { let t: [u8; 3] = [10, 20, 30]; return t[i - 1]; }
            const R: u8 = f(0);
            """)

    def test_multi_value_return_rejected(self):
        with pytest.raises(HIRError, match="returns 2 values"):
            build_hir("""
            const fn two() -> u8, u8 { return 1, 2; }
            const C: u8 = two();
            """)


class TestConstFnWrapping:
    """Values wrap at their type's width, as they do on the 65816."""

    def test_wrapping(self):
        values = const_values("""
        const fn inc(x: u8) -> u8 { let mut y: u8 = x; y = y + 1; return y; }
        const fn under(a: u16, b: u16) -> u16 { return a - b; }
        const fn xs(s: u16) -> u16 {
            let mut x: u16 = s;
            x = x ^ (x << 7); x = x ^ (x >> 9); x = x ^ (x << 8);
            return x;
        }
        const fn nib(x: u8) -> u8 { let y: u8 = ~x; return y >> 4; }
        const fn sign(x: i8) -> u8 { if x < 0 { return 1; } return 0; }
        const fn wide() -> u8 { return 200 + 100; }
        const fn untyped() -> u8 { let mut c = 250; c = c + 10; return c; }
        const A1: u8 = inc(255);
        const B1: u16 = under(1, 2);
        const C1: u16 = xs(0xFFFF);
        const D1: u8 = nib(0x0F);
        const E1: u8 = sign(200 as i8);
        const F1: u8 = wide();
        const G1: u8 = untyped();
        """)
        assert values['A1'] == 0
        assert values['B1'] == 0xFFFF
        assert values['C1'] == 0x7F7F
        assert values['D1'] == 0x0F
        assert values['E1'] == 1
        assert values['F1'] == 44
        assert values['G1'] == 4

    def test_shift_amount_checked(self):
        with pytest.raises(HIRError, match="negative shift amount"):
            build_hir("const M: u8 = 1 << (0 - 1);")
        assert const_values("""
            const fn f(n: u16) -> u16 { return 1 << n; }
            const R: u16 = f(60000);
            """)['R'] == 0


class TestConstFnScoping:
    """R65 scoping, whatever the names mean to the Python it runs as."""

    def test_scopes_and_names(self):
        values = const_values("""
        const n: u8 = 1;
        const fn block() -> u8 { let x: u8 = 1; if true { let x: u8 = 9; } return x; }
        const fn local() -> u8 { let n: u8 = 5; return n; }
        const fn lerp(from: u8, to: u8) -> u8 { return (from + to) / 2; }
        const fn count(range: u8) -> u8 { let mut s: u8 = 0; for i in 0..range { s = s + 1; } return s; }
        const BK: u8 = block();
        const L: u8 = local();
        const P: u8 = lerp(2, 4);
        const R: u8 = count(3);
        """)
        assert (values['BK'], values['L'], values['P'], values['R']) == (1, 5, 3, 3)

    def test_uninitialized_and_nested_arrays(self):
        values = const_values("""
        const fn f() -> [u8; 2] { let mut t: [u8; 2]; t[0] = 1; t[1] = 2; return t; }
        const fn g() -> u8 { let mut t: [[u8; 2]; 2] = [[0; 2]; 2]; t[0][1] = 5; return t[1][1]; }
        const T: [u8; 2] = f();
        const G: u8 = g();
        """)
        assert values['T'] == [1, 2]
        assert values['G'] == 0


class TestConstFnControlFlow:
    """Control flow R65 allows in any fn works in a const fn."""

    def test_loops(self):
        values = const_values("""
        const fn incl() -> u8 { let mut s: u8 = 0; for i in 0..=3 { s = s + 1; } return s; }
        const fn big() -> u16 { let mut s: u16 = 0; for i in 0..128 { for j in 0..128 { s = s + 1; } } return s; }
        const fn lab() -> u8 {
            let mut n: u8 = 0;
            'outer: for i in 0..10 { for j in 0..10 { if i * j == 12 { break 'outer; } n = n + 1; } }
            return n;
        }
        const fn cont() -> u8 {
            let mut n: u8 = 0;
            'outer: for i in 0..4 { for j in 0..4 { if j == 2 { continue 'outer; } n = n + 1; } }
            return n;
        }
        const fn val() -> u8 { let mut i: u8 = 0; let v: u8 = loop { i = i + 1; if i == 7 { break i * 2; } }; return v; }
        const I: u8 = incl();
        const BIG: u16 = big();
        const LAB: u8 = lab();
        const CONT: u8 = cont();
        const VAL: u8 = val();
        """)
        assert (values['I'], values['BIG'], values['LAB'], values['CONT'], values['VAL']) == \
            (4, 16384, 26, 8, 14)

    def test_expressions_with_statements(self):
        values = const_values("""
        const fn tail() -> u8 { 5 }
        const fn tail_if(x: u8) -> u8 { if x > 3 { 1 } else { 2 } }
        const fn blk(x: u8) -> u8 { let y: u8 = if x > 3 { let z: u8 = x + 1; z } else { 0 }; return y; }
        const fn arms(x: u8) -> u8 {
            let mut r: u8 = 0;
            match x { 0 => { r = 10; } 1..=4 => { r = 20; } _ => { r = 30; } };
            return r;
        }
        const fn cond() -> u8 { let mut i: u8 = 0; while match i { 5 => false, _ => true } { i = i + 1; } return i; }
        const T: u8 = tail();
        const TI: u8 = tail_if(9);
        const BL: u8 = blk(5);
        const A0: u8 = arms(0);
        const A3: u8 = arms(3);
        const A9: u8 = arms(9);
        const C: u8 = cond();
        """)
        assert (values['T'], values['TI'], values['BL']) == (5, 1, 6)
        assert (values['A0'], values['A3'], values['A9'], values['C']) == (10, 20, 30, 5)

    def test_cfg(self):
        assert const_values("""
            const fn f() -> u8 { if cfg!(debug) { return 1; } return 2; }
            const R: u8 = f();
            """)['R'] == 2


class TestConstFnResolution:
    """Const methods, forward references and typed results."""

    def test_const_methods(self):
        values = const_values("""
        struct Foo { a: u8 }
        struct Bar { a: u8 }
        impl Foo { const fn k(x: u8) -> u8 { return x + 1; } }
        impl Bar { const fn k(x: u8) -> u8 { return x + 2; } }
        const C: u8 = Foo::k(1);
        const C2: u8 = Bar::k(1);
        """)
        assert (values['C'], values['C2']) == (2, 3)

    def test_forward_references(self):
        values = const_values("""
        const K: u8 = f();
        const fn f() -> u8 { return M * 2; }
        const M: u8 = 21;
        const X1: u8 = Y1 + 1;
        const Y1: u8 = 2;
        """)
        assert (values['K'], values['X1']) == (42, 3)

    def test_const_cycle_reported(self):
        with pytest.raises(HIRError, match="depends on itself"):
            build_hir("""
            const A1: u8 = B1 + 1;
            const B1: u8 = A1;
            """)

    def test_enum_and_newtype_results(self):
        source = """
        enum Dir { N = 0, E, S, W }
        struct Tile(u8);
        const fn opp(d: Dir) -> Dir {
            return match d { Dir::N => Dir::S, Dir::S => Dir::N, Dir::E => Dir::W, Dir::W => Dir::E };
        }
        const fn next(x: u8) -> Tile { return (x + 1) as Tile; }
        const R: Dir = opp(Dir::E);
        const T: Tile = next(2);
        """
        build_and_typecheck(source)
        values = const_values(source)
        assert (values['R'], values['T']) == (3, 3)
