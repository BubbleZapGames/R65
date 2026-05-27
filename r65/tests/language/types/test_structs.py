# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for struct types."""

from r65.compiler.frontend import ast
from r65.tests.language.common import parse_struct, parse_static, parse_function


class TestStructDeclaration:
    """Tests for struct declaration parsing."""

    def test_basic_struct(self):
        """Test basic struct with multiple fields."""
        struct = parse_struct("struct Point { x: u8, y: u8 }")
        assert struct.name == "Point"
        assert len(struct.fields) == 2
        assert struct.fields[0].name == "x"
        assert struct.fields[1].name == "y"

    def test_struct_with_various_types(self):
        """Test struct with different field types."""
        struct = parse_struct("""
            struct Entity {
                x: u16,
                y: u16,
                health: u8,
                alive: bool
            }
        """)
        assert len(struct.fields) == 4
        assert struct.fields[0].field_type.name == "u16"
        assert struct.fields[3].field_type.name == "bool"

    def test_empty_struct(self):
        """Test empty struct."""
        struct = parse_struct("struct Empty { }")
        assert struct.name == "Empty"
        assert len(struct.fields) == 0


class TestStructInstances:
    """Tests for struct instances and access."""

    def test_struct_static_declaration(self):
        """Test struct as static variable type."""
        prog_source = """
            struct Player { x: u8, y: u8 }
            #[ram] static mut P: Player;
        """
        from r65.tests.language.common import parse_program
        prog = parse_program(prog_source)
        static = prog.items[1]
        # Named types (like struct names) are parsed as BasicType
        assert isinstance(static.var_type, ast.BasicType)
        assert static.var_type.name == "Player"

    def test_struct_field_access(self):
        """Test struct field access with dot operator."""
        func = parse_function("fn test() { let x: u8 = player.x; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.FieldAccess)
        assert let_stmt.initializer.field == "x"

    def test_nested_field_access(self):
        """Test nested struct field access."""
        func = parse_function("fn test() { let v: u8 = outer.inner.value; }")
        let_stmt = func.body.statements[0]
        # outer.inner.value is FieldAccess(FieldAccess(outer, inner), value)
        assert isinstance(let_stmt.initializer, ast.FieldAccess)
        assert let_stmt.initializer.field == "value"


class TestStructLiterals:
    """Tests for struct literal initialization."""

    def test_struct_literal(self):
        """Test struct literal syntax."""
        func = parse_function("fn test() { let p = Point { x: 10, y: 20 }; }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.StructLiteralExpr)
        assert let_stmt.initializer.struct_name == "Point"
        assert len(let_stmt.initializer.fields) == 2



class TestStructArrayTypeCheck:
    """Tests for struct array type checking (requires full type check)."""

    def test_array_index_field_assignment(self):
        """Test that array[index].field = value type checks correctly.

        This was a regression where the type checker compared AST BasicType
        with HIR BasicTypeInfo, causing false type mismatch errors.
        """
        from r65.compiler.frontend import Parser
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker

        source = """
            struct Card { suit: u8, rank: u8 }
            #[ram] static mut CARDS: [Card; 8];

                        fn test() {
                let idx: u8 = 0;
                CARDS[idx].suit = 3;
                CARDS[idx].rank = 7;
            }
        """

        parser = Parser()
        ast_prog = parser.parse(source)
        hir_builder = HIRBuilder()
        hir_prog = hir_builder.build_program(ast_prog)
        type_checker = TypeChecker(hir_prog)
        # Should not raise - this was the bug
        type_checker.check()

    def test_array_const_index_field_assignment(self):
        """Test array[constant].field = value."""
        from r65.compiler.frontend import Parser
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker

        source = """
            struct Point { x: u8, y: u8 }
            #[ram] static mut POINTS: [Point; 4];

                        fn init() {
                POINTS[0].x = 10;
                POINTS[0].y = 20;
                POINTS[1].x = 30;
                POINTS[1].y = 40;
            }
        """

        parser = Parser()
        ast_prog = parser.parse(source)
        hir_builder = HIRBuilder()
        hir_prog = hir_builder.build_program(ast_prog)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()

    def test_array_field_read(self):
        """Test reading array[index].field."""
        from r65.compiler.frontend import Parser
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker

        source = """
            struct Entity { x: u8, y: u8, health: u16 }
            #[ram] static mut ENTITIES: [Entity; 8];

                        fn get_health(idx @ X: u16) -> u16 {
                return ENTITIES[idx].health;
            }
        """

        parser = Parser()
        ast_prog = parser.parse(source)
        hir_builder = HIRBuilder()
        hir_prog = hir_builder.build_program(ast_prog)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()


class TestNestedStructArrayCodegen:
    """Codegen for an array-of-struct that is itself a field of a struct.

    Regression: MIR lowering required the array base of `arr[i]` /
    `arr[i].field` to be a bare static identifier, so it rejected
    `BUF.sprites[i].x = v` (array is a struct field) with
    "Array field assignment requires static array, got: HIRFieldAccess".
    The fix resolves the array base through `resolve_array_base_memloc`,
    folding the array field's offset into the base memory location.
    """

    def test_nested_array_field_const_index(self):
        """BUF.sprites[3].x = v writes at struct_base + 3*elem + field_off."""
        import re
        from r65.compiler.main import compile_string

        source = """
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 128], ext: [u8; 32] }
            #[lowram] static mut BUF: OamBuffer;

            fn main() {
                BUF.sprites[3].x = 10;
                BUF.sprites[3].attr = 1;
            }
        """
        asm = compile_string(source, "test.r65")
        # sprites at field offset 0; sprites[3] = base + 3*4 = base + 12.
        # x is +0, attr is +3. The two stores land 3 bytes apart.
        addrs = [int(m, 16) for m in re.findall(r"STA \$([0-9A-Fa-f]{4})\b", asm)]
        assert len(addrs) >= 2, f"expected absolute stores, got: {addrs}"
        assert addrs[1] - addrs[0] == 3, f"attr should be 3 bytes past x: {addrs}"

    def test_nested_u8_array_field_const_index(self):
        """BUF.ext[2] = v must fold the ext field's offset into the address."""
        import re
        from r65.compiler.main import compile_string

        source = """
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 128], ext: [u8; 32] }
            #[lowram] static mut BUF: OamBuffer;

            fn main() {
                BUF.sprites[0].x = 1;
                BUF.ext[2] = 0xFF;
            }
        """
        asm = compile_string(source, "test.r65")
        # sprites[0].x is at the struct base; ext[2] is at base + 128*4 + 2 = base + 514.
        addrs = [int(m, 16) for m in re.findall(r"STA \$([0-9A-Fa-f]{4})\b", asm)]
        assert len(addrs) >= 2, f"expected absolute stores, got: {addrs}"
        assert addrs[1] - addrs[0] == 514, f"ext[2] offset wrong: {addrs}"

    def test_nested_array_field_variable_index(self):
        """BUF.sprites[i].x = v uses X indexing off the resolved base."""
        import re
        from r65.compiler.main import compile_string

        source = """
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 128], ext: [u8; 32] }
            #[lowram] static mut BUF: OamBuffer;

            fn main() {
                let i: u8 = 5;
                BUF.sprites[i].x = 7;
            }
        """
        asm = compile_string(source, "test.r65")
        # i*4 scaled into X (two ASL), then an X-indexed store.
        assert "ASL A" in asm
        assert "TAX" in asm
        assert re.search(r"STA \$[0-9A-Fa-f]{4},X", asm), \
            "expected X-indexed absolute store for BUF.sprites[i].x"

    def test_nested_array_field_reuses_x_index(self):
        """Consecutive writes to the same struct-field element reuse X.

        The scaled index (i * elem_size) in X is identical for every field of
        BUF.sprites[i], so the second/third writes must not recompute it — even
        though the array sits inside a struct.
        """
        from r65.compiler.main import compile_string

        source = """
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 128], ext: [u8; 32] }
            #[lowram] static mut BUF: OamBuffer;

            fn main() {
                let i: u8 = 5;
                BUF.sprites[i].x = 0;
                BUF.sprites[i].y = 9;
                BUF.sprites[i].attr = 3;
            }
        """
        asm = compile_string(source, "test.r65")
        # X computed exactly once (one i*4 scale = two ASL, one TAX) and shared.
        assert asm.count("ASL A") == 2, f"expected one i*4 scale, got:\n{asm}"
        assert asm.count("TAX") == 1, f"expected X loaded once, got:\n{asm}"

    def test_different_field_arrays_dont_share_x_index(self):
        """sprites[i] (elem 4) and ext[i] (elem 1) must not share X.

        Distinct field paths get distinct reuse keys, so the scaled index is
        never falsely reused across arrays of different element size.
        """
        from r65.compiler.main import compile_string

        source = """
            struct OamEntry { x: u8, y: u8, tile: u8, attr: u8 }
            struct OamBuffer { sprites: [OamEntry; 128], ext: [u8; 32] }
            #[lowram] static mut BUF: OamBuffer;

            fn main() {
                let i: u8 = 5;
                BUF.sprites[i].x = 0;
                BUF.ext[i] = 1;
                BUF.sprites[i].y = 2;
            }
        """
        asm = compile_string(source, "test.r65")
        # sprites scale i*4 twice (recomputed after ext clobbers X) → 4 ASL.
        # ext scales i*1 (no ASL). Reusing X here would corrupt addresses.
        assert asm.count("ASL A") == 4, f"expected sprites scale recomputed, got:\n{asm}"
