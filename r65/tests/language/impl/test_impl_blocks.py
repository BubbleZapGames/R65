"""Tests for impl block functionality."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, ParseError


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    return hir_prog


class TestImplBlockParsing:
    """Tests for parsing impl blocks."""

    def test_basic_impl_block(self):
        """Basic impl block with method parses."""
        source = """
            struct Player { x: u8, y: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
        """
        program = parse(source, "test.r65")
        assert len(program.items) == 2
        impl_decl = program.items[1]
        assert impl_decl.struct_name == "Player"
        assert len(impl_decl.methods) == 1
        assert impl_decl.methods[0].name == "get_x"

    def test_impl_far_block(self):
        """impl far block parses."""
        source = """
            struct Player { x: u8 }
            impl far Player {
                                fn update(far *self) {
                }
            }
        """
        program = parse(source, "test.r65")
        impl_decl = program.items[1]
        assert impl_decl.is_far is True

    def test_impl_with_constants(self):
        """impl block with associated constants parses."""
        source = """
            struct Player { health: u8 }
            impl Player {
                const MAX_HEALTH: u8 = 100;
                const MIN_HEALTH: u8 = 0;
            }
        """
        program = parse(source, "test.r65")
        impl_decl = program.items[1]
        assert len(impl_decl.constants) == 2
        assert impl_decl.constants[0].name == "MAX_HEALTH"
        assert impl_decl.constants[1].name == "MIN_HEALTH"

    def test_impl_method_with_params(self):
        """impl method with additional parameters parses."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn move_by(*self, dx @ A: u8, dy @ X: u16) {
                }
            }
        """
        program = parse(source, "test.r65")
        method = program.items[1].methods[0]
        assert method.name == "move_by"
        assert len(method.params) == 2
        assert method.params[0].name == "dx"
        assert method.params[1].name == "dy"


class TestImplBlockHIR:
    """Tests for HIR building of impl blocks."""

    def test_method_mangling(self):
        """Methods are mangled to StructName__method."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
        """
        hir = build_and_check(source)

        # Find the impl decl
        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations if isinstance(d, HIRImplDecl))

        # Check method has mangled name
        assert impl_decl.methods[0].name == "Player__get_x"

    def test_constant_qualified_name(self):
        """Associated constants have qualified names."""
        source = """
            struct Player { health: u8 }
            impl Player {
                const MAX_HEALTH: u8 = 100;
            }
        """
        hir = build_and_check(source)

        # Check constant is registered with qualified name
        symbol = hir.symbol_table.lookup("Player::MAX_HEALTH")
        assert symbol is not None
        assert symbol.const_value == 100

    def test_self_parameter_type(self):
        """Self parameter has correct pointer type."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations if isinstance(d, HIRImplDecl))
        method = impl_decl.methods[0]

        # First param is self
        assert method.parameters[0].name == "self"
        param_type = method.parameters[0].param_type
        assert param_type.is_far is False  # Near pointer for impl Player

    def test_far_self_parameter_type(self):
        """Far self parameter has far pointer type."""
        source = """
            struct Player { x: u8 }
            impl far Player {
                                fn update(far *self) {
                }
            }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations if isinstance(d, HIRImplDecl))
        method = impl_decl.methods[0]

        param_type = method.parameters[0].param_type
        assert param_type.is_far is True


class TestMethodCalls:
    """Tests for method call syntax."""

    def test_method_call_on_static(self):
        """Method call on static variable works."""
        source = """
            struct Player { x: u8, health: u8 }
            impl Player {
                                fn take_damage(*self, amount @ A: u8) {
                    self.health -= amount;
                }
            }
            #[zeropage]
            static mut PLAYER: Player;

                        fn test() {
                PLAYER.take_damage(5);
            }
        """
        hir = build_and_check(source)

        # Find test function
        from r65.compiler.hir import HIRFunctionDecl
        test_func = next(d for d in hir.declarations
                         if isinstance(d, HIRFunctionDecl) and d.name == "test")

        # Check method call info is set
        stmt = test_func.body.statements[0]
        call_expr = stmt.expr
        assert call_expr.method_call_info is not None
        assert call_expr.method_call_info['mangled_name'] == "Player__take_damage"

    def test_method_call_on_pointer(self):
        """Method call on pointer works."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
            #[zeropage]
            static mut PLAYER_PTR: *Player;

                        fn test() -> u8 {
                return PLAYER_PTR.get_x();
            }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRFunctionDecl
        test_func = next(d for d in hir.declarations
                         if isinstance(d, HIRFunctionDecl) and d.name == "test")

        # Method call should work on pointer
        ret_stmt = test_func.body.statements[0]
        call_expr = ret_stmt.values[0]
        assert call_expr.method_call_info is not None


class TestAutoDeref:
    """Tests for auto-dereferencing in field access."""

    def test_self_field_access(self):
        """self.field auto-dereferences."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
        """
        hir = build_and_check(source)

        # Should type check without errors
        # (auto-deref handles *self.x -> (*self).x)


class TestImplErrors:
    """Tests for impl block error handling."""

    def test_impl_undefined_struct(self):
        """impl block for undefined struct fails."""
        source = """
            impl Undefined {
                                fn foo(*self) {}
            }
        """
        with pytest.raises(Exception) as exc_info:
            build_and_check(source)
        assert "undefined" in str(exc_info.value).lower()

    def test_wrong_pointer_type(self):
        """Calling near method with far pointer fails."""
        source = """
            struct Player { x: u8 }
            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }
            #[zeropage]
            static mut FAR_PTR: far *Player;

                        fn test() -> u8 {
                return FAR_PTR.get_x();
            }
        """
        with pytest.raises(TypeCheckError) as exc_info:
            build_and_check(source)
        assert "far pointer" in str(exc_info.value).lower() or "near" in str(exc_info.value).lower()


class TestMultipleImplBlocks:
    """Tests for multiple impl blocks on same struct."""

    def test_multiple_impl_blocks(self):
        """Multiple impl blocks for same struct allowed."""
        source = """
            struct Player { x: u8, y: u8 }

            impl Player {
                                fn get_x(*self) -> u8 {
                    return self.x;
                }
            }

            impl Player {
                                fn get_y(*self) -> u8 {
                    return self.y;
                }
            }
        """
        hir = build_and_check(source)

        # Both methods should be registered
        get_x = hir.symbol_table.lookup("Player.get_x")
        get_y = hir.symbol_table.lookup("Player.get_y")
        assert get_x is not None
        assert get_y is not None
