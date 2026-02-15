"""Tests for trait declarations and trait impls."""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import TypeCheckError, ParseError, HIRError


def build_and_check(source: str):
    """Parse, build HIR, and type check source."""
    program = parse(source, "test.r65")
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    return hir_prog


def compile_to_asm(source: str) -> str:
    """Compile R65 source to assembly string."""
    from r65.compiler.frontend.preprocessor import preprocess
    from r65.compiler.frontend.macros import expand_macros
    from r65.compiler.mir.builder import MIRBuilder
    from r65.compiler.codegen.codegen import ProgramCodeGenerator

    program = parse(source, "test.r65")
    program = preprocess(program, "test.r65")
    program = expand_macros(program)
    hir_builder = HIRBuilder(source_file="test.r65")
    hir_prog = hir_builder.build_program(program)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    mir_builder = MIRBuilder()
    mir_prog = mir_builder.build_program(hir_prog)
    from r65.compiler.analysis import RecursionChecker
    RecursionChecker(mir_prog).check()
    codegen = ProgramCodeGenerator()
    return codegen.generate(mir_prog)


class TestTraitParsing:
    """Tests for parsing trait declarations."""

    def test_basic_trait(self):
        """Basic trait with one method parses."""
        source = """
            trait Drawable {
                fn draw(*self);
            }
        """
        program = parse(source, "test.r65")
        assert len(program.items) == 1
        trait = program.items[0]
        assert trait.name == "Drawable"
        assert len(trait.methods) == 1
        assert trait.methods[0].name == "draw"

    def test_trait_multiple_methods(self):
        """Trait with multiple methods parses."""
        source = """
            trait Renderable {
                fn draw(*self);
                fn get_width(*self) -> u8;
                fn get_height(*self) -> u8;
            }
        """
        program = parse(source, "test.r65")
        trait = program.items[0]
        assert len(trait.methods) == 3
        assert trait.methods[0].name == "draw"
        assert trait.methods[1].name == "get_width"
        assert trait.methods[2].name == "get_height"

    def test_trait_method_with_params(self):
        """Trait method with additional parameters parses."""
        source = """
            trait Movable {
                fn move_by(*self, dx @ A: u8, dy @ X: u16);
            }
        """
        program = parse(source, "test.r65")
        method = program.items[0].methods[0]
        assert method.name == "move_by"
        assert len(method.params) == 2

    def test_impl_trait_for_struct(self):
        """impl Trait for Struct parses."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player {
                fn draw(*self) { }
            }
        """
        program = parse(source, "test.r65")
        assert len(program.items) == 3
        impl_decl = program.items[2]
        assert impl_decl.trait_name == "Drawable"
        assert impl_decl.struct_name == "Player"

    def test_trait_with_return_type(self):
        """Trait method with return type parses."""
        source = """
            trait HasHealth {
                fn get_health(*self) -> u8;
            }
        """
        program = parse(source, "test.r65")
        method = program.items[0].methods[0]
        assert method.return_type is not None


class TestTraitHIR:
    """Tests for HIR building of traits."""

    def test_trait_declaration_in_hir(self):
        """Trait declaration is in HIR."""
        source = """
            trait Drawable {
                fn draw(*self);
            }
        """
        hir = build_and_check(source)
        from r65.compiler.hir.nodes import HIRTraitDecl
        traits = [d for d in hir.declarations if isinstance(d, HIRTraitDecl)]
        assert len(traits) == 1
        assert traits[0].name == "Drawable"

    def test_trait_symbol_registered(self):
        """Trait is registered in symbol table."""
        source = """
            trait Drawable {
                fn draw(*self);
            }
        """
        hir = build_and_check(source)
        symbol = hir.symbol_table.lookup("Drawable")
        assert symbol is not None
        assert symbol.kind.value == "trait"

    def test_type_id_assigned(self):
        """Structs implementing traits get TypeId."""
        source = """
            struct Player { x: u8 }
            struct Enemy { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            impl Drawable for Enemy { fn draw(*self) { } }
        """
        hir = build_and_check(source)

        # Each struct should have a TYPE_ID constant
        player_id = hir.symbol_table.lookup("Player::TYPE_ID")
        enemy_id = hir.symbol_table.lookup("Enemy::TYPE_ID")
        assert player_id is not None
        assert enemy_id is not None
        assert player_id.const_value != enemy_id.const_value
        assert player_id.const_value >= 1
        assert enemy_id.const_value >= 1

    def test_struct_offset_adjusted_for_type_id(self):
        """Struct fields are shifted by 1 byte for TypeId."""
        source = """
            struct Player { x: u8, y: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRStructDecl
        player = next(d for d in hir.declarations
                      if isinstance(d, HIRStructDecl) and d.name == "Player")

        # __type_id at offset 0, user fields start at offset 1
        type_id_field = next(f for f in player.fields if f.name == "__type_id")
        assert type_id_field.offset == 0

        x_field = next(f for f in player.fields if f.name == "x")
        y_field = next(f for f in player.fields if f.name == "y")
        assert x_field.offset == 1
        assert y_field.offset == 2

    def test_struct_field_count_includes_type_id(self):
        """Struct has 3 fields: __type_id, x, y."""
        source = """
            struct Player { x: u8, y: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRStructDecl
        player = next(d for d in hir.declarations
                      if isinstance(d, HIRStructDecl) and d.name == "Player")
        # Should have 3 fields: __type_id (offset 0), x (offset 1), y (offset 2)
        assert len(player.fields) == 3

    def test_struct_without_trait_no_type_id(self):
        """Structs without trait impls don't get TypeId."""
        source = """
            struct Plain { x: u8, y: u8 }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRStructDecl
        plain = next(d for d in hir.declarations
                     if isinstance(d, HIRStructDecl) and d.name == "Plain")
        assert len(plain.fields) == 2
        assert not any(f.name == "__type_id" for f in plain.fields)

    def test_trait_method_mangling(self):
        """Trait impl methods are mangled like regular impl methods."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
        """
        hir = build_and_check(source)

        from r65.compiler.hir import HIRImplDecl
        impl_decl = next(d for d in hir.declarations
                         if isinstance(d, HIRImplDecl) and d.trait_name == "Drawable")
        assert impl_decl.methods[0].name == "Player__draw"

    def test_trait_dispatch_symbol(self):
        """Trait dispatch symbols are registered."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
        """
        hir = build_and_check(source)

        # Check dispatch symbol exists
        dispatch = hir.symbol_table.lookup("Drawable.draw.Player")
        assert dispatch is not None


    def test_trait_method_register_binding_error(self):
        """Trait method params cannot have register bindings."""
        source = """
            trait Foo {
                fn bar(*self, x @ A: u8);
            }
        """
        with pytest.raises(HIRError, match="cannot have a register binding"):
            build_and_check(source)

    def test_trait_method_variable_binding_error(self):
        """Trait method params cannot have variable bindings."""
        source = """
            #[zeropage]
            static mut TEMP: u8;
            trait Foo {
                fn bar(*self, x @ TEMP: u8);
            }
        """
        with pytest.raises(HIRError, match="cannot have a register binding"):
            build_and_check(source)


class TestTraitTypeChecking:
    """Tests for type checking of traits."""

    def test_trait_pointer_assignment(self):
        """*Struct can be assigned to *Trait if struct implements trait."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
            }
        """
        build_and_check(source)  # Should not raise

    def test_trait_pointer_wrong_struct_fails(self):
        """*Struct cannot be assigned to *Trait if struct doesn't implement trait."""
        source = """
            struct Player { x: u8 }
            struct Enemy { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut ENEMY: Enemy;
            fn test() {
                let p: *dyn Drawable = &ENEMY;
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)

    def test_trait_method_call_type_checks(self):
        """Method call on trait pointer type checks."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        build_and_check(source)  # Should not raise

    def test_trait_method_wrong_name_fails(self):
        """Calling non-existent method on trait pointer fails."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
                p.update();
            }
        """
        with pytest.raises(TypeCheckError):
            build_and_check(source)

    def test_impl_missing_method_fails(self):
        """impl Trait for Struct must provide all methods."""
        source = """
            struct Player { x: u8 }
            trait Renderable {
                fn draw(*self);
                fn get_width(*self) -> u8;
            }
            impl Renderable for Player {
                fn draw(*self) { }
            }
        """
        with pytest.raises(Exception):
            build_and_check(source)

    def test_impl_extra_method_fails(self):
        """impl Trait for Struct with wrong method name fails."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player {
                fn render(*self) { }
            }
        """
        with pytest.raises(Exception):
            build_and_check(source)


class TestTraitMIR:
    """Tests for MIR lowering of traits."""

    def test_trait_dispatch_node_emitted(self):
        """Trait method call emits TraitDispatch MIR node."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        from r65.compiler.frontend import parse
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker
        from r65.compiler.mir.builder import MIRBuilder
        from r65.compiler.mir.nodes import TraitDispatch

        program = parse(source, "test.r65")
        hir_builder = HIRBuilder(source_file="test.r65")
        hir_prog = hir_builder.build_program(program)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()

        mir_builder = MIRBuilder()
        mir_prog = mir_builder.build_program(hir_prog)

        # Find the test function
        test_func = next(f for f in mir_prog.functions if f.name == "test")

        # Look for TraitDispatch instruction
        found_dispatch = False
        for block in test_func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, TraitDispatch):
                    found_dispatch = True
                    assert instr.trait_name == "Drawable"
                    assert instr.method_name == "draw"
                    break

        assert found_dispatch, "TraitDispatch node not found in MIR"

    def test_trait_dispatch_info_populated(self):
        """MIR program has trait_dispatch_info populated."""
        source = """
            struct Player { x: u8 }
            struct Enemy { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            impl Drawable for Enemy { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        from r65.compiler.frontend import parse
        from r65.compiler.hir import HIRBuilder
        from r65.compiler.typeck import TypeChecker
        from r65.compiler.mir.builder import MIRBuilder

        program = parse(source, "test.r65")
        hir_builder = HIRBuilder(source_file="test.r65")
        hir_prog = hir_builder.build_program(program)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()

        mir_builder = MIRBuilder()
        mir_prog = mir_builder.build_program(hir_prog)

        assert hasattr(mir_prog, 'trait_dispatch_info')
        assert 'Drawable' in mir_prog.trait_dispatch_info
        info = mir_prog.trait_dispatch_info['Drawable']
        assert 'draw' in info['methods']
        assert len(info['implementors']) == 2

    def test_static_struct_literal_includes_type_id(self):
        """Static struct literal init data includes TypeId byte in assembly output."""
        source = """
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;
            struct Player { x: u8, y: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[lowram]
            static mut PLAYER: Player = Player { x: 42, y: 10 };
        """
        asm = compile_to_asm(source)

        # Init data should contain TypeId (1), x (42=0x2A), y (10=0x0A)
        # as .db $01, $2A, $0A
        assert ".db $01, $2A, $0A" in asm


class TestTraitCodeGen:
    """Tests for code generation of traits."""

    def test_dispatch_table_in_assembly(self):
        """Assembly output contains dispatch table."""
        source = """
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;
            struct Player { x: u8 }
            struct Enemy { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            impl Drawable for Enemy { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        asm = compile_to_asm(source)

        # Check dispatch wrapper function exists
        assert "Drawable__draw__dispatch:" in asm

        # Check jump table exists
        assert "Drawable__draw__table:" in asm

        # Check table has entries for both implementors
        assert "Player__draw" in asm
        assert "Enemy__draw" in asm

        # Check error handler exists
        assert "_trait_error:" in asm

    def test_dispatch_wrapper_loads_type_id(self):
        """Dispatch wrapper loads TypeId from self pointer."""
        source = """
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        asm = compile_to_asm(source)

        # Dispatch wrapper should load TypeId via Y-pointer: LDA $0000,Y
        assert "LDA $0000,Y" in asm

    def test_main_calls_dispatch(self):
        """Main function calls dispatch wrapper via JSR."""
        source = """
            #[zeropage(0x10, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x12, register)]
            static mut SCRATCH1: u16;
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            #[entry]
            fn main() {
                let p: *dyn Drawable = &PLAYER;
                p.draw();
            }
        """
        asm = compile_to_asm(source)

        assert "JSR Drawable__draw__dispatch" in asm


class TestDynSyntax:
    """Tests for *dyn TraitName syntax validation."""

    def test_dyn_trait_pointer_parses(self):
        """*dyn TraitName parses correctly."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *dyn Drawable = &PLAYER;
            }
        """
        build_and_check(source)  # Should not raise

    def test_trait_pointer_without_dyn_errors(self):
        """*TraitName (without dyn) gives error with hint."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: *Drawable = &PLAYER;
            }
        """
        with pytest.raises(HIRError, match=r"trait pointer requires 'dyn' keyword.*\*dyn Drawable"):
            build_and_check(source)

    def test_dyn_on_non_trait_errors(self):
        """*dyn StructName gives error."""
        source = """
            struct Player { x: u8 }
            fn test() {
                let p: *dyn Player = 0;
            }
        """
        with pytest.raises(HIRError, match="'dyn' can only be used with trait types"):
            build_and_check(source)

    def test_dyn_on_basic_type_errors(self):
        """*dyn u8 gives error."""
        source = """
            fn test() {
                let p: *dyn u8 = 0;
            }
        """
        with pytest.raises(HIRError, match="'dyn' can only be used with trait types"):
            build_and_check(source)

    def test_far_dyn_trait_pointer_parses(self):
        """far *dyn TraitName parses and resolves to correct type."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
        """
        # Just verify the type resolves - far *dyn requires a far pointer value
        # which would need far address-of (&far), so we just test parsing + HIR
        program = parse(source + """
            #[lowram]
            static mut PTRS: [far *dyn Drawable; 2];
        """, "test.r65")
        hir_builder = HIRBuilder(source_file="test.r65")
        hir_prog = hir_builder.build_program(program)
        type_checker = TypeChecker(hir_prog)
        type_checker.check()  # Should not raise

    def test_array_of_dyn_trait_pointers(self):
        """[*dyn Drawable; 4] array type works."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            #[lowram]
            static mut PTRS: [*dyn Drawable; 4];
        """
        build_and_check(source)  # Should not raise

    def test_type_alias_dyn_trait_pointer(self):
        """type Dp = *dyn Drawable; works."""
        source = """
            struct Player { x: u8 }
            trait Drawable { fn draw(*self); }
            impl Drawable for Player { fn draw(*self) { } }
            type DrawablePtr = *dyn Drawable;
            #[zeropage]
            static mut PLAYER: Player;
            fn test() {
                let p: DrawablePtr = &PLAYER;
            }
        """
        build_and_check(source)  # Should not raise
