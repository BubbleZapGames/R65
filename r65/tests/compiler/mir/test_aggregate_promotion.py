"""Tests for local aggregate variable promotion and struct decomposition.

Local struct and array variables are handled in two ways:
1. Small flat structs: decomposed into per-field virtual registers (stack-safe)
2. Arrays and ineligible structs: promoted to auto-allocated lowram statics
"""

import pytest
from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.mir.builder import MIRBuilder
from r65.compiler.mir.nodes import (
    MemoryFill, BlockCopy, Store, Load, Move, MIRFunction,
)
from r65.compiler.analysis.call_graph import RecursionChecker, RecursionError


def build_mir(source: str) -> 'MIRProgram':
    """Helper to build MIR from source code."""
    parser = Parser()
    ast = parser.parse(source)
    hir_builder = HIRBuilder()
    hir_prog = hir_builder.build_program(ast)
    type_checker = TypeChecker(hir_prog)
    type_checker.check()
    mir_builder = MIRBuilder()
    return mir_builder.build_program(hir_prog)


def get_function(mir_prog, func_name: str) -> MIRFunction:
    """Get MIRFunction by name."""
    for func in mir_prog.functions:
        if func.name == func_name:
            return func
    return None


def get_function_instructions(mir_prog, func_name: str) -> list:
    """Get all instructions from a function's blocks."""
    func = get_function(mir_prog, func_name)
    if func is None:
        return []
    instrs = []
    for block in func.blocks.values():
        instrs.extend(block.instructions)
    return instrs


# =============================================================================
# Small flat struct decomposition tests
# =============================================================================

class TestStructDecomposition:
    """Test that small flat structs are decomposed into per-field vregs."""

    def test_small_struct_decomposed(self):
        """Small flat struct is decomposed, not promoted to static."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let p: Point;
                p.x = 42;
                RESULT = p.x;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func is not None
        # Decomposed structs should NOT set has_promoted_locals
        assert not func.has_promoted_locals

        # No synthetic static should be created for this struct
        promoted = [s for s in mir.statics if '__local_test_func_p_' in s.name]
        assert len(promoted) == 0

    def test_decomposed_struct_with_initializer(self):
        """Initialized decomposed struct emits per-field Move instructions."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let p: Point = Point { x: 10, y: 20 };
                RESULT = p.x;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert not func.has_promoted_locals

        # Should have Move instructions for field init, NOT BlockCopy
        instrs = get_function_instructions(mir, 'test_func')
        block_copies = [i for i in instrs if isinstance(i, BlockCopy)]
        assert len(block_copies) == 0

        # Should have Move instructions for field initialization
        moves = [i for i in instrs if isinstance(i, Move)]
        assert len(moves) >= 2  # At least x=10, y=20

    def test_decomposed_struct_field_write_read(self):
        """Decomposed struct field writes and reads use vregs."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            #[zeropage(0x20)]
            static mut RX: u8;
            #[zeropage(0x21)]
            static mut RY: u8;

            fn test_func() {
                let p: Point;
                p.x = 42;
                p.y = 99;
                RX = p.x;
                RY = p.y;
            }
        ''')
        # Should compile successfully with Move instructions (not Store to promoted static)
        instrs = get_function_instructions(mir, 'test_func')
        moves = [i for i in instrs if isinstance(i, Move)]
        assert len(moves) >= 2  # Field writes use Move to vregs


# =============================================================================
# Struct decomposition fallback tests
# =============================================================================

class TestStructDecompositionFallback:
    """Test that ineligible structs fall back to static promotion."""

    def test_large_struct_promoted(self):
        """Struct >= 16 bytes falls back to static promotion."""
        mir = build_mir('''
            struct BigStruct {
                a: u16, b: u16, c: u16, d: u16,
                e: u16, f: u16, g: u16, h: u16
            }

            fn test_func() {
                let s: BigStruct;
                s.a = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

    def test_struct_with_array_field_promoted(self):
        """Struct containing array field falls back to static promotion."""
        mir = build_mir('''
            struct WithArray { data: [u8; 4], len: u8 }

            fn test_func() {
                let s: WithArray;
                s.len = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

    def test_struct_with_nested_struct_promoted(self):
        """Struct containing nested struct field falls back to static promotion."""
        mir = build_mir('''
            struct Inner { x: u8 }
            struct Outer { inner: Inner, y: u8 }

            fn test_func() {
                let s: Outer;
                s.y = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

    def test_address_of_struct_promoted(self):
        """Struct whose address is taken falls back to static promotion."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            fn takes_ptr(p: *Point) { }

            fn test_func() {
                let p: Point;
                p.x = 42;
                takes_ptr(&p);
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

    def test_array_always_promoted(self):
        """Arrays are always promoted (never decomposed)."""
        mir = build_mir('''
            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let buf: [u8; 8];
                buf[0] = 42;
                RESULT = buf[0];
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals


# =============================================================================
# Local array promotion tests
# =============================================================================

class TestLocalArrayPromotion:
    """Test that local array variables are promoted to static storage."""

    def test_local_array_fill(self):
        """Local array with fill expression emits MemoryFill."""
        mir = build_mir('''
            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let buf: [u8; 16] = [0; 16];
                buf[0] = 42;
                RESULT = buf[0];
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

        instrs = get_function_instructions(mir, 'test_func')
        fills = [i for i in instrs if isinstance(i, MemoryFill)]
        assert len(fills) == 1
        assert fills[0].fill_value == 0
        assert fills[0].count == 16
        assert fills[0].element_size == 1

    def test_local_array_literal(self):
        """Local array with literal expression emits BlockCopy."""
        mir = build_mir('''
            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let data: [u8; 4] = [10, 20, 30, 40];
                RESULT = data[0];
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

        instrs = get_function_instructions(mir, 'test_func')
        block_copies = [i for i in instrs if isinstance(i, BlockCopy)]
        assert len(block_copies) >= 1

    def test_local_array_uninitialized(self):
        """Uninitialized local array is promoted without init code."""
        mir = build_mir('''
            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let buf: [u8; 8];
                buf[0] = 42;
                RESULT = buf[0];
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

        # No init instructions for the array itself
        instrs = get_function_instructions(mir, 'test_func')
        fills = [i for i in instrs if isinstance(i, MemoryFill)]
        assert len(fills) == 0


# =============================================================================
# Naming uniqueness tests (for promoted locals)
# =============================================================================

class TestPromotedLocalNaming:
    """Test unique naming for promoted locals (arrays and ineligible structs)."""

    def test_multiple_array_locals_different_names(self):
        """Multiple aggregate locals get distinct synthetic static names."""
        mir = build_mir('''
            fn test_func() {
                let a: [u8; 4];
                let b: [u8; 4];
                a[0] = 1;
                b[0] = 2;
            }
        ''')
        promoted = [s for s in mir.statics if '__local_test_func_' in s.name]
        assert len(promoted) == 2
        names = [s.name for s in promoted]
        assert len(set(names)) == 2  # All names are unique

    def test_same_name_different_functions(self):
        """Same local name in different functions gets distinct statics."""
        mir = build_mir('''
            fn func_a() {
                let buf: [u8; 4];
                buf[0] = 1;
            }

            fn func_b() {
                let buf: [u8; 4];
                buf[0] = 2;
            }
        ''')
        promoted = [s for s in mir.statics if '__local_' in s.name]
        assert len(promoted) == 2
        names = [s.name for s in promoted]
        assert any('func_a' in n for n in names)
        assert any('func_b' in n for n in names)


# =============================================================================
# has_promoted_locals flag tests
# =============================================================================

class TestHasPromotedLocalsFlag:
    """Test has_promoted_locals flag on MIRFunction."""

    def test_no_aggregates_no_flag(self):
        """Functions without aggregate locals have flag = False."""
        mir = build_mir('''
            fn test_func() {
                let x: u8 = 42;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func is not None
        assert not func.has_promoted_locals

    def test_with_array_has_flag(self):
        """Functions with array locals have flag = True."""
        mir = build_mir('''
            fn test_func() {
                let buf: [u8; 4];
                buf[0] = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals

    def test_with_small_struct_no_flag(self):
        """Functions with small flat struct locals do NOT have flag = True."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }
            fn test_func() {
                let p: Point;
                p.x = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert not func.has_promoted_locals


# =============================================================================
# Recursion checker tests
# =============================================================================

class TestRecursionWithPromotedLocals:
    """Test recursion interactions with promoted locals and decomposed structs."""

    def test_direct_recursion_with_array_rejected(self):
        """Direct recursion with array local raises RecursionError."""
        mir = build_mir('''
            fn recursive_func(n: u8) {
                let buf: [u8; 4];
                buf[0] = n;
                if n > 0 {
                    recursive_func(n - 1);
                }
            }
        ''')
        checker = RecursionChecker(mir)
        with pytest.raises(RecursionError, match="local struct/array variables"):
            checker.check()

    def test_direct_recursion_with_small_struct_allowed(self):
        """Direct recursion with small flat struct is allowed (decomposed to stack vregs)."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            fn recursive_func(n: u8) {
                let p: Point;
                p.x = n;
                if n > 0 {
                    recursive_func(n - 1);
                }
            }
        ''')
        checker = RecursionChecker(mir)
        checker.check()  # Should NOT raise

    def test_mutual_recursion_with_array_rejected(self):
        """Mutual recursion with array local raises RecursionError."""
        mir = build_mir('''
            fn func_a(n: u8) {
                let buf: [u8; 4];
                buf[0] = n;
                if n > 0 {
                    func_b(n - 1);
                }
            }

            fn func_b(n: u8) {
                func_a(n);
            }
        ''')
        checker = RecursionChecker(mir)
        with pytest.raises(RecursionError, match="local struct/array variables"):
            checker.check()

    def test_no_recursion_no_error(self):
        """Non-recursive function with aggregate local passes check."""
        mir = build_mir('''
            fn func_a() {
                let buf: [u8; 4];
                buf[0] = 1;
            }

            fn func_b() {
                func_a();
            }
        ''')
        checker = RecursionChecker(mir)
        checker.check()  # Should not raise


# =============================================================================
# Struct with enum/pointer fields (eligible for decomposition)
# =============================================================================

class TestStructDecompositionWithScalarFields:
    """Test decomposition of structs with various scalar field types."""

    def test_struct_with_u16_field(self):
        """Struct with u16 field is decomposed."""
        mir = build_mir('''
            struct Entity { id: u8, health: u16 }

            #[zeropage(0x20)]
            static mut RESULT: u16;

            fn test_func() {
                let e: Entity;
                e.health = 100;
                RESULT = e.health;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert not func.has_promoted_locals

    def test_struct_with_pointer_field(self):
        """Struct with pointer field is decomposed (pointers are scalar)."""
        mir = build_mir('''
            struct Ref { ptr: *u8, len: u8 }

            #[zeropage(0x20)]
            static mut RESULT: u8;

            fn test_func() {
                let r: Ref;
                r.len = 5;
                RESULT = r.len;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert not func.has_promoted_locals
