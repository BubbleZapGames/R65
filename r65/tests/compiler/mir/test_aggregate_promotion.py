"""Tests for local aggregate variable promotion to static storage.

Local struct and array variables are promoted to auto-allocated lowram statics
because the 65816 lacks stack-indexed addressing for variable-index array access.
"""

import pytest
from r65.compiler.frontend import Parser
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker
from r65.compiler.mir.builder import MIRBuilder
from r65.compiler.mir.nodes import (
    MemoryFill, BlockCopy, Store, Load, MIRFunction,
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
# Local struct promotion tests
# =============================================================================

class TestLocalStructPromotion:
    """Test that local struct variables are promoted to static storage."""

    def test_local_struct_uninitialized(self):
        """Uninitialized local struct is promoted without init code."""
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
        assert func.has_promoted_locals

        # Check that a synthetic static was created
        promoted = [s for s in mir.statics if '__local_test_func_p_' in s.name]
        assert len(promoted) == 1
        assert promoted[0].is_mutable
        assert promoted[0].storage_attr is not None
        assert promoted[0].storage_attr.storage_kind.value == 'lowram'

    def test_local_struct_with_initializer(self):
        """Initialized local struct emits BlockCopy for initialization."""
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
        assert func.has_promoted_locals

        # Should have BlockCopy for struct initialization
        instrs = get_function_instructions(mir, 'test_func')
        block_copies = [i for i in instrs if isinstance(i, BlockCopy)]
        assert len(block_copies) >= 1

        # Check ROM data was created
        assert len(mir.rom_data_sections) >= 1

    def test_local_struct_field_write_read(self):
        """Local struct field writes and reads work via promoted static."""
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
        # Should compile successfully with stores to promoted static
        instrs = get_function_instructions(mir, 'test_func')
        stores = [i for i in instrs if isinstance(i, Store)]
        assert len(stores) >= 2  # At least field writes


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
# Naming uniqueness tests
# =============================================================================

class TestPromotedLocalNaming:
    """Test unique naming for promoted locals."""

    def test_multiple_locals_different_names(self):
        """Multiple aggregate locals get distinct synthetic static names."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            fn test_func() {
                let a: Point;
                let b: Point;
                a.x = 1;
                b.x = 2;
            }
        ''')
        promoted = [s for s in mir.statics if '__local_test_func_' in s.name]
        assert len(promoted) == 2
        names = [s.name for s in promoted]
        assert len(set(names)) == 2  # All names are unique

    def test_same_name_different_functions(self):
        """Same local name in different functions gets distinct statics."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            fn func_a() {
                let p: Point;
                p.x = 1;
            }

            fn func_b() {
                let p: Point;
                p.x = 2;
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

    def test_with_aggregate_has_flag(self):
        """Functions with aggregate locals have flag = True."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }
            fn test_func() {
                let p: Point;
                p.x = 1;
            }
        ''')
        func = get_function(mir, 'test_func')
        assert func.has_promoted_locals


# =============================================================================
# Recursion checker tests
# =============================================================================

class TestRecursionWithPromotedLocals:
    """Test that recursion with promoted locals is rejected."""

    def test_direct_recursion_rejected(self):
        """Direct recursion with aggregate local raises RecursionError."""
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
        with pytest.raises(RecursionError, match="local struct/array variables"):
            checker.check()

    def test_mutual_recursion_rejected(self):
        """Mutual recursion with aggregate local raises RecursionError."""
        mir = build_mir('''
            struct Point { x: u8, y: u8 }

            fn func_a(n: u8) {
                let p: Point;
                p.x = n;
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
            struct Point { x: u8, y: u8 }

            fn func_a() {
                let p: Point;
                p.x = 1;
            }

            fn func_b() {
                func_a();
            }
        ''')
        checker = RecursionChecker(mir)
        checker.check()  # Should not raise
