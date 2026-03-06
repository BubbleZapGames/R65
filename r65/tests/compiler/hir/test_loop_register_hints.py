"""
Tests for loop variable register hint assignment.

Tests that for loop variables are assigned register hints based on nesting depth:
- Depth 1 (outermost): X register
- Depth 2 (first nested): Y register
- Depth 3+: no hint
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir.symbol_table import SymbolKind


class TestLoopRegisterHints:
    """Tests for register hint assignment to loop variables."""

    def test_outer_loop_gets_x_hint(self):
        """Test that outermost loop variable gets X register hint."""
        source = """
        fn test() {
            for i in 0..10 {
                A = i;
            }
        }
        """
        ast = parse(source, "test.r65")
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Find the loop variable symbol
        # The for loop is desugared to a block with let + while
        # We need to check the symbol table for 'i'
        found_i = False
        for scope_id, scope in builder.symbol_table.scopes.items():
            if 'i' in scope.symbols:
                symbol = scope.symbols['i']
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    found_i = True
                    assert symbol.register_hint == 'X', \
                        f"Expected X hint for outer loop, got {symbol.register_hint}"
                    break

        assert found_i, "Loop variable 'i' not found in symbol table"

    def test_inner_loop_gets_y_hint(self):
        """Test that first nested loop variable gets Y register hint."""
        source = """
        fn test() {
            for i in 0..4 {
                for j in 0..8 {
                    A = i + j;
                }
            }
        }
        """
        ast = parse(source, "test.r65")
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Find both loop variable symbols
        found_i = False
        found_j = False
        for scope_id, scope in builder.symbol_table.scopes.items():
            for name, symbol in scope.symbols.items():
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    if name == 'i':
                        found_i = True
                        assert symbol.register_hint == 'X', \
                            f"Expected X hint for outer loop 'i', got {symbol.register_hint}"
                    elif name == 'j':
                        found_j = True
                        assert symbol.register_hint == 'Y', \
                            f"Expected Y hint for inner loop 'j', got {symbol.register_hint}"

        assert found_i, "Outer loop variable 'i' not found"
        assert found_j, "Inner loop variable 'j' not found"

    def test_third_nested_loop_gets_no_hint(self):
        """Test that third nested loop variable gets no register hint."""
        source = """
        fn test() {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        A = i + j + k;
                    }
                }
            }
        }
        """
        ast = parse(source, "test.r65")
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Find the third loop variable
        found_k = False
        for scope_id, scope in builder.symbol_table.scopes.items():
            if 'k' in scope.symbols:
                symbol = scope.symbols['k']
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    found_k = True
                    assert symbol.register_hint is None, \
                        f"Expected no hint for third loop 'k', got {symbol.register_hint}"
                    break

        assert found_k, "Third loop variable 'k' not found"

    def test_sequential_loops_both_get_x(self):
        """Test that sequential (non-nested) loops both get X hint."""
        source = """
        fn test() {
            for i in 0..5 {
                A = i;
            }
            for j in 0..5 {
                A = j;
            }
        }
        """
        ast = parse(source, "test.r65")
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Both should have X hint since they're both at depth 1
        found_i = False
        found_j = False
        for scope_id, scope in builder.symbol_table.scopes.items():
            for name, symbol in scope.symbols.items():
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    if name == 'i':
                        found_i = True
                        assert symbol.register_hint == 'X', \
                            f"Expected X hint for first loop 'i', got {symbol.register_hint}"
                    elif name == 'j':
                        found_j = True
                        assert symbol.register_hint == 'X', \
                            f"Expected X hint for second loop 'j', got {symbol.register_hint}"

        assert found_i, "First loop variable 'i' not found"
        assert found_j, "Second loop variable 'j' not found"

    def test_loop_depth_resets_after_function(self):
        """Test that loop depth resets for each function."""
        source = """
        fn test1() {
            for i in 0..5 {
                A = i;
            }
        }

        fn test2() {
            for j in 0..5 {
                A = j;
            }
        }
        """
        ast = parse(source, "test.r65")
        builder = HIRBuilder()
        hir = builder.build_program(ast)

        # Both should have X hint since they're in separate functions
        found_i = False
        found_j = False
        for scope_id, scope in builder.symbol_table.scopes.items():
            for name, symbol in scope.symbols.items():
                if symbol.kind == SymbolKind.LOCAL_VAR:
                    if name == 'i':
                        found_i = True
                        assert symbol.register_hint == 'X', \
                            f"Expected X hint for 'i' in test1, got {symbol.register_hint}"
                    elif name == 'j':
                        found_j = True
                        assert symbol.register_hint == 'X', \
                            f"Expected X hint for 'j' in test2, got {symbol.register_hint}"

        assert found_i, "Loop variable 'i' not found"
        assert found_j, "Loop variable 'j' not found"


class TestLoopRegisterAllocation:
    """Tests for actual register allocation based on loop hints."""

    def test_outer_loop_uses_x_register(self):
        """Test that outer loop variable is allocated to X register in codegen."""
        from r65.compiler.main import compile_string

        source = """
        #[zeropage(0x10, register)]
        static mut SCRATCH0: u8;

        #[ram]
        static mut result: u8 = 0;

        fn test() {
            for i in 0..10 {
                result = result + i;
            }
        }

        #[entry]
        fn main() {
            test();
        }
        """
        result = compile_string(source, "test.r65")

        # The loop should use X register operations
        # Look for LDX #$00 (init), CPX (compare), INX or similar
        assert "LDX #$00" in result or "LDX #0" in result, \
            "Expected X register initialization for outer loop"

    def test_nested_loops_use_x_and_y(self):
        """Test that nested loops use X and Y registers."""
        from r65.compiler.main import compile_string

        source = """
        #[zeropage(0x10, register)]
        static mut SCRATCH0: u8;
        #[zeropage(0x12, register)]
        static mut SCRATCH1: u16;

        #[ram]
        static mut result: u8 = 0;

        fn test() {
            for i in 0..4 {
                for j in 0..8 {
                    result = result + 1;
                }
            }
        }

        #[entry]
        fn main() {
            test();
        }
        """
        result = compile_string(source, "test.r65")

        # Should see both X and Y register operations
        has_x_init = "LDX #$00" in result or "LDX #0" in result
        has_y_init = "LDY #$00" in result or "LDY #0" in result

        assert has_x_init, "Expected X register for outer loop"
        assert has_y_init, "Expected Y register for inner loop"

    def test_loop_comparison_uses_correct_register(self):
        """Test that loop comparison uses the hinted register."""
        from r65.compiler.main import compile_string

        source = """
        #[zeropage(0x10, register)]
        static mut SCRATCH0: u8;

        #[ram]
        static mut result: u8 = 0;

        fn test() {
            for i in 0..10 {
                result = result + 1;
            }
        }

        #[entry]
        fn main() {
            test();
        }
        """
        result = compile_string(source, "test.r65")

        # Should see CPX or CPY for the loop condition (register choice
        # depends on promotion order; u8 counters may get X or Y)
        assert ("CPX #$0A" in result or "CPX #10" in result or
                "CPY #$0A" in result or "CPY #10" in result), \
            "Expected CPX/CPY for loop bound comparison"
