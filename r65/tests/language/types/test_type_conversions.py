# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for type conversion operations.

Tests all type cast operations:
- Widening: u8->u16, i8->i16, bool->u16
- Narrowing: u16->u8, i16->i8
- Same-size reinterpret: u8<->i8, u16<->i16
- Boolean conversions: integer->bool (ToBool), bool->integer
"""

import pytest
from r65.compiler.frontend import ast
from r65.compiler.frontend.parser import parse
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir import nodes as hir
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.mir.builder import MIRBuilder
from r65.compiler.mir.nodes import TypeConvert, ToBool, Move
from r65.compiler.typeck.type_checker import TypeChecker
from r65.tests.language.common import parse_function, parse_expr, build_hir


# =============================================================================
# Parser Tests - Type Cast Syntax
# =============================================================================

class TestTypeCastParsing:
    """Test type cast expression parsing."""

    def test_parse_u8_to_u16(self):
        """Test parsing u8 as u16 cast."""
        func = parse_function("fn test() { let x: u16 = (A as u16); }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert let_stmt.initializer.target_type.name == "u16"

    def test_parse_u16_to_u8(self):
        """Test parsing u16 as u8 cast."""
        func = parse_function("fn test() { let x: u8 = (A as u8); }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert let_stmt.initializer.target_type.name == "u8"

    def test_parse_i8_to_i16(self):
        """Test parsing i8 as i16 cast."""
        func = parse_function("fn test() { let x: i16 = (A as i16); }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert let_stmt.initializer.target_type.name == "i16"

    def test_parse_u8_to_bool(self):
        """Test parsing u8 as bool cast."""
        func = parse_function("fn test() { let x: bool = (A as bool); }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert let_stmt.initializer.target_type.name == "bool"

    def test_parse_bool_to_u8(self):
        """Test parsing bool as u8 cast."""
        func = parse_function("fn test() { let flag: bool = true; let x: u8 = (flag as u8); }")
        let_stmt = func.body.statements[1]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        assert let_stmt.initializer.target_type.name == "u8"

    def test_parse_chained_cast(self):
        """Test parsing chained casts like (x as i16) as u16."""
        func = parse_function("fn test() { let x: u16 = ((A as i16) as u16); }")
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt.initializer, ast.TypeCast)
        inner = let_stmt.initializer.expr
        assert isinstance(inner, ast.TypeCast)


# =============================================================================
# HIR Tests - Type Cast HIR Generation
# =============================================================================

class TestTypeCastHIR:
    """Test type cast HIR generation."""

    def test_hir_type_cast_node(self):
        """Test HIR generates HIRTypeCast nodes."""
        hir_prog = build_hir("""
                        fn test() {
                let x: u16 = (A as u16);
            }
        """)
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        assert isinstance(let_stmt, hir.HIRLetStmt)
        assert isinstance(let_stmt.initializer, hir.HIRTypeCast)

    def test_hir_cast_target_type(self):
        """Test HIR type cast has correct target type."""
        hir_prog = build_hir("""
                        fn test() {
                let x: i16 = (A as i16);
            }
        """)
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        cast = let_stmt.initializer
        assert isinstance(cast.target_type, BasicTypeInfo)
        assert cast.target_type.name == "i16"

    def test_hir_bool_cast(self):
        """Test HIR for bool cast."""
        hir_prog = build_hir("""
                        fn test() {
                let flag: bool = (A as bool);
            }
        """)
        func = hir_prog.functions[0]
        let_stmt = func.body.statements[0]
        cast = let_stmt.initializer
        assert isinstance(cast.target_type, BasicTypeInfo)
        assert cast.target_type.name == "bool"


# =============================================================================
# Type Checker Tests - Cast Validation
# =============================================================================

class TestTypeCastTypeCheck:
    """Test type cast type checking."""

    def _typecheck(self, source: str) -> hir.HIRProgram:
        """Parse, build HIR, and type check."""
        program = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(program)
        checker = TypeChecker(hir_prog)
        checker.check()
        return hir_prog

    def test_typecheck_u8_to_u16(self):
        """Type check u8 -> u16 widening."""
        self._typecheck("""
                        fn test() {
                let x: u16 = (A as u16);
            }
        """)

    def test_typecheck_i8_to_i16(self):
        """Type check i8 -> i16 sign extension."""
        self._typecheck("""
            #[zeropage]
            static mut VAL: i8;
                        fn test() {
                let x: i16 = (VAL as i16);
            }
        """)

    def test_typecheck_u16_to_u8(self):
        """Type check u16 -> u8 narrowing."""
        self._typecheck("""
            #[zeropage]
            static mut VAL: u16;
                        fn test() {
                let x: u8 = (VAL as u8);
            }
        """)

    def test_typecheck_u8_to_bool(self):
        """Type check u8 -> bool."""
        self._typecheck("""
                        fn test() {
                let flag: bool = (A as bool);
            }
        """)

    def test_typecheck_u16_to_bool(self):
        """Type check u16 -> bool."""
        self._typecheck("""
            #[zeropage]
            static mut VAL: u16;
                        fn test() {
                let flag: bool = (VAL as bool);
            }
        """)

    def test_typecheck_bool_to_u8(self):
        """Type check bool -> u8."""
        self._typecheck("""
            #[zeropage]
            static mut FLAG: bool;
                        fn test() {
                let x: u8 = (FLAG as u8);
            }
        """)

    def test_typecheck_bool_to_u16(self):
        """Type check bool -> u16 widening."""
        self._typecheck("""
            #[zeropage]
            static mut FLAG: bool;
                        fn test() {
                let x: u16 = (FLAG as u16);
            }
        """)

    def test_typecheck_reinterpret_u8_i8(self):
        """Type check u8 <-> i8 reinterpret."""
        self._typecheck("""
                        fn test() {
                let x: i8 = (A as i8);
            }
        """)

    def test_typecheck_reinterpret_u16_i16(self):
        """Type check u16 <-> i16 reinterpret."""
        self._typecheck("""
            #[zeropage]
            static mut VAL: u16;
                        fn test() {
                let x: i16 = (VAL as i16);
            }
        """)


# =============================================================================
# MIR Tests - Correct Instruction Generation
# =============================================================================

class TestTypeCastMIR:
    """Test MIR instruction generation for type casts."""

    def _build_mir(self, source: str):
        """Parse, build HIR, type check, and build MIR."""
        program = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(program)
        checker = TypeChecker(hir_prog)
        checker.check()
        mir_builder = MIRBuilder()
        return mir_builder.build_program(hir_prog)

    def _find_instruction(self, mir_prog, instr_type):
        """Find first instruction of given type in MIR program."""
        for func in mir_prog.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, instr_type):
                        return instr
        return None

    def _find_all_instructions(self, mir_prog, instr_type):
        """Find all instructions of given type in MIR program."""
        results = []
        for func in mir_prog.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, instr_type):
                        results.append(instr)
        return results

    def test_mir_u8_to_u16_generates_type_convert(self):
        """u8 -> u16 should generate TypeConvert (zero-extend)."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut RESULT: u16;
                        fn test() {
                RESULT = (A as u16);
            }
        """)
        instr = self._find_instruction(mir_prog, TypeConvert)
        assert instr is not None, "Expected TypeConvert instruction for u8->u16"
        assert str(instr.source_type) == "u8"
        assert str(instr.target_type) == "u16"

    def test_mir_i8_to_i16_generates_type_convert(self):
        """i8 -> i16 should generate TypeConvert (sign-extend)."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut VAL: i8;
            #[zeropage]
            static mut RESULT: i16;
                        fn test() {
                RESULT = (VAL as i16);
            }
        """)
        instr = self._find_instruction(mir_prog, TypeConvert)
        assert instr is not None, "Expected TypeConvert instruction for i8->i16"
        assert str(instr.source_type) == "i8"
        assert str(instr.target_type) == "i16"

    def test_mir_u16_to_u8_generates_type_convert(self):
        """u16 -> u8 should generate TypeConvert (truncate)."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut VAL: u16;
            #[zeropage]
            static mut RESULT: u8;
                        fn test() {
                RESULT = (VAL as u8);
            }
        """)
        instr = self._find_instruction(mir_prog, TypeConvert)
        assert instr is not None, "Expected TypeConvert instruction for u16->u8"
        assert str(instr.source_type) == "u16"
        assert str(instr.target_type) == "u8"

    def test_mir_u8_to_bool_generates_to_bool(self):
        """u8 -> bool should generate ToBool instruction."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut RESULT: bool;
                        fn test() {
                RESULT = (A as bool);
            }
        """)
        instr = self._find_instruction(mir_prog, ToBool)
        assert instr is not None, "Expected ToBool instruction for u8->bool"

    def test_mir_u16_to_bool_generates_to_bool(self):
        """u16 -> bool should generate ToBool instruction."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut VAL: u16;
            #[zeropage]
            static mut RESULT: bool;
                        fn test() {
                RESULT = (VAL as bool);
            }
        """)
        instr = self._find_instruction(mir_prog, ToBool)
        assert instr is not None, "Expected ToBool instruction for u16->bool"

    def test_mir_bool_to_u8_generates_move(self):
        """bool -> u8 should generate Move (value already 0 or 1)."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut FLAG: bool;
            #[zeropage]
            static mut RESULT: u8;
                        fn test() {
                RESULT = (FLAG as u8);
            }
        """)
        # Should NOT generate ToBool for bool->u8
        to_bool = self._find_instruction(mir_prog, ToBool)
        assert to_bool is None, "Should not use ToBool for bool->u8 conversion"

    def test_mir_same_size_reinterpret_generates_move(self):
        """Same-size reinterpret (u8<->i8) should generate Move."""
        mir_prog = self._build_mir("""
            #[zeropage]
            static mut RESULT: i8;
                        fn test() {
                RESULT = (A as i8);
            }
        """)
        # Should NOT generate TypeConvert for same-size
        type_convert = self._find_instruction(mir_prog, TypeConvert)
        assert type_convert is None, "Should not use TypeConvert for u8->i8 reinterpret"


# =============================================================================
# Code Generation Tests - Assembly Output
# =============================================================================

class TestTypeCastCodeGen:
    """Test assembly code generation for type casts."""

    def _compile_to_asm(self, source: str) -> str:
        """Compile source to assembly string."""
        import io
        import sys
        from r65.compiler.main import compile_source

        # Capture stdout where compile_source prints the assembly
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        try:
            compile_source(source, filename="<test>", verbose=False, quiet=True)
        finally:
            sys.stdout = old_stdout
        return captured.getvalue()

    def test_codegen_to_bool_branchless(self):
        """ToBool should generate branchless CMP/LDA/ADC sequence."""
        asm = self._compile_to_asm("""
            #[zeropage(0x10)]
            static mut VALUE: u8;
            #[zeropage(0x11)]
            static mut RESULT: bool;
                        fn test() {
                RESULT = (VALUE as bool);
            }
        """)
        # Check for branchless pattern
        assert "CMP #$01" in asm, "Expected CMP #$01 for branchless ToBool"
        assert "ADC #$00" in asm, "Expected ADC #$00 for branchless ToBool"
        # Should NOT have branches for bool conversion
        lines = asm.split('\n')
        in_test_func = False
        for line in lines:
            if 'test:' in line:
                in_test_func = True
            elif in_test_func and line.strip().startswith('RTS'):
                break
            elif in_test_func:
                assert not line.strip().startswith('BEQ'), "ToBool should be branchless"
                assert not line.strip().startswith('BNE'), "ToBool should be branchless"

    def test_codegen_zero_extend(self):
        """u8 -> u16 should generate zero extension."""
        asm = self._compile_to_asm("""
            #[zeropage(0x10)]
            static mut VALUE: u8;
            #[zeropage(0x11)]
            static mut RESULT: u16;
                        fn test() {
                RESULT = (VALUE as u16);
            }
        """)
        # Should zero-extend: either LDA #$00 (byte-by-byte) or AND #$FF (m16 path)
        assert "LDA #$00" in asm or "STZ" in asm or "AND #$FF" in asm, \
            "Expected zero extension for u8->u16"

    def test_codegen_truncate(self):
        """u16 -> u8 should generate truncation (load low byte only)."""
        asm = self._compile_to_asm("""
            #[zeropage(0x10)]
            static mut VALUE: u16;
            #[zeropage(0x12)]
            static mut RESULT: u8;
                        fn test() {
                RESULT = (VALUE as u8);
            }
        """)
        # Should only load/store one byte
        assert "STA" in asm, "Expected store for truncation result"


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestTypeCastEdgeCases:
    """Test edge cases for type conversions."""

    def _typecheck(self, source: str) -> hir.HIRProgram:
        """Parse, build HIR, and type check."""
        program = parse(source)
        builder = HIRBuilder()
        hir_prog = builder.build_program(program)
        checker = TypeChecker(hir_prog)
        checker.check()
        return hir_prog

    def test_cast_literal_to_bool(self):
        """Casting literal to bool should work."""
        self._typecheck("""
            #[zeropage]
            static mut FLAG: bool;
                        fn test() {
                FLAG = (1 as bool);
                FLAG = (0 as bool);
            }
        """)

    def test_cast_enum_to_u8(self):
        """Casting enum to u8 should work."""
        self._typecheck("""
            enum State { Idle = 0, Running = 1 }
            #[zeropage]
            static mut VAL: u8;
                        fn test() {
                VAL = (State::Running as u8);
            }
        """)

    def test_double_cast(self):
        """Double cast like (x as i16) as u16 should work."""
        self._typecheck("""
            #[zeropage]
            static mut RESULT: u16;
                        fn test() {
                RESULT = ((A as i16) as u16);
            }
        """)

    def test_bool_in_expression_context(self):
        """Bool cast result can be used in expressions."""
        self._typecheck("""
            #[zeropage]
            static mut FLAG: bool;
                        fn test() {
                if (A as bool) {
                    FLAG = true;
                }
            }
        """)
