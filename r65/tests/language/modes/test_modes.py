"""
Comprehensive processor mode tests for R65.

Tests mode annotations (#[mode(m8/m16, x8/x16)]), mode transitions,
bank attributes, interrupt handlers, and mode control built-ins.
"""

import pytest
from r65.compiler.frontend.parser import parse
from r65.compiler.frontend import ast
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.hir import nodes as hir


# =============================================================================
# Helper Functions
# =============================================================================

def parse_program(source: str) -> ast.Program:
    """Parse source and return the program."""
    return parse(source)


def parse_function(source: str) -> ast.FunctionDecl:
    """Parse source and return the first function declaration."""
    program = parse(source)
    for item in program.items:
        if isinstance(item, ast.FunctionDecl):
            return item
    raise AssertionError("No function found")


def parse_statement(source: str) -> ast.Statement:
    """Parse a function with a single statement and return that statement."""
    func = parse_function(f"fn test() {{ {source} }}")
    assert len(func.body.statements) == 1
    return func.body.statements[0]


def get_attr(decl, name: str) -> ast.Attribute:
    """Get an attribute by name from a declaration."""
    for attr in decl.attributes:
        if attr.name == name:
            return attr
    return None


def get_attr_arg_by_name(attr: ast.Attribute, name: str):
    """Get a named attribute argument."""
    for arg in attr.args:
        if arg.name == name:
            return arg.value
    return None


def get_attr_positional_args(attr: ast.Attribute) -> list:
    """Get positional (unnamed) attribute arguments."""
    return [arg.value for arg in attr.args if arg.name is None]


def build_hir(source: str) -> hir.HIRProgram:
    """Parse source and build HIR."""
    program = parse(source)
    builder = HIRBuilder()
    return builder.build_program(program)


def get_hir_function(hir_prog: hir.HIRProgram, name: str) -> hir.HIRFunctionDecl:
    """Get a function by name from HIR program."""
    for func in hir_prog.functions:
        if func.name == name:
            return func
    raise KeyError(f"Function '{name}' not found")


# =============================================================================
# Test Classes
# =============================================================================

class TestAccumulatorMode:
    """Tests for accumulator mode (m8/m16)."""

    def test_mode_m8(self):
        """Test 8-bit accumulator mode."""
        func = parse_function("#[mode(m8)] fn process() { }")
        attr = get_attr(func, "mode")
        assert attr is not None
        args = get_attr_positional_args(attr)
        assert any(arg.name == "m8" for arg in args)

    def test_mode_m16(self):
        """Test 16-bit accumulator mode."""
        func = parse_function("#[mode(m16)] fn process() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        assert any(arg.name == "m16" for arg in args)

    def test_mode_m8_affects_a_register(self):
        """Test that m8 mode implies A is u8."""
        func = parse_function("""
            #[mode(m8)]
            fn byte_ops() {
                A = 0xFF;
            }
        """)
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_m16_affects_a_register(self):
        """Test that m16 mode implies A is u16."""
        func = parse_function("""
            #[mode(m16)]
            fn word_ops() {
                A = 0xFFFF;
            }
        """)
        attr = get_attr(func, "mode")
        assert attr is not None


class TestIndexMode:
    """Tests for index register mode (x8/x16)."""

    def test_mode_x8(self):
        """Test 8-bit index register mode."""
        func = parse_function("#[mode(x8)] fn process() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        assert any(arg.name == "x8" for arg in args)

    def test_mode_x16(self):
        """Test 16-bit index register mode."""
        func = parse_function("#[mode(x16)] fn process() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        assert any(arg.name == "x16" for arg in args)

    def test_mode_x8_affects_xy_registers(self):
        """Test that x8 mode implies X/Y are u8."""
        func = parse_function("""
            #[mode(x8)]
            fn byte_index() {
                X = 0xFF;
                Y = 0xFF;
            }
        """)
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_mode_x16_affects_xy_registers(self):
        """Test that x16 mode implies X/Y are u16."""
        func = parse_function("""
            #[mode(x16)]
            fn word_index() {
                X = 0xFFFF;
                Y = 0xFFFF;
            }
        """)
        attr = get_attr(func, "mode")
        assert attr is not None


class TestCombinedModes:
    """Tests for combined mode annotations."""

    def test_mode_m8_x8(self):
        """Test 8-bit accumulator and index mode."""
        func = parse_function("#[mode(m8, x8)] fn all_8bit() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        names = [arg.name for arg in args]
        assert "m8" in names
        assert "x8" in names

    def test_mode_m16_x16(self):
        """Test 16-bit accumulator and index mode."""
        func = parse_function("#[mode(m16, x16)] fn all_16bit() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        names = [arg.name for arg in args]
        assert "m16" in names
        assert "x16" in names

    def test_mode_m8_x16(self):
        """Test mixed mode: 8-bit accumulator, 16-bit index."""
        func = parse_function("#[mode(m8, x16)] fn mixed() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        names = [arg.name for arg in args]
        assert "m8" in names
        assert "x16" in names

    def test_mode_m16_x8(self):
        """Test mixed mode: 16-bit accumulator, 8-bit index."""
        func = parse_function("#[mode(m16, x8)] fn mixed() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        names = [arg.name for arg in args]
        assert "m16" in names
        assert "x8" in names

    def test_partial_mode_m8_only(self):
        """Test partial mode specification (m8 only)."""
        func = parse_function("#[mode(m8)] fn partial() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        # Only m8 specified
        assert len(args) == 1
        assert args[0].name == "m8"

    def test_partial_mode_x16_only(self):
        """Test partial mode specification (x16 only)."""
        func = parse_function("#[mode(x16)] fn partial() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        assert len(args) == 1
        assert args[0].name == "x16"


class TestModeTransitions:
    """Tests for mode transition options."""

    def test_transition_none(self):
        """Test transition=none (default, manual control)."""
        func = parse_function("#[mode(m8, x8, transition=none)] fn manual() { }")
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans is not None
        assert trans.name == "none"

    def test_transition_inline(self):
        """Test transition=inline (callee manages mode)."""
        func = parse_function("#[mode(m16, x16, transition=inline)] fn safe() { }")
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans is not None
        assert trans.name == "inline"

    def test_transition_caller(self):
        """Test transition=caller (caller manages mode)."""
        func = parse_function("#[mode(m16, x16, transition=caller)] fn batch() { }")
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans is not None
        assert trans.name == "caller"

    def test_mode_with_transition_and_all_modes(self):
        """Test full mode specification with transition."""
        func = parse_function("#[mode(m8, x8, transition=inline)] fn full() { }")
        attr = get_attr(func, "mode")
        args = get_attr_positional_args(attr)
        trans = get_attr_arg_by_name(attr, "transition")
        assert len(args) == 2
        assert trans.name == "inline"

    def test_transition_without_explicit_modes(self):
        """Test transition option alone."""
        func = parse_function("#[mode(transition=none)] fn minimal() { }")
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans.name == "none"


class TestBankAttribute:
    """Tests for #[bank(n)] attribute."""

    def test_bank_basic(self):
        """Test basic bank attribute."""
        func = parse_function("#[bank(1)] far fn bank1_code() { }")
        attr = get_attr(func, "bank")
        assert attr is not None
        args = get_attr_positional_args(attr)
        assert args[0].value == 1

    def test_bank_number(self):
        """Test various bank numbers."""
        for bank_num in [0, 1, 2, 127]:
            func = parse_function(f"#[bank({bank_num})] far fn code() {{ }}")
            attr = get_attr(func, "bank")
            args = get_attr_positional_args(attr)
            assert args[0].value == bank_num

    def test_bank_with_data_bank_none(self):
        """Test bank with data_bank=none (default)."""
        func = parse_function("#[bank(1, data_bank=none)] far fn code() { }")
        attr = get_attr(func, "bank")
        db = get_attr_arg_by_name(attr, "data_bank")
        assert db is not None
        assert db.name == "none"

    def test_bank_with_data_bank_inline(self):
        """Test bank with data_bank=inline."""
        func = parse_function("#[bank(1, data_bank=inline)] far fn gfx() { }")
        attr = get_attr(func, "bank")
        db = get_attr_arg_by_name(attr, "data_bank")
        assert db.name == "inline"

    def test_bank_with_data_bank_caller(self):
        """Test bank with data_bank=caller."""
        func = parse_function("#[bank(2, data_bank=caller)] far fn decompress() { }")
        attr = get_attr(func, "bank")
        db = get_attr_arg_by_name(attr, "data_bank")
        assert db.name == "caller"

    def test_bank_requires_far(self):
        """Test that bank attribute is used with far functions."""
        func = parse_function("#[bank(1)] far fn cross_bank() { }")
        assert func.is_far == True
        assert get_attr(func, "bank") is not None


class TestFarFunctions:
    """Tests for far function declarations."""

    def test_far_function_basic(self):
        """Test basic far function."""
        func = parse_function("far fn remote() { }")
        assert func.is_far == True

    def test_far_function_with_params(self):
        """Test far function with parameters."""
        func = parse_function("far fn remote(val @ A: u8) { }")
        assert func.is_far == True
        assert len(func.params) == 1

    def test_far_function_with_return(self):
        """Test far function with return type."""
        func = parse_function("far fn remote() -> u8 { return A; }")
        assert func.is_far == True
        assert func.return_type is not None

    def test_far_function_with_bank(self):
        """Test far function with bank attribute."""
        func = parse_function("#[bank(3)] far fn sound_engine() { }")
        assert func.is_far == True
        attr = get_attr(func, "bank")
        assert attr is not None

    def test_near_function_default(self):
        """Test that functions without far are near (default)."""
        func = parse_function("fn local() { }")
        assert func.is_far == False


class TestInterruptHandlers:
    """Tests for interrupt handler declarations."""

    def test_interrupt_nmi(self):
        """Test NMI interrupt handler."""
        func = parse_function("#[interrupt(nmi)] fn vblank() { }")
        attr = get_attr(func, "interrupt")
        assert attr is not None
        args = get_attr_positional_args(attr)
        assert args[0].name == "nmi"

    def test_interrupt_irq(self):
        """Test IRQ interrupt handler."""
        func = parse_function("#[interrupt(irq)] fn timer() { }")
        attr = get_attr(func, "interrupt")
        args = get_attr_positional_args(attr)
        assert args[0].name == "irq"

    def test_interrupt_brk(self):
        """Test BRK interrupt handler."""
        func = parse_function("#[interrupt(brk)] fn breakpoint() { }")
        attr = get_attr(func, "interrupt")
        args = get_attr_positional_args(attr)
        assert args[0].name == "brk"

    def test_interrupt_cop(self):
        """Test COP interrupt handler."""
        func = parse_function("#[interrupt(cop)] fn syscall() { }")
        attr = get_attr(func, "interrupt")
        args = get_attr_positional_args(attr)
        assert args[0].name == "cop"

    def test_interrupt_abort(self):
        """Test ABORT interrupt handler."""
        func = parse_function("#[interrupt(abort)] fn abort_handler() { }")
        attr = get_attr(func, "interrupt")
        args = get_attr_positional_args(attr)
        assert args[0].name == "abort"

    def test_interrupt_preserve_true(self):
        """Test interrupt with preserve=true (default)."""
        func = parse_function("#[interrupt(nmi, preserve=true)] fn safe_nmi() { }")
        attr = get_attr(func, "interrupt")
        preserve = get_attr_arg_by_name(attr, "preserve")
        assert preserve is not None
        assert preserve.value == True

    def test_interrupt_preserve_false(self):
        """Test interrupt with preserve=false (manual control)."""
        func = parse_function("#[interrupt(irq, preserve=false)] fn fast_irq() { }")
        attr = get_attr(func, "interrupt")
        preserve = get_attr_arg_by_name(attr, "preserve")
        assert preserve is not None
        assert preserve.value == False

    def test_interrupt_with_preserves_attribute(self):
        """Test interrupt with explicit preserves attribute."""
        func = parse_function("""
            #[interrupt(nmi)]
            #[preserves(A, X)]
            fn custom_nmi() { }
        """)
        int_attr = get_attr(func, "interrupt")
        pres_attr = get_attr(func, "preserves")
        assert int_attr is not None
        assert pres_attr is not None


class TestEntryAttribute:
    """Tests for #[entry] attribute."""

    def test_entry_basic(self):
        """Test basic entry point."""
        func = parse_function("#[entry] fn main() { }")
        attr = get_attr(func, "entry")
        assert attr is not None

    def test_entry_with_mode(self):
        """Test entry point with mode."""
        func = parse_function("#[entry] #[mode(m8, x8)] fn main() { }")
        assert get_attr(func, "entry") is not None
        assert get_attr(func, "mode") is not None

    def test_entry_function_body(self):
        """Test entry function with initialization code."""
        func = parse_function("""
            #[entry]
            fn main() {
                A = 0;
                X = 0;
                Y = 0;
            }
        """)
        assert get_attr(func, "entry") is not None
        assert len(func.body.statements) == 3


class TestModeControlBuiltins:
    """Tests for mode control built-in functions."""

    def test_sep_instruction(self):
        """Test SEP() built-in for setting status bits."""
        stmt = parse_statement("SEP(0x30);")
        assert isinstance(stmt, ast.ExprStmt)
        call = stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert call.func.name == "SEP"
        assert call.args[0].value == 0x30

    def test_rep_instruction(self):
        """Test REP() built-in for resetting status bits."""
        stmt = parse_statement("REP(0x30);")
        call = stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert call.func.name == "REP"
        assert call.args[0].value == 0x30

    def test_sep_m8_mode(self):
        """Test SEP(0x20) for m8 mode."""
        stmt = parse_statement("SEP(0x20);")
        call = stmt.expr
        assert call.args[0].value == 0x20

    def test_sep_x8_mode(self):
        """Test SEP(0x10) for x8 mode."""
        stmt = parse_statement("SEP(0x10);")
        call = stmt.expr
        assert call.args[0].value == 0x10

    def test_rep_m16_mode(self):
        """Test REP(0x20) for m16 mode."""
        stmt = parse_statement("REP(0x20);")
        call = stmt.expr
        assert call.args[0].value == 0x20

    def test_rep_x16_mode(self):
        """Test REP(0x10) for x16 mode."""
        stmt = parse_statement("REP(0x10);")
        call = stmt.expr
        assert call.args[0].value == 0x10

    def test_xba_instruction(self):
        """Test xba() built-in for exchanging B and A."""
        stmt = parse_statement("xba();")
        call = stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert call.func.name == "xba"
        assert len(call.args) == 0

    def test_wai_instruction(self):
        """Test wai() built-in for wait for interrupt."""
        stmt = parse_statement("wai();")
        call = stmt.expr
        assert call.func.name == "wai"

    def test_stp_instruction(self):
        """Test stp() built-in for stop processor."""
        stmt = parse_statement("stp();")
        call = stmt.expr
        assert call.func.name == "stp"


class TestBlockMoveBuiltins:
    """Tests for block move built-in functions."""

    def test_mvn_instruction(self):
        """Test mvn() built-in for move block negative."""
        stmt = parse_statement("mvn(0x00, 0x7E);")
        call = stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert call.func.name == "mvn"
        assert len(call.args) == 2
        assert call.args[0].value == 0x00
        assert call.args[1].value == 0x7E

    def test_mvp_instruction(self):
        """Test mvp() built-in for move block positive."""
        stmt = parse_statement("mvp(0x7E, 0x00);")
        call = stmt.expr
        assert call.func.name == "mvp"
        assert len(call.args) == 2


class TestCopBuiltin:
    """Tests for COP built-in function."""

    def test_cop_instruction(self):
        """Test cop() built-in for software interrupt."""
        stmt = parse_statement("cop(0x00);")
        call = stmt.expr
        assert isinstance(call, ast.FunctionCall)
        assert call.func.name == "cop"
        assert call.args[0].value == 0x00

    def test_cop_with_signature(self):
        """Test cop() with different signature byte."""
        stmt = parse_statement("cop(0x80);")
        call = stmt.expr
        assert call.args[0].value == 0x80


class TestInlineAssembly:
    """Tests for inline assembly."""

    def test_asm_single_instruction(self):
        """Test single instruction asm."""
        stmt = parse_statement('asm!("WAI");')
        assert isinstance(stmt, ast.AsmStmt)
        assert "WAI" in stmt.instructions

    def test_asm_multiple_instructions(self):
        """Test multiple instruction asm."""
        stmt = parse_statement('asm!("PHP", "WAI", "PLP");')
        assert isinstance(stmt, ast.AsmStmt)
        assert len(stmt.instructions) == 3

    def test_asm_in_function(self):
        """Test asm in function body."""
        func = parse_function("""
            fn wait() {
                asm!("WAI");
            }
        """)
        stmt = func.body.statements[0]
        assert isinstance(stmt, ast.AsmStmt)


class TestModeHIR:
    """Tests for HIR generation of mode-related constructs."""

    def test_hir_mode_attribute(self):
        """Test HIR for mode attribute."""
        hir_prog = build_hir("#[mode(m8, x8)] fn process() { }")
        func = get_hir_function(hir_prog, "process")
        assert func.mode_attr is not None

    def test_hir_bank_attribute(self):
        """Test HIR for bank attribute."""
        hir_prog = build_hir("#[bank(1)] far fn remote() { }")
        func = get_hir_function(hir_prog, "remote")
        assert func.bank_attr is not None

    def test_hir_interrupt_attribute(self):
        """Test HIR for interrupt attribute."""
        hir_prog = build_hir("#[interrupt(nmi)] fn vblank() { }")
        func = get_hir_function(hir_prog, "vblank")
        assert func.interrupt_attr is not None

    def test_hir_entry_attribute(self):
        """Test HIR for entry attribute."""
        hir_prog = build_hir("#[entry] fn main() { }")
        func = get_hir_function(hir_prog, "main")
        assert func.is_entry == True

    def test_hir_far_function(self):
        """Test HIR for far function."""
        hir_prog = build_hir("far fn remote() { }")
        func = get_hir_function(hir_prog, "remote")
        assert func.is_far == True


class TestModePatterns:
    """Tests for common mode usage patterns."""

    def test_mode_switch_pattern(self):
        """Test typical mode switching pattern."""
        func = parse_function("""
            fn switch_modes() {
                SEP(0x30);
                A = 0xFF;
                REP(0x30);
                A = 0xFFFF;
            }
        """)
        assert len(func.body.statements) == 4

    def test_safe_16bit_call_pattern(self):
        """Test safe 16-bit function pattern."""
        prog = parse_program("""
            #[mode(m16, x16, transition=inline)]
            fn safe_16bit() {
                A = 0x1234;
            }
        """)
        func = prog.items[0]
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans.name == "inline"

    def test_interrupt_with_mode_restoration(self):
        """Test interrupt handler pattern."""
        func = parse_function("""
            #[interrupt(nmi)]
            fn vblank_handler() {
                SEP(0x20);
                A = 0x0F;
            }
        """)
        assert get_attr(func, "interrupt") is not None

    def test_cross_bank_call_pattern(self):
        """Test cross-bank call pattern."""
        prog = parse_program("""
            #[bank(1, data_bank=inline)]
            far fn sound_update() { }

            fn main_loop() {
                sound_update();
            }
        """)
        sound_func = prog.items[0]
        assert sound_func.is_far == True
        assert get_attr(sound_func, "bank") is not None


class TestModeEdgeCases:
    """Tests for edge cases in mode handling."""

    def test_multiple_mode_attributes(self):
        """Test function with multiple attributes."""
        func = parse_function("""
            #[mode(m8, x8)]
            #[preserves(A, X, Y)]
            fn multi_attr() { }
        """)
        assert get_attr(func, "mode") is not None
        assert get_attr(func, "preserves") is not None

    def test_mode_and_bank_together(self):
        """Test mode with bank attribute."""
        func = parse_function("""
            #[bank(2)]
            #[mode(m16, x16)]
            far fn banked_16bit() { }
        """)
        assert get_attr(func, "bank") is not None
        assert get_attr(func, "mode") is not None
        assert func.is_far == True

    def test_interrupt_and_mode_together(self):
        """Test interrupt with mode attribute."""
        func = parse_function("""
            #[interrupt(irq)]
            #[mode(m8, x8)]
            fn fast_irq() { }
        """)
        assert get_attr(func, "interrupt") is not None
        assert get_attr(func, "mode") is not None

    def test_entry_with_all_attributes(self):
        """Test entry with multiple attributes."""
        func = parse_function("""
            #[entry]
            #[mode(m8, x8)]
            fn main() { }
        """)
        assert get_attr(func, "entry") is not None
        assert get_attr(func, "mode") is not None

    def test_no_mode_attribute(self):
        """Test function without mode attribute (convention-based)."""
        func = parse_function("fn unspecified() { }")
        assert get_attr(func, "mode") is None


class TestModeParseErrors:
    """Tests for mode-related parse errors."""

    def test_invalid_mode_parses(self):
        """Test that invalid mode names parse (type checker catches)."""
        # Parser allows any identifier; type checker validates
        func = parse_function("#[mode(invalid)] fn bad() { }")
        attr = get_attr(func, "mode")
        assert attr is not None

    def test_invalid_transition_parses(self):
        """Test that invalid transition names parse."""
        func = parse_function("#[mode(m8, transition=invalid)] fn bad() { }")
        attr = get_attr(func, "mode")
        trans = get_attr_arg_by_name(attr, "transition")
        assert trans.name == "invalid"

    def test_invalid_interrupt_vector_parses(self):
        """Test that invalid interrupt vector parses."""
        func = parse_function("#[interrupt(invalid)] fn bad() { }")
        attr = get_attr(func, "interrupt")
        args = get_attr_positional_args(attr)
        assert args[0].name == "invalid"

    def test_bank_without_far_parses(self):
        """Test bank without far parses (type checker catches)."""
        func = parse_function("#[bank(1)] fn not_far() { }")
        assert func.is_far == False
        assert get_attr(func, "bank") is not None

    def test_sep_without_arg_parses(self):
        """Test SEP without argument parses (type checker catches)."""
        # Parser allows function calls with any number of args
        func = parse_function("fn test() { SEP(); }")
        stmt = func.body.statements[0]
        call = stmt.expr
        assert len(call.args) == 0  # No args - type checker catches

    def test_asm_without_string_fails(self):
        """Test asm without string argument fails."""
        with pytest.raises(Exception):
            parse("fn test() { asm!(); }")
