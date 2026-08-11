# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests that codegen errors raised during instruction selection are annotated
with the enclosing function name, a source location, and the failing MIR
instruction.

Historically these errors (e.g. "Cannot resolve hardware register X as memory
operand") were raised with source_loc=None, forcing callers to instrument the
compiler just to learn which function/instruction failed.
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import CodegenError, SourceLocation
from r65.compiler.codegen.function_gen import FunctionCodeGenerator
from r65.compiler.codegen.instruction_select import InstructionSelector


class TestAnnotateHelper:
    """Unit tests for FunctionCodeGenerator._annotate_codegen_error."""

    class _Instr:
        def __init__(self, loc):
            self.source_loc = loc

        def __repr__(self):
            return "%1 = Move A : u8"

    class _Func:
        def __init__(self, name, loc):
            self.name = name
            self.source_loc = loc

    def test_fills_missing_source_loc_from_instruction(self):
        instr_loc = SourceLocation(file_path="g.r65", line=7, column=3)
        err = CodegenError("boom")
        assert err.source_loc is None
        FunctionCodeGenerator._annotate_codegen_error(
            err, self._Func("draw", SourceLocation("g.r65", 1, 1)),
            self._Instr(instr_loc))
        assert err.source_loc is instr_loc
        assert "draw" in err.message
        assert "Move A" in err.hint

    def test_falls_back_to_function_loc_when_instr_has_none(self):
        func_loc = SourceLocation(file_path="g.r65", line=1, column=1)
        err = CodegenError("boom")
        FunctionCodeGenerator._annotate_codegen_error(
            err, self._Func("draw", func_loc), self._Instr(None))
        assert err.source_loc is func_loc

    def test_preserves_existing_source_loc(self):
        existing = SourceLocation(file_path="g.r65", line=9, column=2)
        instr_loc = SourceLocation(file_path="g.r65", line=7, column=3)
        err = CodegenError("boom", source_loc=existing)
        FunctionCodeGenerator._annotate_codegen_error(
            err, self._Func("draw", None), self._Instr(instr_loc))
        assert err.source_loc is existing  # not overwritten

    def test_idempotent(self):
        err = CodegenError("boom")
        f = self._Func("draw", SourceLocation("g.r65", 1, 1))
        FunctionCodeGenerator._annotate_codegen_error(err, f, self._Instr(None))
        msg_once = err.message
        FunctionCodeGenerator._annotate_codegen_error(err, f, self._Instr(None))
        assert err.message == msg_once  # no duplicate "(in function ...)"

    def test_refreshes_str(self):
        err = CodegenError("boom")
        FunctionCodeGenerator._annotate_codegen_error(
            err, self._Func("draw", SourceLocation("g.r65", 4, 1)),
            self._Instr(None))
        assert "draw" in str(err)
        assert "g.r65:4:1" in str(err)


class TestEndToEndAnnotation:
    """A codegen error surfacing through generate_function is annotated."""

    def test_forced_error_gets_function_context(self, monkeypatch):
        orig = InstructionSelector.select_instruction

        def boom(self, instr):
            if self.current_function.name == "foo" and not getattr(self, "_boomed", False):
                self._boomed = True
                raise CodegenError("Cannot resolve hardware register X as memory operand")
            return orig(self, instr)

        monkeypatch.setattr(InstructionSelector, "select_instruction", boom)

        src = """
        fn foo(x @ A: u8) -> u8 {
            let y: u8 = x + 1;
            return y;
        }
        """
        with pytest.raises(CodegenError) as exc:
            compile_string(src)
        err = exc.value
        assert "in function 'foo'" in err.message
        assert err.source_loc is not None and err.source_loc.line > 0
        assert err.hint and "MIR instruction" in err.hint
