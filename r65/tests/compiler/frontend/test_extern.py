# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Parser tests for `extern fn`, `extern static`, and `include_asm!`.

Covers grammar acceptance, AST flag propagation, and rejection of malformed
forms (e.g. body present, missing type annotation).
"""

import pytest
from r65.compiler.frontend import parse, ParseError, ast


# ============================================================================
# extern fn
# ============================================================================

def test_extern_fn_near_no_args():
    """Bare `extern fn name();` parses with is_extern=True and body=None."""
    prog = parse("extern fn helper();")
    assert len(prog.items) == 1
    fn = prog.items[0]
    assert isinstance(fn, ast.FunctionDecl)
    assert fn.name == 'helper'
    assert fn.is_extern is True
    assert fn.is_far is False
    assert fn.body is None
    assert fn.params == []
    assert fn.return_type is None


def test_extern_far_fn():
    """`extern far fn` flips is_far on the resulting FunctionDecl."""
    prog = parse("extern far fn sound_engine();")
    fn = prog.items[0]
    assert fn.is_extern is True
    assert fn.is_far is True


def test_extern_fn_with_register_params_and_return():
    """Register-bound params and return types flow through the extern path."""
    src = "extern fn add_u8(a @ A: u8, b @ X: u16) -> u8;"
    prog = parse(src)
    fn = prog.items[0]
    assert fn.is_extern is True
    assert len(fn.params) == 2
    # a @ A
    assert fn.params[0].name == 'a'
    assert isinstance(fn.params[0].binding, ast.Register)
    assert fn.params[0].binding.name == 'A'
    # b @ X
    assert fn.params[1].name == 'b'
    assert isinstance(fn.params[1].binding, ast.Register)
    assert fn.params[1].binding.name == 'X'
    # return type
    assert isinstance(fn.return_type, ast.BasicType)
    assert fn.return_type.name == 'u8'


def test_extern_fn_with_preserves_attribute():
    """`#[preserves(...)]` attaches to extern fns like any other function."""
    src = "#[preserves(X, Y)] extern fn pure_helper();"
    prog = parse(src)
    fn = prog.items[0]
    assert fn.is_extern is True
    assert len(fn.attributes) == 1
    assert fn.attributes[0].name == 'preserves'


def test_extern_fn_with_body_is_rejected():
    """`extern fn` ending in `{ }` is a parse error — externs must be body-less."""
    with pytest.raises(ParseError):
        parse("extern fn nope() { A = 0; }")


# ============================================================================
# extern static
# ============================================================================

def test_extern_static_immutable_array():
    """`extern static NAME: [u8; N];` parses with is_extern=True, is_mut=False."""
    prog = parse("extern static PALETTE: [u8; 32];")
    s = prog.items[0]
    assert isinstance(s, ast.StaticDecl)
    assert s.name == 'PALETTE'
    assert s.is_extern is True
    assert s.is_mut is False
    assert s.initializer is None


def test_extern_static_mut():
    """`extern static mut NAME: T;` carries is_mut=True."""
    prog = parse("extern static mut SCRATCH: [u8; 16];")
    s = prog.items[0]
    assert s.is_extern is True
    assert s.is_mut is True


def test_extern_static_requires_type_annotation():
    """`extern static NAME;` (no type) is a parse error — grammar requires the `: T`."""
    with pytest.raises(ParseError):
        parse("extern static MYSTERY;")


def test_extern_static_rejects_initializer_in_grammar():
    """`extern static FOO: u8 = 5;` is a parse error — initializers are not allowed."""
    with pytest.raises(ParseError):
        parse("extern static FOO: u8 = 5;")


# ============================================================================
# include_asm!
# ============================================================================

def test_include_asm_emits_directive_node():
    """`include_asm!("foo.s");` produces an IncludeAsmStmt with the raw path."""
    prog = parse('include_asm!("vendor/sound.s");')
    node = prog.items[0]
    assert isinstance(node, ast.IncludeAsmStmt)
    assert node.path == 'vendor/sound.s'


def test_include_asm_semicolon_optional():
    """The trailing `;` is optional, matching include!()."""
    # With semicolon
    prog1 = parse('include_asm!("a.s");')
    # Without — also acceptable
    prog2 = parse('include_asm!("a.s")')
    assert isinstance(prog1.items[0], ast.IncludeAsmStmt)
    assert isinstance(prog2.items[0], ast.IncludeAsmStmt)
    assert prog1.items[0].path == prog2.items[0].path == 'a.s'


def test_mixed_program_parses():
    """A realistic interop preamble: extern decls + include_asm + r65 caller."""
    src = """
    extern fn add_asm(a @ A: u8, b @ X: u16) -> u8;
    extern far fn audio_tick();
    extern static PALETTE: [u8; 32];

    include_asm!("game_helpers.s");

    fn main() {
        A = add_asm(1, 2);
        audio_tick();
    }
    """
    prog = parse(src)
    kinds = [type(it).__name__ for it in prog.items]
    assert kinds == [
        'FunctionDecl', 'FunctionDecl', 'StaticDecl',
        'IncludeAsmStmt', 'FunctionDecl',
    ]
    # Both extern fns
    assert prog.items[0].is_extern and prog.items[0].is_far is False
    assert prog.items[1].is_extern and prog.items[1].is_far is True
    # Extern static
    assert prog.items[2].is_extern
    # Caller is not extern
    assert prog.items[4].is_extern is False
    assert prog.items[4].body is not None
