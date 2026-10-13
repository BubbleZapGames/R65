# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Near-to-far pointer parameter promotion (typeck/far_pointer_params.py)."""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir import HIRAddressOf, HIRExprStmt
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck.type_checker import TypeChecker


def test_cast_is_stripped_on_every_call_not_just_the_promoting_one():
    """`fill` is promoted by `a`'s call; `b`'s `&BUF2 as *u8` must still lose
    its near cast, or the bank byte of a WRAM address is dropped."""
    hir = HIRBuilder().build_program(expand_macros(parse("""
        #[ram] static mut BUF: [u8; 4];
        #[ram] static mut BUF2: [u8; 4];
        fn fill(ptr: *u8) { *ptr = 1; }
        fn a() { fill(&BUF); }
        fn b() { fill(&BUF2 as *u8); }
    """)))
    checker = TypeChecker(hir)
    checker.check()

    funcs = {f.name: f for f in hir.all_functions()}
    assert funcs['fill'].parameters[0].param_type.is_far
    assert [(p.function, p.parameter.name) for p in checker.far_pointer_promotions] == [('fill', 'ptr')]
    call = funcs['b'].body.statements[0]
    assert isinstance(call, HIRExprStmt)
    assert isinstance(call.expr.args[0], HIRAddressOf)
