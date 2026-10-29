# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`#[clobbers(...)]`: the registers a function may leave changed.

An extern's list is its whole X/Y contract (none listed: both kept); a
function with a body is summarized from its code, the list added on top and
standing in for `asm!` text. A listed D or DBR is saved around each call.
"""

import pytest

from r65.compiler.codegen.codegen import ProgramCodeGenerator
from r65.compiler.errors import CompilerError
from r65.compiler.main import compile_string, run_frontend
from r65.compiler.mir.builder import MIRBuilder

HEADER = (
    '#[snesrom(name="T", lorom)]\n'
    '#[stack(0x0100, 0x01FF)]\n'
    '#[zeropage]\n'
    'static mut G: u16;\n'
)


def _loop_calling(decls: str, call: str) -> str:
    """A program whose Y loop counter is live across *call*."""
    return HEADER + decls + (
        '#[entry]\n'
        'fn main() -> ! {\n'
        '    loop {\n'
        '        let mut i: u16 = 0;\n'
        f'        while i < 10 {{ {call} G = i; i = i + 1; }}\n'
        '    }\n'
        '}\n')


def _generate(src: str, opt_level: int = 1):
    hir = run_frontend(src, "<test>").hir_program
    gen = ProgramCodeGenerator()
    asm = gen.generate(MIRBuilder().build_program(hir), opt_level=opt_level)
    return asm, {name: regs & {'X', 'Y'}
                 for name, regs in gen.func_gen.func_gen.register_summaries.items()}


def _main(asm: str) -> str:
    start = asm.index("\nmain:")
    return asm[start:asm.index("\n\n", start + 1)]


@pytest.mark.parametrize("opt_level", [0, 1])
def test_bare_extern_keeps_index_registers(opt_level):
    asm, _ = _generate(_loop_calling('extern fn ext();\n', 'ext();'), opt_level)
    assert "JSL ext" in asm
    assert "PHY" not in _main(asm)


def test_listed_register_is_saved():
    asm, _ = _generate(_loop_calling('#[clobbers(Y)]\nextern fn ext();\n', 'ext();'))
    assert "PHY" in _main(asm)
    asm, _ = _generate(_loop_calling('#[clobbers(X)]\nextern fn ext();\n', 'ext();'))
    assert "PHY" not in _main(asm)


def test_stack_arg_cleanup_keeps_index_registers():
    """The pushed argument comes off without pulling into X or Y."""
    asm, _ = _generate(_loop_calling('extern fn ext(v: u16);\n', 'ext(G);'))
    main = _main(asm)
    assert "JSL ext" in main
    assert "PLX" not in main and "PLY" not in main


def test_summaries():
    """An extern's list is its summary, and a caller's takes it; an `asm!`
    function's list stands in for the text, its own code still read."""
    src = HEADER + (
        'extern fn ext();\n'
        'fn wraps_ext() { ext(); }\n'
        '#[clobbers()]\n'
        'fn trusted_asm() { asm!("LDX #$0000"); }\n'
        'fn untrusted_asm() { asm!("NOP"); }\n'
        '#[clobbers(Y)]\n'
        'fn asm_and_x() { X = G; asm!("NOP"); }\n'
        '#[entry]\n'
        'fn main() -> ! {\n'
        '    loop { wraps_ext(); trusted_asm(); untrusted_asm(); asm_and_x(); }\n'
        '}\n')
    _, summaries = _generate(src)
    assert summaries['ext'] == frozenset()
    assert summaries['wraps_ext'] == frozenset()
    assert summaries['trusted_asm'] == frozenset()
    assert summaries['untrusted_asm'] == frozenset({'X', 'Y'})
    assert summaries['asm_and_x'] == frozenset({'X', 'Y'})


def test_dbr_saved_around_call():
    asm, _ = _generate(_loop_calling('#[clobbers(DBR)]\nextern fn ext();\n', 'ext();'))
    lines = [l.split(';')[0].strip() for l in _main(asm).splitlines()]
    call = lines.index("JSL ext")
    assert lines[call - 1] == "PHB" and lines[call + 1] == "PLB"


def test_d_restored_after_stack_args_before_results():
    """The pushed argument is above the saved D, so it comes off first; D is
    back before the result is stored."""
    asm, _ = _generate(_loop_calling(
        '#[clobbers(D)]\nextern fn ext(v: u8) -> u8;\n', 'G = ext(1);'))
    lines = [l.split(';')[0].strip() for l in _main(asm).splitlines()]
    call = lines.index("JSL ext")
    pld = lines.index("PLD")
    assert lines.index("PHD") < call < pld
    assert "PLA" in lines[call:pld]
    assert not any(l.startswith("ST") and not l.endswith(",S") for l in lines[call:pld])


def test_never_returning_callee_is_not_wrapped():
    src = HEADER + (
        '#[clobbers(DBR)]\nextern fn ext() -> !;\n'
        '#[entry]\nfn main() -> ! { ext(); }\n')
    assert "PHB" not in _main(compile_string(src))


def test_databank_caller_bracket_restores_dbr_already():
    src = HEADER + (
        '#[bank(1)]\n'
        '#[clobbers(DBR)]\n#[mode(databank=caller)]\nfn far_dbr() { G = 1; }\n'
        '#[bank(0)]\n'
        '#[entry]\nfn main() -> ! { loop { far_dbr(); } }\n')
    assert _main(compile_string(src)).count("PHB") == 1


@pytest.mark.parametrize("decls, message", [
    ('#[clobbers(X)]\n#[interrupt(nmi)]\nfn nmi() {}\n', "reached by a vector"),
    ('#[clobbers(X)]\n#[preserves(X)]\nfn f() {}\n', "both preserves and clobbers X"),
    ('#[bank(1)]\n#[clobbers(DBR)]\n#[mode(databank=inline)]\nfn f() {}\n#[bank(0)]\n',
     "databank=inline"),
    ('#[clobbers(D)]\nfn f() {}\nstatic TABLE: [fn(); 1] = [f];\n', "used as a value"),
    ('trait T { fn m(*self); }\nstruct Sprite { a: u8 }\n'
     'impl T for Sprite {\n    #[clobbers(DBR)]\n    fn m(*self) {}\n}\n', "trait method"),
])
def test_rejected(decls, message):
    src = HEADER + decls + '#[entry]\nfn main() -> ! { loop {} }\n'
    with pytest.raises(CompilerError, match=message):
        compile_string(src)
