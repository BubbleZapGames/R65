# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Impl method bodies must be linted, and reachable through method calls.

Root cause (fixed): both the linter and the call-graph builder filtered
``program.declarations`` with ``isinstance(decl, HIRFunctionDecl)``. Impl
methods are desugared to ``HIRFunctionDecl``s named ``Struct__method`` but
live inside their ``HIRImplDecl``, so they were never walked by any rule and
were absent from the call graph entirely.

On top of that, ``_record_call`` resolved callees only via ``call.func``.
The type checker rewrites a method call to ``HIRFunctionCall(func=None)`` with
the target in ``method_call_info['mangled_name']``, so every method call
degraded to "indirect" and contributed nothing to reachability.

Net effect: a PPU write inside a stdlib method reached from ``main`` — e.g.
``Console::print`` writing ``$2118`` during active rendering, which is a real
VRAM-corrupting bug — was invisible to C001, while the identical write in a
free function was flagged.
"""

from r65.compiler.frontend.parser import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.hir.builder import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.lint.config import LintConfig
from r65.compiler.lint.linter import run_lint
from r65.compiler.lint.call_graph import build_call_graph
from r65.compiler.lint.rule_kinds.reachability_forbidden_access import from_config


SOURCE = '''
#[hw(0x2118)]
static mut VMDATAL: u8;

struct Blitter { n: u8 }

#[ram] static mut BLIT: Blitter;

fn plain_write() {
    VMDATAL = 1;
}

impl Blitter {
    far fn method_write(far *self) {
        VMDATAL = 2;
    }
}

#[entry]
fn main() {
    plain_write();
    BLIT.method_write();
}
'''


def _hir(source: str):
    program = parse(source, '<test>')
    program = expand_macros(program)
    hir = HIRBuilder().build_program(program)
    # method_call_info is populated by the type checker, and the real pipeline
    # runs lint after typeck — so the test must too.
    TypeChecker(hir).check()
    return hir


def _c001_config():
    rule = from_config({
        "code": "C001",
        "kind": "reachability_forbidden_access",
        "message": "PPU register write unsafe during active rendering",
        "entry_points": ["main"],
        "forbid_addrs": [0x2118],
    })
    return LintConfig(enabled_codes={"C001"}, custom_rules=[rule])


def test_method_body_is_reachable_from_main():
    graph = build_call_graph(_hir(SOURCE))
    reachable = graph.reachable_from(["main"])
    assert "plain_write" in reachable
    assert "Blitter__method_write" in reachable, (
        "method call did not create a call-graph edge; reachable set was "
        f"{sorted(reachable)}"
    )


def test_forbidden_write_flagged_in_both_free_function_and_method():
    diags = run_lint(_hir(SOURCE), _c001_config())
    flagged_lines = sorted(
        d.source_loc.line for d in diags.diagnostics if d.code == "C001"
    )
    # One write in plain_write, one in Blitter::method_write.
    assert len(flagged_lines) == 2, (
        f"expected both writes flagged, got lines {flagged_lines}"
    )
